//! Bridge between Dendrite's Tokenizer and llguidance's TokEnv.
//!
//! Enables grammar-constrained decoding by connecting the HuggingFace
//! tokenizer to llguidance's token trie environment.
//!
//! # Usage
//!
//! ```ignore
//! use dendrite_core::grammar::tokenizer_bridge::build_tok_env;
//! use dendrite_core::grammar::{Grammar, to_llguidance, LlgConstraint, ParserFactory};
//! use dendrite_core::model::Tokenizer;
//!
//! let tokenizer = Tokenizer::from_dir("path/to/model")?;
//! let tok_env = build_tok_env(&tokenizer)?;
//! let factory = ParserFactory::new_simple(&tok_env)?;
//!
//! let schema = r#"{"type": "object", "properties": {"name": {"type": "string"}}}"#;
//! let grammar = to_llguidance(&Grammar::json_schema(schema)).unwrap();
//! let parser = factory.create_parser(grammar)?;
//! let mut constraint = LlgConstraint::new(parser);
//!
//! // At each decoding step:
//! let mask_bytes = constraint.compute_mask_ptr();
//! // Apply mask to logits, sample token, then:
//! constraint.commit_token(Some(sampled_token));
//! ```

use crate::error::Result;
use crate::model::Tokenizer;
use llguidance::toktrie::{ApproximateTokEnv, TokEnv, TokRxInfo, TokTrie};
use std::sync::Arc;

/// Build a `TokEnv` from a Dendrite `Tokenizer`.
///
/// Iterates the tokenizer vocabulary to build a TokTrie, then wraps
/// it in a `TokEnvWithTrie` for use with llguidance's `ParserFactory`.
///
/// # Arguments
///
/// * `tokenizer` - A loaded Dendrite tokenizer
///
/// # Returns
///
/// A `TokEnv` suitable for `ParserFactory::new_simple()`
pub fn build_tok_env(tokenizer: &Tokenizer) -> Result<TokEnv> {
    let vocab_size = tokenizer.vocab_size() as u32;
    let eos_token = tokenizer
        .eos_token_id()
        .unwrap_or(vocab_size.saturating_sub(1));

    let info = TokRxInfo::new(vocab_size, eos_token);

    // Build word byte sequences for each token
    let words: Vec<Vec<u8>> = (0..vocab_size)
        .map(|i| {
            // Get raw token bytes — id_to_token returns the string representation
            // including HF's byte-level BPE encoding (e.g. "Ġhello" for " hello")
            tokenizer
                .id_to_token(i)
                .map(|s| decode_hf_token_bytes(&s))
                .unwrap_or_default()
        })
        .collect();

    let trie = TokTrie::from(&info, &words);
    let env = ApproximateTokEnv::new(trie);
    Ok(Arc::new(env))
}

/// Decode a HuggingFace BPE token string to raw bytes.
///
/// HuggingFace byte-level BPE uses special Unicode characters to represent
/// bytes that can't appear in normal text:
/// - 'Ġ' (U+0120) represents a space (0x20)
/// - 'Ċ' (U+010A) represents a newline (0x0A)
/// - Other bytes in range 0x00-0x1F and 0x7F-0xFF use special characters
///
/// This converts back to raw bytes.
fn decode_hf_token_bytes(token: &str) -> Vec<u8> {
    // Handle special tokens (BOS, EOS, PAD, etc.)
    if token.starts_with('<') && token.ends_with('>') {
        // Return the raw UTF-8 bytes of the token string for special tokens
        return token.as_bytes().to_vec();
    }

    let mut bytes = Vec::with_capacity(token.len());
    for ch in token.chars() {
        match ch {
            // HF byte-level BPE mapping (GPT-2/LLaMA style)
            // Characters U+0100..U+0143 map to bytes 0x00..0x43
            // This covers the standard HF BPE byte encoding
            c if (c as u32) == 0x0120 => bytes.push(0x20), // Ġ -> space
            c if (c as u32) == 0x010A => bytes.push(0x0A), // newline variant
            c if (c as u32) >= 0x0100 && (c as u32) <= 0x0100 + 255 => {
                bytes.push((c as u32 - 0x0100) as u8);
            }
            // ASCII printable range maps directly
            c if (c as u8) < 128 && c.is_ascii() => {
                bytes.push(c as u8);
            }
            // Fallback: encode as UTF-8
            c => {
                let mut buf = [0u8; 4];
                let s = c.encode_utf8(&mut buf);
                bytes.extend_from_slice(s.as_bytes());
            }
        }
    }
    bytes
}

/// Apply an llguidance mask to a logit slice.
///
/// llguidance returns masks as a byte array where each bit represents
/// whether a token is allowed. This converts to logit masking.
///
/// # Arguments
///
/// * `logits` - Mutable slice of f32 logits (vocab_size elements)
/// * `mask_ptr` - Pointer to the llguidance mask bytes
/// * `vocab_size` - Number of tokens in vocabulary
///
/// # Safety
///
/// `mask_ptr` must point to a valid buffer of `vocab_size.div_ceil(8)` bytes.
pub fn apply_llg_mask(logits: &mut [f32], mask_bytes: &[u8]) {
    for (token_id, logit) in logits.iter_mut().enumerate() {
        let byte_idx = token_id / 8;
        let bit_idx = token_id % 8;
        if byte_idx < mask_bytes.len() {
            if (mask_bytes[byte_idx] >> bit_idx) & 1 == 0 {
                *logit = f32::NEG_INFINITY;
            }
        } else {
            *logit = f32::NEG_INFINITY;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_hf_token_ascii() {
        // Plain ASCII should pass through directly
        let bytes = decode_hf_token_bytes("hello");
        assert_eq!(bytes, b"hello");
    }

    #[test]
    fn decode_hf_token_space() {
        // Ġ (U+0120) should become space (0x20)
        let token = "\u{0120}hello"; // "Ġhello" = " hello" in BPE
        let bytes = decode_hf_token_bytes(token);
        assert_eq!(bytes[0], 0x20);
        assert_eq!(&bytes[1..], b"hello");
    }

    #[test]
    fn decode_hf_token_special() {
        // Special tokens pass through as UTF-8
        let bytes = decode_hf_token_bytes("<s>");
        assert_eq!(bytes, b"<s>");

        let bytes = decode_hf_token_bytes("<|end_of_text|>");
        assert_eq!(bytes, b"<|end_of_text|>");
    }

    #[test]
    fn apply_mask_blocks_tokens() {
        let mut logits = vec![1.0f32; 16];
        // Mask: first byte 0b00000001 = only token 0 allowed
        let mask_bytes = vec![0b00000001u8, 0u8];
        apply_llg_mask(&mut logits, &mask_bytes);

        assert!(logits[0].is_finite());
        for i in 1..16 {
            assert_eq!(logits[i], f32::NEG_INFINITY);
        }
    }

    #[test]
    fn apply_mask_allows_all() {
        let mut logits = vec![1.0f32; 8];
        let mask_bytes = vec![0xFFu8]; // all bits set
        apply_llg_mask(&mut logits, &mask_bytes);

        for logit in &logits {
            assert!(logit.is_finite());
        }
    }

    #[test]
    #[ignore = "requires model files"]
    fn build_tok_env_from_model() {
        use crate::model::Tokenizer;
        let tokenizer = Tokenizer::from_dir("/path/to/your/model").unwrap();
        let tok_env = build_tok_env(&tokenizer).unwrap();
        println!(
            "TokEnv built successfully, tokenizer vocab: {}",
            tokenizer.vocab_size()
        );
        assert!(tokenizer.vocab_size() > 0);
        drop(tok_env);
    }
}
