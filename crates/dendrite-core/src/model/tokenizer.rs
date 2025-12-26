//! Tokenizer integration for text-to-token and token-to-text conversion.
//!
//! Wraps the HuggingFace tokenizers library for use with Dendrite transformers.

use crate::error::{DendriteError, Result};
use std::path::Path;
use tokenizers::Tokenizer as HfTokenizer;

/// Tokenizer for encoding text to tokens and decoding tokens to text.
#[derive(Clone)]
pub struct Tokenizer {
    /// Underlying HuggingFace tokenizer.
    inner: HfTokenizer,
    /// BOS token ID.
    bos_token_id: Option<u32>,
    /// EOS token ID.
    eos_token_id: Option<u32>,
    /// PAD token ID.
    pad_token_id: Option<u32>,
}

impl Tokenizer {
    /// Load a tokenizer from a tokenizer.json file.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let inner = HfTokenizer::from_file(path.as_ref()).map_err(|e| {
            DendriteError::ModelError(format!("Failed to load tokenizer: {}", e))
        })?;

        // Try to get special token IDs
        let bos_token_id = inner.token_to_id("<s>").or_else(|| inner.token_to_id("<|begin_of_text|>"));
        let eos_token_id = inner.token_to_id("</s>").or_else(|| inner.token_to_id("<|end_of_text|>"));
        let pad_token_id = inner.token_to_id("<pad>").or_else(|| inner.token_to_id("<|pad|>"));

        Ok(Self {
            inner,
            bos_token_id,
            eos_token_id,
            pad_token_id,
        })
    }

    /// Load a tokenizer from a model directory.
    ///
    /// Looks for tokenizer.json in the directory.
    pub fn from_dir<P: AsRef<Path>>(model_dir: P) -> Result<Self> {
        let tokenizer_path = model_dir.as_ref().join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(DendriteError::ModelError(format!(
                "tokenizer.json not found in {}",
                model_dir.as_ref().display()
            )));
        }
        Self::from_file(tokenizer_path)
    }

    /// Encode text to token IDs.
    ///
    /// # Arguments
    ///
    /// * `text` - Text to encode
    /// * `add_bos` - Whether to prepend BOS token
    ///
    /// # Returns
    ///
    /// Vector of token IDs
    pub fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>> {
        let encoding = self.inner.encode(text, false).map_err(|e| {
            DendriteError::ModelError(format!("Failed to encode text: {}", e))
        })?;

        let mut ids: Vec<u32> = encoding.get_ids().to_vec();

        if add_bos {
            if let Some(bos) = self.bos_token_id {
                ids.insert(0, bos);
            }
        }

        Ok(ids)
    }

    /// Encode text with chat template formatting.
    ///
    /// For TinyLlama and similar chat models.
    pub fn encode_chat(&self, prompt: &str) -> Result<Vec<u32>> {
        // Simple chat format for TinyLlama
        let formatted = format!("<|user|>\n{}</s>\n<|assistant|>\n", prompt);
        self.encode(&formatted, true)
    }

    /// Decode token IDs to text.
    ///
    /// # Arguments
    ///
    /// * `ids` - Token IDs to decode
    /// * `skip_special` - Whether to skip special tokens
    ///
    /// # Returns
    ///
    /// Decoded text
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String> {
        self.inner.decode(ids, skip_special).map_err(|e| {
            DendriteError::ModelError(format!("Failed to decode tokens: {}", e))
        })
    }

    /// Decode a single token to text.
    pub fn decode_token(&self, id: u32) -> Result<String> {
        self.decode(&[id], false)
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Get BOS token ID.
    pub fn bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }

    /// Get EOS token ID.
    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }

    /// Get PAD token ID.
    pub fn pad_token_id(&self) -> Option<u32> {
        self.pad_token_id
    }

    /// Check if a token is a special token.
    pub fn is_special_token(&self, id: u32) -> bool {
        id == self.bos_token_id.unwrap_or(u32::MAX)
            || id == self.eos_token_id.unwrap_or(u32::MAX)
            || id == self.pad_token_id.unwrap_or(u32::MAX)
    }

    /// Token to string (for debugging).
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    /// String to token ID.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }
}

impl std::fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tokenizer")
            .field("vocab_size", &self.vocab_size())
            .field("bos_token_id", &self.bos_token_id)
            .field("eos_token_id", &self.eos_token_id)
            .field("pad_token_id", &self.pad_token_id)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests require a real tokenizer.json file
    // Run with: cargo test --package dendrite-core tokenizer -- --ignored

    #[test]
    #[ignore = "requires model files"]
    fn tokenizer_load() {
        let tokenizer = Tokenizer::from_dir("/home/bioinfo/models/tinyllama-1.1b").unwrap();
        println!("Vocab size: {}", tokenizer.vocab_size());
        println!("BOS: {:?}", tokenizer.bos_token_id());
        println!("EOS: {:?}", tokenizer.eos_token_id());
    }

    #[test]
    #[ignore = "requires model files"]
    fn tokenizer_encode_decode() {
        let tokenizer = Tokenizer::from_dir("/home/bioinfo/models/tinyllama-1.1b").unwrap();

        let text = "Hello, my name is";
        let tokens = tokenizer.encode(text, true).unwrap();
        println!("Text: {:?}", text);
        println!("Tokens: {:?}", tokens);

        let decoded = tokenizer.decode(&tokens, false).unwrap();
        println!("Decoded: {:?}", decoded);

        // Should contain the original text
        assert!(decoded.contains("Hello"));
    }

    #[test]
    #[ignore = "requires model files"]
    fn tokenizer_individual_tokens() {
        let tokenizer = Tokenizer::from_dir("/home/bioinfo/models/tinyllama-1.1b").unwrap();

        // TinyLlama tokens for "Hello, my name is" with BOS
        let tokens = vec![1, 15043, 29892, 590, 1024, 338];

        for &token in &tokens {
            let decoded = tokenizer.decode_token(token).unwrap();
            let token_str = tokenizer.id_to_token(token);
            println!("Token {}: {:?} -> {:?}", token, token_str, decoded);
        }
    }
}
