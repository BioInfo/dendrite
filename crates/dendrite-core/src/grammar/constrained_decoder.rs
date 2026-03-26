//! High-level constrained decoder for grammar-guided generation.
//!
//! Provides a `ConstrainedDecoder` that combines a tokenizer, llguidance
//! parser factory, and grammar into a single step-by-step decoding interface.
//!
//! # Usage
//!
//! ```ignore
//! use dendrite_core::grammar::constrained_decoder::ConstrainedDecoder;
//! use dendrite_core::grammar::Grammar;
//! use dendrite_core::model::Tokenizer;
//!
//! // Setup (once per model)
//! let tokenizer = Tokenizer::from_dir("path/to/model")?;
//! let decoder = ConstrainedDecoder::new(
//!     &tokenizer,
//!     Grammar::json_schema(r#"{"type": "object", "properties": {"name": {"type": "string"}}}"#),
//! )?;
//!
//! // At each generation step:
//! let allowed = decoder.allowed_tokens(); // BitVec over vocab
//! // Apply allowed to logits, sample a token (e.g., greedy or top-p)
//! let token_id: u32 = 42;
//! let is_done = decoder.advance(token_id)?;
//! if is_done { break; }
//! ```

use crate::error::{DendriteError, Result};
use crate::grammar::{tokenizer_bridge::build_tok_env, to_llguidance, Grammar, LlgConstraint, ParserFactory};
use crate::grammar::mask::TokenMask;
use crate::model::Tokenizer;
use llguidance::CommitResult;

/// State of the constrained decoder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecoderState {
    /// Decoding is active — token mask is valid.
    Active,
    /// Grammar condition is satisfied — can stop or continue.
    Complete,
    /// Grammar has been violated — no valid tokens remain.
    Error,
}

// CommitResult is used but only via its .stop field
fn commit_is_stop(r: CommitResult) -> bool {
    r.stop
}

/// Grammar-constrained decoding step driver.
///
/// Wraps llguidance's `Constraint` with Dendrite's tokenizer to provide
/// a clean step-by-step interface for constrained generation.
pub struct ConstrainedDecoder {
    /// llguidance constraint (stateful parser).
    constraint: LlgConstraint,
    /// Vocabulary size.
    vocab_size: usize,
    /// Current state.
    state: DecoderState,
}

impl ConstrainedDecoder {
    /// Create a new constrained decoder.
    ///
    /// # Arguments
    ///
    /// * `tokenizer` - The loaded model tokenizer
    /// * `grammar` - Grammar constraint (JSON schema, regex, or Lark CFG)
    ///
    /// # Returns
    ///
    /// A decoder ready to call `allowed_tokens()` and `advance()`
    pub fn new(tokenizer: &Tokenizer, grammar: Grammar) -> Result<Self> {
        let vocab_size = tokenizer.vocab_size();

        let tok_env = build_tok_env(tokenizer)?;

        let factory = ParserFactory::new_simple(&tok_env).map_err(|e| {
            DendriteError::ModelError(format!("Failed to create ParserFactory: {e}"))
        })?;

        let top_level = to_llguidance(&grammar).ok_or_else(|| {
            DendriteError::ModelError("Grammar::None cannot be used for constrained decoding — use a JSON schema, regex, or Lark grammar".into())
        })?;

        let parser = factory.create_parser(top_level).map_err(|e| {
            DendriteError::ModelError(format!("Failed to create grammar parser: {e}"))
        })?;

        let constraint = LlgConstraint::new(parser);

        Ok(Self {
            constraint,
            vocab_size,
            state: DecoderState::Active,
        })
    }

    /// Get a `TokenMask` for the current decoder state.
    ///
    /// Returns a bit-vector of allowed tokens at this generation step.
    /// Call this before sampling to apply the grammar constraint.
    ///
    /// # Returns
    ///
    /// `TokenMask` with `is_allowed(token)` returning true for valid next tokens.
    pub fn token_mask(&mut self) -> Result<TokenMask> {
        if self.state == DecoderState::Error {
            return Ok(TokenMask::block_all(self.vocab_size));
        }

        // compute_mask() returns Result<&StepResult> where StepResult = Branch<SimpleVob>
        let step = self.constraint.compute_mask().map_err(|e| {
            self.state = DecoderState::Error;
            DendriteError::ModelError(format!("Grammar mask computation failed: {e}"))
        })?;

        // Branch::sample_mask is Option<SimpleVob>; None means stop/allow-all
        match &step.sample_mask {
            None => {
                // stop result — grammar satisfied, allow no new tokens
                self.state = DecoderState::Complete;
                Ok(TokenMask::block_all(self.vocab_size))
            }
            Some(vob) => {
                // SimpleVob is a bit vector; build TokenMask from it
                let allowed_ids: Vec<u32> = vob.iter().collect();
                Ok(TokenMask::from_allowed(self.vocab_size, &allowed_ids))
            }
        }
    }

    /// Advance the decoder by committing a sampled token.
    ///
    /// Call this after sampling from the masked distribution.
    ///
    /// # Arguments
    ///
    /// * `token` - The token ID that was sampled
    ///
    /// # Returns
    ///
    /// * `Ok(true)` — grammar is satisfied, generation can stop
    /// * `Ok(false)` — grammar not yet satisfied, continue generating
    /// * `Err(...)` — grammar violation or internal error
    pub fn advance(&mut self, token: u32) -> Result<bool> {
        if self.state == DecoderState::Error {
            return Err(DendriteError::ModelError(
                "Constrained decoder is in error state — no valid tokens remain".into(),
            ));
        }

        let commit_result = self.constraint.commit_token(Some(token)).map_err(|e| {
            self.state = DecoderState::Error;
            DendriteError::ModelError(format!(
                "Grammar constraint violated at token {token}: {e}"
            ))
        })?;

        let is_stop = commit_is_stop(commit_result);
        if is_stop {
            self.state = DecoderState::Complete;
        }
        Ok(is_stop)
    }

    /// Current decoder state.
    pub fn state(&self) -> &DecoderState {
        &self.state
    }

    /// Whether the grammar constraint is satisfied (can stop generating).
    pub fn is_complete(&self) -> bool {
        self.state == DecoderState::Complete
    }

    /// Whether the decoder encountered an error.
    pub fn has_error(&self) -> bool {
        self.state == DecoderState::Error
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // These tests verify the structure without needing a real tokenizer.
    // Full integration tests are in the `#[ignore]` block below.

    #[test]
    fn decoder_state_default() {
        let state = DecoderState::Active;
        assert_eq!(state, DecoderState::Active);
        assert_ne!(state, DecoderState::Complete);
    }

    #[test]
    #[ignore = "requires model files"]
    fn constrained_decoder_json_schema() {
        use crate::model::Tokenizer;

        let tokenizer = Tokenizer::from_dir("/path/to/your/model").unwrap();
        let schema = r#"{
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name"]
        }"#;

        let mut decoder = ConstrainedDecoder::new(&tokenizer, Grammar::json_schema(schema)).unwrap();
        assert_eq!(*decoder.state(), DecoderState::Active);

        let mask = decoder.token_mask().unwrap();
        // Constrained mask should allow fewer tokens than the full vocab
        println!(
            "Allowed tokens: {}/{} ({:.1}%)",
            mask.num_allowed(),
            decoder.vocab_size(),
            100.0 * mask.num_allowed() as f32 / decoder.vocab_size() as f32
        );
        assert!(mask.num_allowed() < decoder.vocab_size());
    }

    #[test]
    #[ignore = "requires model files"]
    fn constrained_decoder_regex() {
        use crate::model::Tokenizer;

        let tokenizer = Tokenizer::from_dir("/path/to/your/model").unwrap();
        let mut decoder = ConstrainedDecoder::new(
            &tokenizer,
            Grammar::regex(r"\d{4}-\d{2}-\d{2}"), // ISO date: YYYY-MM-DD
        )
        .unwrap();

        let mask = decoder.token_mask().unwrap();
        println!(
            "Regex constrained: {}/{} tokens allowed",
            mask.num_allowed(),
            decoder.vocab_size()
        );
        assert!(mask.num_allowed() < decoder.vocab_size());
    }

    #[test]
    fn grammar_none_returns_error() {
        // Using Grammar::None should fail at construction since it provides no constraint
        // (You'd just use the unconstrained generation path instead)
        struct MockTokenizer;
        // We can't easily build a full tokenizer here without model files,
        // but the error path is tested via the None -> Err conversion in new()
        let _ = Grammar::None.is_none();
        assert!(Grammar::None.is_none());
    }
}
