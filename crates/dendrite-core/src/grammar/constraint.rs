//! Grammar constraint types.
//!
//! Integrates with llguidance for constrained decoding.
//!
//! llguidance provides grammar-constrained generation supporting:
//! - JSON Schema validation
//! - Regular expressions
//! - Context-free grammars (Lark format)
//!
//! # Integration Architecture
//!
//! The llguidance integration requires:
//! 1. A `TokEnv` (token environment) from the tokenizer
//! 2. A `ParserFactory` for creating token parsers
//! 3. A `Constraint` for computing masks and committing tokens
//!
//! For full integration, use `LlgConstraint` which wraps the llguidance API.
//! For simpler use cases without a tokenizer, use `GrammarConstraint`.

use crate::error::Result;

/// Type of grammar constraint.
#[derive(Debug, Clone)]
pub enum Grammar {
    /// JSON schema constraint.
    JsonSchema(String),
    /// Regular expression constraint.
    Regex(String),
    /// Context-free grammar (Lark format).
    Lark(String),
    /// No constraint (allow all tokens).
    None,
}

impl Grammar {
    /// Create a JSON schema constraint.
    pub fn json_schema(schema: impl Into<String>) -> Self {
        Grammar::JsonSchema(schema.into())
    }

    /// Create a regex constraint.
    pub fn regex(pattern: impl Into<String>) -> Self {
        Grammar::Regex(pattern.into())
    }

    /// Create a Lark CFG constraint.
    pub fn lark(grammar: impl Into<String>) -> Self {
        Grammar::Lark(grammar.into())
    }

    /// Check if this is a no-constraint grammar.
    pub fn is_none(&self) -> bool {
        matches!(self, Grammar::None)
    }
}

/// A grammar constraint for token masking (without llguidance).
///
/// This is a lightweight constraint that tracks state without
/// full llguidance integration. For production use with actual
/// grammar enforcement, use `LlgConstraint`.
pub struct GrammarConstraint {
    /// The grammar specification.
    grammar: Grammar,
    /// Current parser state.
    state: GrammarState,
    /// Vocabulary size.
    vocab_size: usize,
}

/// Internal parser state.
#[derive(Debug, Clone, Default)]
struct GrammarState {
    /// Tokens generated so far.
    generated: Vec<u32>,
    /// Whether we're in a valid state.
    is_valid: bool,
    /// Whether generation is complete.
    is_done: bool,
}

impl GrammarConstraint {
    /// Create a new grammar constraint.
    pub fn new(grammar: Grammar, vocab_size: usize) -> Result<Self> {
        Ok(Self {
            grammar,
            state: GrammarState {
                is_valid: true,
                ..Default::default()
            },
            vocab_size,
        })
    }

    /// Get the grammar type.
    pub fn grammar(&self) -> &Grammar {
        &self.grammar
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Compute token mask for current state.
    ///
    /// Note: Without llguidance integration, this always allows all tokens.
    /// Use `LlgConstraint` for actual grammar enforcement.
    pub fn compute_mask(&self) -> Result<super::TokenMask> {
        // Without full llguidance integration, allow all tokens
        // This is a placeholder - actual enforcement requires LlgConstraint
        Ok(super::TokenMask::allow_all(self.vocab_size))
    }

    /// Update state with generated token.
    pub fn accept_token(&mut self, token: u32) -> Result<bool> {
        self.state.generated.push(token);
        Ok(self.state.is_valid)
    }

    /// Reset to initial state.
    pub fn reset(&mut self) {
        self.state = GrammarState {
            is_valid: true,
            ..Default::default()
        };
    }

    /// Check if generation is complete according to grammar.
    pub fn is_complete(&self) -> bool {
        self.state.is_done
    }

    /// Check if constraint is in a valid state.
    pub fn is_valid(&self) -> bool {
        self.state.is_valid
    }

    /// Fork the constraint state (for tree search).
    pub fn fork(&self) -> Self {
        Self {
            grammar: self.grammar.clone(),
            state: self.state.clone(),
            vocab_size: self.vocab_size,
        }
    }
}

// Re-export llguidance types for integration
pub use llguidance::api::{ParserLimits, TopLevelGrammar};
pub use llguidance::{Constraint as LlgConstraint, ParserFactory, TokenParser};

/// Convert a Grammar to llguidance TopLevelGrammar.
pub fn to_llguidance(grammar: &Grammar) -> Option<TopLevelGrammar> {
    match grammar {
        Grammar::None => None,
        Grammar::JsonSchema(schema) => {
            let value: serde_json::Value = serde_json::from_str(schema).ok()?;
            Some(TopLevelGrammar::from_json_schema(value))
        }
        Grammar::Regex(pattern) => Some(TopLevelGrammar::from_regex(pattern)),
        Grammar::Lark(grammar) => Some(TopLevelGrammar::from_lark(grammar.clone())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grammar_none() {
        let constraint = GrammarConstraint::new(Grammar::None, 1000).unwrap();
        let mask = constraint.compute_mask().unwrap();
        assert_eq!(mask.num_allowed(), 1000);
    }

    #[test]
    fn grammar_json_schema_without_tokenizer() {
        let schema = r#"{"type": "string"}"#;
        let constraint = GrammarConstraint::new(Grammar::json_schema(schema), 1000).unwrap();
        let mask = constraint.compute_mask().unwrap();
        assert_eq!(mask.num_allowed(), 1000);
    }

    #[test]
    fn grammar_regex_without_tokenizer() {
        let constraint = GrammarConstraint::new(Grammar::regex("[a-z]+"), 1000).unwrap();
        let mask = constraint.compute_mask().unwrap();
        assert_eq!(mask.num_allowed(), 1000);
    }

    #[test]
    fn grammar_lark_without_tokenizer() {
        let lark = r#"start: "hello" | "world""#;
        let constraint = GrammarConstraint::new(Grammar::lark(lark), 1000).unwrap();
        let mask = constraint.compute_mask().unwrap();
        assert_eq!(mask.num_allowed(), 1000);
    }

    #[test]
    fn grammar_fork() {
        let mut constraint = GrammarConstraint::new(Grammar::None, 1000).unwrap();
        constraint.accept_token(42).unwrap();

        let forked = constraint.fork();
        assert_eq!(forked.state.generated, vec![42]);
    }

    #[test]
    fn to_llguidance_json_schema() {
        let grammar = Grammar::json_schema(r#"{"type": "string"}"#);
        let llg = to_llguidance(&grammar);
        assert!(llg.is_some());
    }

    #[test]
    fn to_llguidance_regex() {
        let grammar = Grammar::regex("[a-z]+");
        let llg = to_llguidance(&grammar);
        assert!(llg.is_some());
    }

    #[test]
    fn to_llguidance_lark() {
        let grammar = Grammar::lark(r#"start: "hello""#);
        let llg = to_llguidance(&grammar);
        assert!(llg.is_some());
    }

    #[test]
    fn to_llguidance_none() {
        let grammar = Grammar::None;
        let llg = to_llguidance(&grammar);
        assert!(llg.is_none());
    }
}
