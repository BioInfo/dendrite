//! Grammar constraint types.

use crate::error::Result;

/// Type of grammar constraint.
#[derive(Debug, Clone)]
pub enum Grammar {
    /// JSON schema constraint.
    JsonSchema(String),
    /// Regular expression constraint.
    Regex(String),
    /// Context-free grammar (EBNF).
    Cfg(String),
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

    /// Create a CFG constraint.
    pub fn cfg(grammar: impl Into<String>) -> Self {
        Grammar::Cfg(grammar.into())
    }
}

/// A compiled grammar constraint for token masking.
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
    /// Current position in grammar.
    position: usize,
    /// Whether we're in a valid state.
    is_valid: bool,
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
    pub fn compute_mask(&self) -> Result<super::TokenMask> {
        match &self.grammar {
            Grammar::None => Ok(super::TokenMask::allow_all(self.vocab_size)),
            Grammar::JsonSchema(_) => {
                // TODO: Integrate with llguidance
                Ok(super::TokenMask::allow_all(self.vocab_size))
            }
            Grammar::Regex(_) => {
                // TODO: Integrate with llguidance
                Ok(super::TokenMask::allow_all(self.vocab_size))
            }
            Grammar::Cfg(_) => {
                // TODO: Integrate with llguidance
                Ok(super::TokenMask::allow_all(self.vocab_size))
            }
        }
    }

    /// Update state with generated token.
    pub fn accept_token(&mut self, token: u32) -> Result<bool> {
        self.state.generated.push(token);
        self.state.position += 1;

        // TODO: Validate token against grammar
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
        // TODO: Check if we've reached an accepting state
        false
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
