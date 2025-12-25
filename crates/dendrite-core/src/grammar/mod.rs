//! Grammar-constrained decoding.
//!
//! Integration with llguidance for structured output:
//! - JSON schema validation
//! - Regular expression constraints
//! - Context-free grammar support

mod constraint;
mod mask;

pub use constraint::{Grammar, GrammarConstraint};
pub use mask::TokenMask;
