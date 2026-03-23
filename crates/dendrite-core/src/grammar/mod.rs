//! Grammar-constrained decoding.
//!
//! Integration with llguidance for structured output:
//! - JSON schema validation
//! - Regular expression constraints
//! - Context-free grammar support
//!
//! # Usage
//!
//! For simple use without tokenizer integration:
//! ```ignore
//! use dendrite_core::grammar::{Grammar, GrammarConstraint};
//!
//! let constraint = GrammarConstraint::new(Grammar::regex("[a-z]+"), 50000)?;
//! let mask = constraint.compute_mask()?;
//! ```
//!
//! For full llguidance integration with a tokenizer:
//! ```ignore
//! use dendrite_core::grammar::{Grammar, to_llguidance, ParserFactory, LlgConstraint};
//!
//! let grammar = to_llguidance(&Grammar::json_schema(schema))?;
//! let factory = ParserFactory::new_simple(&tok_env)?;
//! let parser = factory.create_parser(grammar)?;
//! let constraint = LlgConstraint::new(parser);
//! ```

mod constraint;
mod mask;
pub mod constrained_decoder;
pub mod tokenizer_bridge;

pub use constraint::{
    to_llguidance, Grammar, GrammarConstraint, LlgConstraint, ParserFactory, ParserLimits,
    TokenParser, TopLevelGrammar,
};
pub use constrained_decoder::ConstrainedDecoder;
pub use mask::TokenMask;
pub use tokenizer_bridge::{apply_llg_mask, build_tok_env};
