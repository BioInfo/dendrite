//! Model loading and inference.
//!
//! Provides abstractions for transformer models:
//! - Model configuration
//! - Weight loading
//! - Forward pass

mod config;
mod transformer;

pub use config::ModelConfig;
pub use transformer::Transformer;
