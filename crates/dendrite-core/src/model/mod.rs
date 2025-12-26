//! Model loading and inference.
//!
//! Provides abstractions for transformer models:
//! - Model configuration
//! - Weight loading
//! - Forward pass
//!
//! # Components
//!
//! - [`ModelConfig`] - Model hyperparameters (vocab, layers, heads, etc.)
//! - [`Transformer`] - Main model struct with forward pass
//! - [`TransformerLayer`] - Single decoder layer
//! - [`Attention`] - Grouped Query Attention
//! - [`RmsNorm`] - RMS Layer Normalization
//! - [`RotaryEmbedding`] - Rotary Position Embeddings (RoPE)
//! - [`SwiGluMlp`] - SwiGLU MLP block
//! - [`WeightLoader`] - SafeTensors weight loading

mod config;
mod golden;
mod kv_cache;
mod layer;
mod loader;
mod mlp;
mod rmsnorm;
mod rope;
mod tokenizer;
mod transformer;

pub use config::ModelConfig;
pub use golden::{GoldenCase, GoldenResult, GoldenSummary, GoldenTestHarness, GoldenTestable};
pub use kv_cache::{KvCache, LayerCache};
pub use layer::{create_causal_mask, Attention, TransformerLayer};
pub use loader::{map_hf_name, WeightLoader};
pub use mlp::SwiGluMlp;
pub use rmsnorm::RmsNorm;
pub use rope::RotaryEmbedding;
pub use tokenizer::Tokenizer;
pub use transformer::Transformer;
