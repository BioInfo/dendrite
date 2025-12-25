//! Model configuration.

use serde::{Deserialize, Serialize};

/// Configuration for a transformer model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Hidden dimension.
    pub hidden_size: usize,
    /// Intermediate dimension (FFN).
    pub intermediate_size: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Number of KV heads (for GQA).
    pub num_key_value_heads: usize,
    /// Number of layers.
    pub num_hidden_layers: usize,
    /// Maximum sequence length.
    pub max_position_embeddings: usize,
    /// RMS norm epsilon.
    pub rms_norm_eps: f64,
    /// Rope theta.
    pub rope_theta: f64,
    /// Head dimension (derived).
    #[serde(default)]
    pub head_dim: usize,
    /// Model architecture type.
    #[serde(default)]
    pub model_type: String,
}

impl ModelConfig {
    /// Calculate head dimension.
    pub fn head_dim(&self) -> usize {
        if self.head_dim > 0 {
            self.head_dim
        } else {
            self.hidden_size / self.num_attention_heads
        }
    }

    /// Get GQA ratio.
    pub fn gqa_ratio(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Load from JSON file.
    pub fn from_file(path: &std::path::Path) -> crate::Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&contents)?;
        Ok(config)
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        // Llama-3-8B-like defaults
        Self {
            vocab_size: 128256,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            num_hidden_layers: 32,
            max_position_embeddings: 8192,
            rms_norm_eps: 1e-5,
            rope_theta: 500000.0,
            head_dim: 128,
            model_type: "llama".to_string(),
        }
    }
}
