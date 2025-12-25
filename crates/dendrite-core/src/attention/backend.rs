//! Attention backend trait.

use crate::cache::BlockTable;
use crate::error::Result;
use async_trait::async_trait;
use candle_core::Tensor;

/// Configuration for attention computation.
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Number of query heads.
    pub num_heads: usize,
    /// Number of KV heads (for GQA).
    pub num_kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Softmax scale (1/sqrt(head_dim) by default).
    pub scale: f32,
    /// Sliding window size (None for full attention).
    pub sliding_window: Option<usize>,
}

impl AttentionConfig {
    /// Create a new attention config.
    pub fn new(num_heads: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            num_heads,
            num_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            sliding_window: None,
        }
    }

    /// Set sliding window size.
    pub fn with_sliding_window(mut self, window: usize) -> Self {
        self.sliding_window = Some(window);
        self
    }
}

/// Backend for attention computation.
#[async_trait]
pub trait AttentionBackend: Send + Sync {
    /// Compute prefill attention (parallel over sequence).
    async fn prefill(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        block_table: &BlockTable,
        config: &AttentionConfig,
    ) -> Result<Tensor>;

    /// Compute decode attention (single token).
    async fn decode(
        &self,
        query: &Tensor,
        block_table: &BlockTable,
        config: &AttentionConfig,
    ) -> Result<Tensor>;

    /// Compute batched decode attention.
    async fn decode_batch(
        &self,
        queries: &[Tensor],
        block_tables: &[&BlockTable],
        config: &AttentionConfig,
    ) -> Result<Vec<Tensor>>;

    /// Append KV to cache.
    async fn append_kv(
        &self,
        key: &Tensor,
        value: &Tensor,
        block_table: &BlockTable,
        position: usize,
    ) -> Result<()>;
}

/// Reference implementation for testing.
#[derive(Debug, Default)]
pub struct ReferenceBackend;

#[async_trait]
impl AttentionBackend for ReferenceBackend {
    async fn prefill(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        _block_table: &BlockTable,
        config: &AttentionConfig,
    ) -> Result<Tensor> {
        // Simple scaled dot-product attention
        let scale = config.scale as f64;

        // Q @ K^T
        let attn_weights = query.matmul(&key.transpose(2, 3)?)?;
        let attn_weights = (attn_weights * scale)?;

        // Causal mask would go here

        // Softmax
        let attn_weights = candle_nn::ops::softmax(&attn_weights, candle_core::D::Minus1)?;

        // @ V
        let output = attn_weights.matmul(value)?;

        Ok(output)
    }

    async fn decode(
        &self,
        _query: &Tensor,
        _block_table: &BlockTable,
        _config: &AttentionConfig,
    ) -> Result<Tensor> {
        todo!("reference decode not implemented")
    }

    async fn decode_batch(
        &self,
        _queries: &[Tensor],
        _block_tables: &[&BlockTable],
        _config: &AttentionConfig,
    ) -> Result<Vec<Tensor>> {
        todo!("reference decode_batch not implemented")
    }

    async fn append_kv(
        &self,
        _key: &Tensor,
        _value: &Tensor,
        _block_table: &BlockTable,
        _position: usize,
    ) -> Result<()> {
        // No-op for reference implementation
        Ok(())
    }
}
