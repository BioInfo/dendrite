//! Attention backend trait and reference implementation.
//!
//! This module provides:
//! - [`AttentionBackend`] - Trait for attention computation backends
//! - [`ReferenceBackend`] - CPU reference implementation for testing
//! - [`AttentionConfig`] - Configuration for attention computation

use crate::cache::BlockTable;
use crate::error::Result;
use async_trait::async_trait;
use candle_core::{Device, Tensor};
use parking_lot::RwLock;
use std::collections::HashMap;

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

    /// Number of query heads per KV head (for GQA).
    pub fn num_queries_per_kv(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self::new(8, 8, 64) // 8 heads, 64 dim
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

/// Reference CPU implementation for testing.
///
/// This implementation stores KV cache in memory and performs
/// standard scaled dot-product attention. It's not optimized
/// for performance but is useful for correctness testing.
///
/// # Example
///
/// ```rust,ignore
/// use dendrite_core::attention::{ReferenceBackend, AttentionConfig, AttentionBackend};
///
/// let backend = ReferenceBackend::new();
/// let config = AttentionConfig::new(8, 8, 64);
///
/// // Prefill
/// let output = backend.prefill(&query, &key, &value, &block_table, &config).await?;
/// ```
#[derive(Debug, Default)]
pub struct ReferenceBackend {
    /// In-memory KV cache storage: block_id -> (key_tensor, value_tensor)
    kv_cache: RwLock<HashMap<u32, (Tensor, Tensor)>>,
}

impl ReferenceBackend {
    /// Create a new reference backend.
    pub fn new() -> Self {
        Self {
            kv_cache: RwLock::new(HashMap::new()),
        }
    }

    /// Clear the KV cache.
    pub fn clear_cache(&self) {
        self.kv_cache.write().clear();
    }

    /// Get cached KV for a block.
    fn get_kv(&self, block_id: u32) -> Option<(Tensor, Tensor)> {
        self.kv_cache.read().get(&block_id).cloned()
    }

    /// Store KV for a block.
    fn store_kv(&self, block_id: u32, key: Tensor, value: Tensor) {
        self.kv_cache.write().insert(block_id, (key, value));
    }

    /// Create causal mask for attention.
    fn create_causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
        // Create causal mask manually: -inf above diagonal, 0 on and below
        let mut mask_data = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                if j > i {
                    mask_data[i * seq_len + j] = f32::NEG_INFINITY;
                }
            }
        }
        let mask = Tensor::from_slice(&mask_data, (seq_len, seq_len), device)?;
        Ok(mask)
    }
}

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
        // query: [batch, num_heads, seq_len, head_dim]
        // key/value: [batch, num_kv_heads, seq_len, head_dim]
        let scale = config.scale as f64;
        let device = query.device();

        // Handle GQA: repeat KV heads to match query heads
        let (key, value) = if config.num_heads != config.num_kv_heads {
            let repeat_factor = config.num_queries_per_kv();
            let key = key.repeat((1, repeat_factor, 1, 1))?;
            let value = value.repeat((1, repeat_factor, 1, 1))?;
            (key, value)
        } else {
            (key.clone(), value.clone())
        };

        // Q @ K^T -> [batch, num_heads, seq_len, seq_len]
        let attn_weights = query.matmul(&key.transpose(2, 3)?)?;
        let attn_weights = (attn_weights * scale)?;

        // Apply causal mask
        let seq_len = query.dims()[2];
        let causal_mask = Self::create_causal_mask(seq_len, device)?;
        let causal_mask = causal_mask.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, seq, seq]
        let causal_mask = causal_mask.broadcast_as(attn_weights.shape())?;
        let attn_weights = (attn_weights + causal_mask)?;

        // Softmax over last dimension
        let attn_weights = candle_nn::ops::softmax(&attn_weights, candle_core::D::Minus1)?;

        // @ V -> [batch, num_heads, seq_len, head_dim]
        let output = attn_weights.matmul(&value)?;

        Ok(output)
    }

    async fn decode(
        &self,
        query: &Tensor,
        block_table: &BlockTable,
        config: &AttentionConfig,
    ) -> Result<Tensor> {
        // query: [1, num_heads, 1, head_dim]
        // For decode, we need to gather K/V from the cached blocks
        let scale = config.scale as f64;
        let device = query.device();
        let blocks = block_table.blocks();

        if blocks.is_empty() {
            // No cached context, return zeros
            let dims = query.dims();
            return Ok(Tensor::zeros(dims, query.dtype(), device)?);
        }

        // Gather K/V from cache
        let mut keys = Vec::new();
        let mut values = Vec::new();

        for block_id in blocks {
            if let Some((k, v)) = self.get_kv(block_id.0) {
                keys.push(k);
                values.push(v);
            }
        }

        if keys.is_empty() {
            let dims = query.dims();
            return Ok(Tensor::zeros(dims, query.dtype(), device)?);
        }

        // Concatenate K/V across sequence dimension
        let key = Tensor::cat(&keys, 2)?;
        let value = Tensor::cat(&values, 2)?;

        // Handle GQA
        let (key, value) = if config.num_heads != config.num_kv_heads {
            let repeat_factor = config.num_queries_per_kv();
            let key = key.repeat((1, repeat_factor, 1, 1))?;
            let value = value.repeat((1, repeat_factor, 1, 1))?;
            (key, value)
        } else {
            (key, value)
        };

        // Q @ K^T
        let attn_weights = query.matmul(&key.transpose(2, 3)?)?;
        let attn_weights = (attn_weights * scale)?;

        // No causal mask needed for decode (attending to all past tokens)
        let attn_weights = candle_nn::ops::softmax(&attn_weights, candle_core::D::Minus1)?;

        // @ V
        let output = attn_weights.matmul(&value)?;

        Ok(output)
    }

    async fn decode_batch(
        &self,
        queries: &[Tensor],
        block_tables: &[&BlockTable],
        config: &AttentionConfig,
    ) -> Result<Vec<Tensor>> {
        // Simple implementation: process each query individually
        // A production implementation would batch these operations
        let mut outputs = Vec::with_capacity(queries.len());

        for (query, block_table) in queries.iter().zip(block_tables.iter()) {
            let output = self.decode(query, block_table, config).await?;
            outputs.push(output);
        }

        Ok(outputs)
    }

    async fn append_kv(
        &self,
        key: &Tensor,
        value: &Tensor,
        block_table: &BlockTable,
        position: usize,
    ) -> Result<()> {
        // Determine which block to store in
        let tokens_per_block = 16; // TODO: get from config
        let block_idx = position / tokens_per_block;

        if let Some(block_id) = block_table.get(block_idx) {
            self.store_kv(block_id.0, key.clone(), value.clone());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::BlockId;

    fn create_test_config() -> AttentionConfig {
        AttentionConfig::new(4, 4, 64)
    }

    fn create_dummy_tensor(dims: &[usize]) -> Tensor {
        Tensor::randn(0.0f32, 1.0, dims, &Device::Cpu).unwrap()
    }

    #[test]
    fn attention_config_defaults() {
        let config = AttentionConfig::default();
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.head_dim, 64);
        assert!(config.sliding_window.is_none());
    }

    #[test]
    fn attention_config_scale() {
        let config = AttentionConfig::new(8, 8, 64);
        let expected_scale = 1.0 / (64.0f32).sqrt();
        assert!((config.scale - expected_scale).abs() < 1e-6);
    }

    #[test]
    fn attention_config_sliding_window() {
        let config = AttentionConfig::new(8, 8, 64).with_sliding_window(4096);
        assert_eq!(config.sliding_window, Some(4096));
    }

    #[test]
    fn attention_config_gqa() {
        let config = AttentionConfig::new(32, 8, 128); // 32 query heads, 8 KV heads
        assert_eq!(config.num_queries_per_kv(), 4);
    }

    #[test]
    fn reference_backend_new() {
        let backend = ReferenceBackend::new();
        assert!(backend.kv_cache.read().is_empty());
    }

    #[test]
    fn reference_backend_store_kv() {
        let backend = ReferenceBackend::new();
        let key = create_dummy_tensor(&[1, 4, 16, 64]);
        let value = create_dummy_tensor(&[1, 4, 16, 64]);

        backend.store_kv(0, key.clone(), value.clone());

        let retrieved = backend.get_kv(0);
        assert!(retrieved.is_some());
    }

    #[test]
    fn reference_backend_clear_cache() {
        let backend = ReferenceBackend::new();
        let key = create_dummy_tensor(&[1, 4, 16, 64]);
        let value = create_dummy_tensor(&[1, 4, 16, 64]);

        backend.store_kv(0, key, value);
        assert!(!backend.kv_cache.read().is_empty());

        backend.clear_cache();
        assert!(backend.kv_cache.read().is_empty());
    }

    #[tokio::test]
    async fn reference_backend_prefill() {
        let backend = ReferenceBackend::new();
        let config = create_test_config();

        // [batch=1, heads=4, seq_len=8, head_dim=64]
        let query = create_dummy_tensor(&[1, 4, 8, 64]);
        let key = create_dummy_tensor(&[1, 4, 8, 64]);
        let value = create_dummy_tensor(&[1, 4, 8, 64]);
        let block_table = BlockTable::new(16);

        let output = backend
            .prefill(&query, &key, &value, &block_table, &config)
            .await
            .unwrap();

        // Output should have same shape as query
        assert_eq!(output.dims(), query.dims());
    }

    #[tokio::test]
    async fn reference_backend_prefill_gqa() {
        let backend = ReferenceBackend::new();
        let config = AttentionConfig::new(8, 2, 64); // GQA: 8 query, 2 KV

        // Query: [1, 8, 4, 64], Key/Value: [1, 2, 4, 64]
        let query = create_dummy_tensor(&[1, 8, 4, 64]);
        let key = create_dummy_tensor(&[1, 2, 4, 64]);
        let value = create_dummy_tensor(&[1, 2, 4, 64]);
        let block_table = BlockTable::new(16);

        let output = backend
            .prefill(&query, &key, &value, &block_table, &config)
            .await
            .unwrap();

        // Output should match query shape
        assert_eq!(output.dims(), query.dims());
    }

    #[tokio::test]
    async fn reference_backend_decode_empty_cache() {
        let backend = ReferenceBackend::new();
        let config = create_test_config();

        let query = create_dummy_tensor(&[1, 4, 1, 64]);
        let block_table = BlockTable::new(16);

        let output = backend.decode(&query, &block_table, &config).await.unwrap();

        // With empty cache, output should be zeros
        assert_eq!(output.dims(), query.dims());
    }

    #[tokio::test]
    async fn reference_backend_decode_with_cache() {
        let backend = ReferenceBackend::new();
        let config = create_test_config();

        // Store some KV in cache
        let key = create_dummy_tensor(&[1, 4, 16, 64]);
        let value = create_dummy_tensor(&[1, 4, 16, 64]);
        backend.store_kv(0, key, value);

        // Create block table pointing to block 0
        let mut block_table = BlockTable::new(16);
        block_table.push(BlockId(0));

        let query = create_dummy_tensor(&[1, 4, 1, 64]);
        let output = backend.decode(&query, &block_table, &config).await.unwrap();

        assert_eq!(output.dims(), query.dims());
    }

    #[tokio::test]
    async fn reference_backend_decode_batch() {
        let backend = ReferenceBackend::new();
        let config = create_test_config();

        // Store KV for 2 blocks
        backend.store_kv(0, create_dummy_tensor(&[1, 4, 16, 64]), create_dummy_tensor(&[1, 4, 16, 64]));
        backend.store_kv(1, create_dummy_tensor(&[1, 4, 16, 64]), create_dummy_tensor(&[1, 4, 16, 64]));

        let mut bt1 = BlockTable::new(16);
        bt1.push(BlockId(0));
        let mut bt2 = BlockTable::new(16);
        bt2.push(BlockId(1));

        let queries = vec![
            create_dummy_tensor(&[1, 4, 1, 64]),
            create_dummy_tensor(&[1, 4, 1, 64]),
        ];
        let block_tables: Vec<&BlockTable> = vec![&bt1, &bt2];

        let outputs = backend
            .decode_batch(&queries, &block_tables, &config)
            .await
            .unwrap();

        assert_eq!(outputs.len(), 2);
        for (output, query) in outputs.iter().zip(queries.iter()) {
            assert_eq!(output.dims(), query.dims());
        }
    }

    #[tokio::test]
    async fn reference_backend_append_kv() {
        let backend = ReferenceBackend::new();

        let key = create_dummy_tensor(&[1, 4, 1, 64]);
        let value = create_dummy_tensor(&[1, 4, 1, 64]);

        let mut block_table = BlockTable::new(16);
        block_table.push(BlockId(0));

        let result = backend.append_kv(&key, &value, &block_table, 0).await;
        assert!(result.is_ok());

        // KV should now be stored
        assert!(backend.get_kv(0).is_some());
    }
}
