//! GPU attention backends.
//!
//! This module provides GPU-accelerated attention implementations:
//! - [`FlashAttnBackend`] - Uses candle-flash-attn for GPU attention
//! - [`FlashInferBackend`] - Placeholder for FlashInfer paged attention (FFI)

use crate::cache::BlockTable;
use crate::error::Result;
use async_trait::async_trait;
use candle_core::{Device, Tensor};
use parking_lot::RwLock;
use std::collections::HashMap;

use super::{AttentionBackend, AttentionConfig};

/// GPU-accelerated attention using candle-flash-attn.
///
/// This backend uses Flash Attention 2 kernels for efficient GPU attention.
/// It stores KV cache on GPU and uses standard flash attention (not paged).
///
/// For paged attention with O(1) fork support, use [`FlashInferBackend`].
#[derive(Debug)]
pub struct FlashAttnBackend {
    /// Device for computation.
    device: Device,
    /// In-memory KV cache storage on GPU.
    kv_cache: RwLock<HashMap<u32, (Tensor, Tensor)>>,
}

impl FlashAttnBackend {
    /// Create a new Flash Attention backend on the specified CUDA device.
    pub fn new(device_id: usize) -> Result<Self> {
        let device = Device::new_cuda(device_id)?;
        Ok(Self {
            device,
            kv_cache: RwLock::new(HashMap::new()),
        })
    }

    /// Get the device.
    pub fn device(&self) -> &Device {
        &self.device
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
}

#[async_trait]
impl AttentionBackend for FlashAttnBackend {
    async fn prefill(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        _block_table: &BlockTable,
        config: &AttentionConfig,
    ) -> Result<Tensor> {
        // Ensure tensors are on GPU
        let query = query.to_device(&self.device)?;
        let key = key.to_device(&self.device)?;
        let value = value.to_device(&self.device)?;

        // Flash attention expects [batch, seq_len, num_heads, head_dim]
        // Our tensors are [batch, num_heads, seq_len, head_dim]
        // Transpose to flash attention format
        let q = query.transpose(1, 2)?;
        let k = key.transpose(1, 2)?;
        let v = value.transpose(1, 2)?;

        // Handle GQA: flash_attn expects same number of heads or will broadcast
        let (k, v) = if config.num_heads != config.num_kv_heads {
            let repeat_factor = config.num_queries_per_kv();
            let k = k.repeat((1, 1, repeat_factor, 1))?;
            let v = v.repeat((1, 1, repeat_factor, 1))?;
            (k, v)
        } else {
            (k, v)
        };

        // Use candle-flash-attn
        let output = candle_flash_attn::flash_attn(&q, &k, &v, config.scale, true)?;

        // Transpose back to [batch, num_heads, seq_len, head_dim]
        let output = output.transpose(1, 2)?;

        Ok(output)
    }

    async fn decode(
        &self,
        query: &Tensor,
        block_table: &BlockTable,
        config: &AttentionConfig,
    ) -> Result<Tensor> {
        // Gather K/V from cache
        let blocks = block_table.blocks();

        if blocks.is_empty() {
            // No cached context, return zeros
            let dims = query.dims();
            return Ok(Tensor::zeros(dims, query.dtype(), &self.device)?);
        }

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
            return Ok(Tensor::zeros(dims, query.dtype(), &self.device)?);
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

        // Ensure on GPU
        let query = query.to_device(&self.device)?;

        // Transpose for flash attention
        let q = query.transpose(1, 2)?;
        let k = key.transpose(1, 2)?;
        let v = value.transpose(1, 2)?;

        // For decode, causal=false since we're attending to all cached tokens
        let output = candle_flash_attn::flash_attn(&q, &k, &v, config.scale, false)?;

        // Transpose back
        let output = output.transpose(1, 2)?;

        Ok(output)
    }

    async fn decode_batch(
        &self,
        queries: &[Tensor],
        block_tables: &[&BlockTable],
        config: &AttentionConfig,
    ) -> Result<Vec<Tensor>> {
        // Process each query - could be optimized with varlen flash attention
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
        let tokens_per_block = 16;
        let block_idx = position / tokens_per_block;

        if let Some(block_id) = block_table.get(block_idx) {
            let key = key.to_device(&self.device)?;
            let value = value.to_device(&self.device)?;
            self.store_kv(block_id.0, key, value);
        }

        Ok(())
    }
}

/// FlashInfer-based attention backend for paged attention.
///
/// This backend uses FlashInfer's paged attention kernels which support:
/// - O(1) fork latency via shared KV cache pages
/// - Efficient batch decode with paged KV cache
/// - Cascade attention for shared prefixes
///
/// # Status
///
/// This backend requires FlashInfer FFI bindings (not yet implemented).
/// Use [`FlashAttnBackend`] for basic GPU attention.
#[derive(Debug)]
pub struct FlashInferBackend {
    /// Device ordinal.
    device_id: usize,
}

impl FlashInferBackend {
    /// Create a new FlashInfer backend on the specified device.
    pub fn new(device_id: usize) -> Self {
        Self { device_id }
    }

    /// Get the device ID.
    pub fn device_id(&self) -> usize {
        self.device_id
    }
}

#[async_trait]
impl AttentionBackend for FlashInferBackend {
    async fn prefill(
        &self,
        _query: &Tensor,
        _key: &Tensor,
        _value: &Tensor,
        _block_table: &BlockTable,
        _config: &AttentionConfig,
    ) -> Result<Tensor> {
        todo!("FlashInfer prefill requires FFI bindings - use FlashAttnBackend for now")
    }

    async fn decode(
        &self,
        _query: &Tensor,
        _block_table: &BlockTable,
        _config: &AttentionConfig,
    ) -> Result<Tensor> {
        todo!("FlashInfer decode requires FFI bindings - use FlashAttnBackend for now")
    }

    async fn decode_batch(
        &self,
        _queries: &[Tensor],
        _block_tables: &[&BlockTable],
        _config: &AttentionConfig,
    ) -> Result<Vec<Tensor>> {
        todo!("FlashInfer decode_batch requires FFI bindings - use FlashAttnBackend for now")
    }

    async fn append_kv(
        &self,
        _key: &Tensor,
        _value: &Tensor,
        _block_table: &BlockTable,
        _position: usize,
    ) -> Result<()> {
        todo!("FlashInfer append_kv requires FFI bindings - use FlashAttnBackend for now")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flash_infer_backend_new() {
        let backend = FlashInferBackend::new(0);
        assert_eq!(backend.device_id(), 0);
    }
}
