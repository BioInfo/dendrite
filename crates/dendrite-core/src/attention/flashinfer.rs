//! FlashInfer attention backend.
//!
//! High-performance attention kernels via FlashInfer FFI.

use crate::cache::BlockTable;
use crate::error::Result;
use async_trait::async_trait;
use candle_core::Tensor;

use super::{AttentionBackend, AttentionConfig};

/// FlashInfer-based attention backend for production use.
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
        todo!("FlashInfer prefill not yet implemented")
    }

    async fn decode(
        &self,
        _query: &Tensor,
        _block_table: &BlockTable,
        _config: &AttentionConfig,
    ) -> Result<Tensor> {
        todo!("FlashInfer decode not yet implemented")
    }

    async fn decode_batch(
        &self,
        _queries: &[Tensor],
        _block_tables: &[&BlockTable],
        _config: &AttentionConfig,
    ) -> Result<Vec<Tensor>> {
        todo!("FlashInfer decode_batch not yet implemented")
    }

    async fn append_kv(
        &self,
        _key: &Tensor,
        _value: &Tensor,
        _block_table: &BlockTable,
        _position: usize,
    ) -> Result<()> {
        todo!("FlashInfer append_kv not yet implemented")
    }
}
