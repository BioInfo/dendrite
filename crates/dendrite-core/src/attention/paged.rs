//! Paged attention implementation.

use crate::cache::{BlockTable, KvCache};
use crate::error::Result;
use parking_lot::RwLock;
use std::sync::Arc;

/// Paged attention manager.
///
/// Coordinates between the KV cache and attention backend
/// for paged attention computation.
#[derive(Debug)]
pub struct PagedAttention {
    /// Shared KV cache.
    kv_cache: Arc<RwLock<KvCache>>,
    /// Maximum sequence length.
    max_seq_len: usize,
    /// Tokens per block.
    tokens_per_block: usize,
}

impl PagedAttention {
    /// Create a new paged attention manager.
    pub fn new(
        kv_cache: Arc<RwLock<KvCache>>,
        max_seq_len: usize,
        tokens_per_block: usize,
    ) -> Self {
        Self {
            kv_cache,
            max_seq_len,
            tokens_per_block,
        }
    }

    /// Allocate blocks for a sequence.
    pub fn allocate_sequence(&self, num_tokens: usize) -> Result<BlockTable> {
        let num_blocks = (num_tokens + self.tokens_per_block - 1) / self.tokens_per_block;
        let mut block_table = BlockTable::with_capacity(self.tokens_per_block, num_blocks);

        let mut cache = self.kv_cache.write();
        for _ in 0..num_blocks {
            let block_id = cache.allocate_block()?;
            block_table.push(block_id);
        }
        block_table.set_num_tokens(num_tokens);

        Ok(block_table)
    }

    /// Extend blocks for growing sequence.
    pub fn extend_sequence(&self, block_table: &mut BlockTable, new_tokens: usize) -> Result<()> {
        let current_tokens = block_table.num_tokens();
        let new_total = current_tokens + new_tokens;

        if new_total > self.max_seq_len {
            return Err(crate::error::DendriteError::OutOfMemory(format!(
                "sequence length {} exceeds maximum {}",
                new_total, self.max_seq_len
            )));
        }

        // Check if we need new blocks
        let current_blocks = block_table.num_blocks();
        let needed_blocks = (new_total + self.tokens_per_block - 1) / self.tokens_per_block;

        let mut cache = self.kv_cache.write();
        for _ in current_blocks..needed_blocks {
            let block_id = cache.allocate_block()?;
            block_table.push(block_id);
        }

        block_table.set_num_tokens(new_total);
        Ok(())
    }

    /// Free all blocks for a sequence.
    pub fn free_sequence(&self, block_table: &BlockTable) -> Result<()> {
        let mut cache = self.kv_cache.write();
        for block_id in block_table.blocks() {
            if block_id.is_valid() {
                cache.free_block(*block_id)?;
            }
        }
        Ok(())
    }

    /// Get KV cache statistics.
    pub fn cache_stats(&self) -> CacheStats {
        let cache = self.kv_cache.read();
        CacheStats {
            total_blocks: cache.total_blocks(),
            free_blocks: cache.free_blocks(),
            used_blocks: cache.total_blocks() - cache.free_blocks(),
            utilization: 1.0 - (cache.free_blocks() as f32 / cache.total_blocks() as f32),
        }
    }
}

/// KV cache statistics.
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Total blocks in pool.
    pub total_blocks: usize,
    /// Free blocks available.
    pub free_blocks: usize,
    /// Blocks currently in use.
    pub used_blocks: usize,
    /// Cache utilization (0.0 - 1.0).
    pub utilization: f32,
}
