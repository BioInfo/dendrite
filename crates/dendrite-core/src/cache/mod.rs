//! KV Cache management with copy-on-write semantics.
//!
//! This module implements a paged KV cache with:
//! - Fixed-size blocks (16 tokens per block)
//! - Reference counting for copy-on-write
//! - Block tables for logical-to-physical mapping

mod block;
mod block_table;
mod pool;

pub use block::{Block, BlockId};
pub use block_table::BlockTable;
pub use pool::BlockPool;

/// Number of tokens per KV cache block.
pub const TOKENS_PER_BLOCK: usize = 16;

/// KV Cache configuration.
#[derive(Debug, Clone)]
pub struct KvCacheConfig {
    /// Number of layers in the model.
    pub num_layers: usize,
    /// Number of KV heads.
    pub num_kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Maximum number of blocks in the pool.
    pub max_blocks: usize,
    /// Tokens per block.
    pub tokens_per_block: usize,
}

impl Default for KvCacheConfig {
    fn default() -> Self {
        Self {
            num_layers: 32,
            num_kv_heads: 8,
            head_dim: 128,
            max_blocks: 65536,
            tokens_per_block: TOKENS_PER_BLOCK,
        }
    }
}

/// Main KV Cache handle.
#[derive(Debug)]
pub struct KvCache {
    config: KvCacheConfig,
    pool: BlockPool,
}

impl KvCache {
    /// Create a new KV cache with the given configuration.
    pub fn new(config: KvCacheConfig) -> crate::Result<Self> {
        let pool = BlockPool::new(config.max_blocks, config.tokens_per_block)?;
        Ok(Self { config, pool })
    }

    /// Get the cache configuration.
    pub fn config(&self) -> &KvCacheConfig {
        &self.config
    }

    /// Allocate a new block.
    pub fn allocate_block(&mut self) -> crate::Result<BlockId> {
        self.pool.allocate()
    }

    /// Free a block (decrements refcount, frees if zero).
    pub fn free_block(&mut self, block_id: BlockId) -> crate::Result<()> {
        self.pool.free(block_id)
    }

    /// Increment reference count for copy-on-write.
    pub fn share_block(&mut self, block_id: BlockId) -> crate::Result<()> {
        self.pool.share(block_id)
    }

    /// Copy block for write (if refcount > 1).
    pub fn copy_on_write(&mut self, block_id: BlockId) -> crate::Result<BlockId> {
        self.pool.copy_on_write(block_id)
    }

    /// Get number of free blocks.
    pub fn free_blocks(&self) -> usize {
        self.pool.free_count()
    }

    /// Get total number of blocks.
    pub fn total_blocks(&self) -> usize {
        self.config.max_blocks
    }
}
