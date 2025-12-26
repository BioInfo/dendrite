//! KV Cache management with copy-on-write semantics.
//!
//! This module implements a paged KV cache with:
//! - Fixed-size blocks (16 tokens per block)
//! - Reference counting for copy-on-write
//! - Block tables for logical-to-physical mapping
//!
//! # Architecture
//!
//! The KV cache is organized into three layers:
//!
//! 1. **Block Pool** - Pre-allocated pool of physical blocks
//! 2. **Block Table** - Per-sequence logical-to-physical mapping
//! 3. **KV Cache** - High-level API for allocation and CoW
//!
//! # Example
//!
//! ```rust
//! use dendrite_core::cache::{KvCache, KvCacheConfig};
//!
//! // Configure the cache
//! let config = KvCacheConfig {
//!     num_layers: 32,
//!     num_kv_heads: 8,
//!     head_dim: 128,
//!     max_blocks: 1000,
//!     tokens_per_block: 16,
//! };
//!
//! // Create the cache
//! let mut cache = KvCache::new(config).unwrap();
//!
//! // Allocate a block
//! let block = cache.allocate_block().unwrap();
//!
//! // Share for copy-on-write
//! cache.share_block(block).unwrap();
//!
//! // Copy-on-write returns new block if shared
//! let new_block = cache.copy_on_write(block).unwrap();
//! assert_ne!(block, new_block);
//!
//! // Free blocks
//! cache.free_block(block).unwrap();
//! cache.free_block(new_block).unwrap();
//! ```

mod block;
mod block_table;
mod pool;
mod radix;

pub use block::{Block, BlockId};
pub use block_table::BlockTable;
pub use pool::BlockPool;
pub use radix::{RadixTree, RadixTreeStats};

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

    /// Get used blocks count.
    pub fn used_blocks(&self) -> usize {
        self.config.max_blocks - self.pool.free_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_cache(max_blocks: usize) -> KvCache {
        let config = KvCacheConfig {
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 64,
            max_blocks,
            tokens_per_block: 16,
        };
        KvCache::new(config).unwrap()
    }

    #[test]
    fn new_cache_has_all_blocks_free() {
        let cache = create_test_cache(100);
        assert_eq!(cache.free_blocks(), 100);
        assert_eq!(cache.used_blocks(), 0);
        assert_eq!(cache.total_blocks(), 100);
    }

    #[test]
    fn allocate_reduces_free_count() {
        let mut cache = create_test_cache(10);

        let _b1 = cache.allocate_block().unwrap();
        assert_eq!(cache.free_blocks(), 9);
        assert_eq!(cache.used_blocks(), 1);

        let _b2 = cache.allocate_block().unwrap();
        assert_eq!(cache.free_blocks(), 8);
        assert_eq!(cache.used_blocks(), 2);
    }

    #[test]
    fn free_returns_block_to_pool() {
        let mut cache = create_test_cache(10);

        let b1 = cache.allocate_block().unwrap();
        assert_eq!(cache.free_blocks(), 9);

        cache.free_block(b1).unwrap();
        assert_eq!(cache.free_blocks(), 10);
    }

    #[test]
    fn share_and_free_refcount_behavior() {
        let mut cache = create_test_cache(10);

        let b1 = cache.allocate_block().unwrap();
        assert_eq!(cache.free_blocks(), 9);

        // Share twice (refcount now 3)
        cache.share_block(b1).unwrap();
        cache.share_block(b1).unwrap();

        // Free once (refcount 2) - not returned
        cache.free_block(b1).unwrap();
        assert_eq!(cache.free_blocks(), 9);

        // Free again (refcount 1) - not returned
        cache.free_block(b1).unwrap();
        assert_eq!(cache.free_blocks(), 9);

        // Free last (refcount 0) - returned
        cache.free_block(b1).unwrap();
        assert_eq!(cache.free_blocks(), 10);
    }

    #[test]
    fn copy_on_write_unshared_returns_same() {
        let mut cache = create_test_cache(10);

        let b1 = cache.allocate_block().unwrap();
        let result = cache.copy_on_write(b1).unwrap();

        assert_eq!(result, b1);
        assert_eq!(cache.free_blocks(), 9);
    }

    #[test]
    fn copy_on_write_shared_allocates_new() {
        let mut cache = create_test_cache(10);

        let b1 = cache.allocate_block().unwrap();
        cache.share_block(b1).unwrap();

        let new_block = cache.copy_on_write(b1).unwrap();

        assert_ne!(new_block, b1);
        // Original still has refcount 1, new block allocated
        assert_eq!(cache.free_blocks(), 8);
    }

    #[test]
    fn config_accessible() {
        let cache = create_test_cache(50);
        let config = cache.config();

        assert_eq!(config.max_blocks, 50);
        assert_eq!(config.tokens_per_block, 16);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    prop_compose! {
        fn arb_cache_config()(
            max_blocks in 10usize..100,
            tokens_per_block in prop::sample::select(vec![8usize, 16, 32]),
        ) -> KvCacheConfig {
            KvCacheConfig {
                num_layers: 1,
                num_kv_heads: 1,
                head_dim: 64,
                max_blocks,
                tokens_per_block,
            }
        }
    }

    proptest! {
        /// Invariant: free_blocks + used_blocks == total_blocks
        #[test]
        fn invariant_block_count_consistency(
            config in arb_cache_config(),
            alloc_count in 0usize..50,
        ) {
            let mut cache = KvCache::new(config.clone()).unwrap();
            let max_alloc = alloc_count.min(config.max_blocks);

            let mut allocated = Vec::new();
            for _ in 0..max_alloc {
                if let Ok(block) = cache.allocate_block() {
                    allocated.push(block);
                }
            }

            // Invariant must hold
            prop_assert_eq!(
                cache.free_blocks() + cache.used_blocks(),
                cache.total_blocks()
            );
        }

        /// Invariant: allocate -> free cycle returns to original state
        #[test]
        fn invariant_allocate_free_cycle(
            config in arb_cache_config(),
            cycle_count in 1usize..20,
        ) {
            let mut cache = KvCache::new(config.clone()).unwrap();
            let initial_free = cache.free_blocks();

            for _ in 0..cycle_count {
                let block = cache.allocate_block().unwrap();
                cache.free_block(block).unwrap();
            }

            prop_assert_eq!(cache.free_blocks(), initial_free);
        }

        /// Invariant: share N times requires N+1 frees to return block
        #[test]
        fn invariant_refcount_matches_shares(
            config in arb_cache_config(),
            share_count in 0usize..10,
        ) {
            let mut cache = KvCache::new(config).unwrap();
            let initial_free = cache.free_blocks();

            let block = cache.allocate_block().unwrap();

            // Share N times (refcount = N+1)
            for _ in 0..share_count {
                cache.share_block(block).unwrap();
            }

            // Free N times - block not returned yet
            for _ in 0..share_count {
                cache.free_block(block).unwrap();
                prop_assert_eq!(cache.free_blocks(), initial_free - 1);
            }

            // Final free - block returned
            cache.free_block(block).unwrap();
            prop_assert_eq!(cache.free_blocks(), initial_free);
        }

        /// Invariant: CoW on unshared block is identity
        #[test]
        fn invariant_cow_unshared_identity(config in arb_cache_config()) {
            let mut cache = KvCache::new(config).unwrap();

            let block = cache.allocate_block().unwrap();
            let result = cache.copy_on_write(block).unwrap();

            prop_assert_eq!(result, block);
        }

        /// Invariant: CoW on shared block creates new block
        #[test]
        fn invariant_cow_shared_creates_new(
            config in arb_cache_config(),
        ) {
            let mut cache = KvCache::new(config).unwrap();

            let block = cache.allocate_block().unwrap();
            cache.share_block(block).unwrap();

            let new_block = cache.copy_on_write(block).unwrap();

            prop_assert_ne!(new_block, block);
        }
    }
}
