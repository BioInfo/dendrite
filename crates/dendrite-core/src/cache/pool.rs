//! Block pool management.

use super::{Block, BlockId};
use crate::error::{DendriteError, Result};
use parking_lot::Mutex;
use std::collections::VecDeque;

/// Pool of KV cache blocks with free list management.
#[derive(Debug)]
pub struct BlockPool {
    /// All blocks in the pool.
    blocks: Vec<Mutex<Block>>,
    /// Free block IDs.
    free_list: Mutex<VecDeque<BlockId>>,
    /// Tokens per block.
    #[allow(dead_code)]
    tokens_per_block: usize,
}

impl BlockPool {
    /// Create a new block pool.
    pub fn new(max_blocks: usize, tokens_per_block: usize) -> Result<Self> {
        let mut blocks = Vec::with_capacity(max_blocks);
        let mut free_list = VecDeque::with_capacity(max_blocks);

        for i in 0..max_blocks {
            let id = BlockId(i as u32);
            blocks.push(Mutex::new(Block::new(id, tokens_per_block as u32)));
            free_list.push_back(id);
        }

        Ok(Self {
            blocks,
            free_list: Mutex::new(free_list),
            tokens_per_block,
        })
    }

    /// Allocate a block from the pool.
    pub fn allocate(&self) -> Result<BlockId> {
        let mut free_list = self.free_list.lock();
        free_list
            .pop_front()
            .ok_or_else(|| DendriteError::OutOfMemory("no free blocks".into()))
    }

    /// Free a block back to the pool.
    pub fn free(&self, block_id: BlockId) -> Result<()> {
        if !self.is_valid(block_id) {
            return Err(DendriteError::InvalidBlock(block_id.0));
        }

        let mut block = self.blocks[block_id.0 as usize].lock();
        let new_refcount = block.dec_ref();

        if new_refcount == 0 {
            block.reset();
            drop(block);
            self.free_list.lock().push_back(block_id);
        }

        Ok(())
    }

    /// Increment reference count for sharing.
    pub fn share(&self, block_id: BlockId) -> Result<()> {
        if !self.is_valid(block_id) {
            return Err(DendriteError::InvalidBlock(block_id.0));
        }

        self.blocks[block_id.0 as usize].lock().inc_ref();
        Ok(())
    }

    /// Copy-on-write: if shared, allocate new block and copy.
    pub fn copy_on_write(&self, block_id: BlockId) -> Result<BlockId> {
        if !self.is_valid(block_id) {
            return Err(DendriteError::InvalidBlock(block_id.0));
        }

        let block = self.blocks[block_id.0 as usize].lock();
        if !block.is_shared() {
            return Ok(block_id);
        }
        drop(block);

        // Allocate new block
        let new_id = self.allocate()?;

        // TODO: Copy KV data from old block to new block
        // This requires access to the actual tensor data

        // Decrement old block refcount
        self.free(block_id)?;

        Ok(new_id)
    }

    /// Get number of free blocks.
    pub fn free_count(&self) -> usize {
        self.free_list.lock().len()
    }

    /// Check if a block ID is valid.
    fn is_valid(&self, block_id: BlockId) -> bool {
        (block_id.0 as usize) < self.blocks.len()
    }

    /// Get refcount for a block (for testing).
    #[cfg(test)]
    fn get_refcount(&self, block_id: BlockId) -> Option<u32> {
        if self.is_valid(block_id) {
            Some(self.blocks[block_id.0 as usize].lock().refcount())
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_pool_has_all_blocks_free() {
        let pool = BlockPool::new(10, 16).unwrap();
        assert_eq!(pool.free_count(), 10);
    }

    #[test]
    fn allocate_returns_valid_block() {
        let pool = BlockPool::new(10, 16).unwrap();
        let block_id = pool.allocate().unwrap();
        assert!(pool.is_valid(block_id));
        assert_eq!(pool.free_count(), 9);
    }

    #[test]
    fn allocate_exhausts_pool() {
        let pool = BlockPool::new(3, 16).unwrap();

        let _b1 = pool.allocate().unwrap();
        let _b2 = pool.allocate().unwrap();
        let _b3 = pool.allocate().unwrap();

        assert_eq!(pool.free_count(), 0);

        let result = pool.allocate();
        assert!(result.is_err());
    }

    #[test]
    fn free_returns_block_to_pool() {
        let pool = BlockPool::new(5, 16).unwrap();

        let block_id = pool.allocate().unwrap();
        assert_eq!(pool.free_count(), 4);

        pool.free(block_id).unwrap();
        assert_eq!(pool.free_count(), 5);
    }

    #[test]
    fn free_invalid_block_returns_error() {
        let pool = BlockPool::new(5, 16).unwrap();
        let invalid_id = BlockId(100);

        let result = pool.free(invalid_id);
        assert!(result.is_err());
    }

    #[test]
    fn share_increments_refcount() {
        let pool = BlockPool::new(5, 16).unwrap();
        let block_id = pool.allocate().unwrap();

        assert_eq!(pool.get_refcount(block_id), Some(1));

        pool.share(block_id).unwrap();
        assert_eq!(pool.get_refcount(block_id), Some(2));

        pool.share(block_id).unwrap();
        assert_eq!(pool.get_refcount(block_id), Some(3));
    }

    #[test]
    fn share_invalid_block_returns_error() {
        let pool = BlockPool::new(5, 16).unwrap();
        let invalid_id = BlockId(100);

        let result = pool.share(invalid_id);
        assert!(result.is_err());
    }

    #[test]
    fn free_shared_block_decrements_refcount() {
        let pool = BlockPool::new(5, 16).unwrap();
        let block_id = pool.allocate().unwrap();

        pool.share(block_id).unwrap();
        pool.share(block_id).unwrap();
        assert_eq!(pool.get_refcount(block_id), Some(3));
        assert_eq!(pool.free_count(), 4);

        // First free: refcount 3 -> 2, not returned to pool
        pool.free(block_id).unwrap();
        assert_eq!(pool.get_refcount(block_id), Some(2));
        assert_eq!(pool.free_count(), 4);

        // Second free: refcount 2 -> 1, not returned to pool
        pool.free(block_id).unwrap();
        assert_eq!(pool.get_refcount(block_id), Some(1));
        assert_eq!(pool.free_count(), 4);

        // Third free: refcount 1 -> 0, returned to pool
        pool.free(block_id).unwrap();
        assert_eq!(pool.free_count(), 5);
    }

    #[test]
    fn copy_on_write_returns_same_if_not_shared() {
        let pool = BlockPool::new(5, 16).unwrap();
        let block_id = pool.allocate().unwrap();

        let result = pool.copy_on_write(block_id).unwrap();
        assert_eq!(result, block_id);
        assert_eq!(pool.free_count(), 4); // No new allocation
    }

    #[test]
    fn copy_on_write_allocates_new_if_shared() {
        let pool = BlockPool::new(5, 16).unwrap();
        let block_id = pool.allocate().unwrap();

        pool.share(block_id).unwrap();
        assert_eq!(pool.get_refcount(block_id), Some(2));

        let new_id = pool.copy_on_write(block_id).unwrap();

        // Should get a different block
        assert_ne!(new_id, block_id);
        // Original block refcount decremented
        assert_eq!(pool.get_refcount(block_id), Some(1));
        // New block has refcount 1
        assert_eq!(pool.get_refcount(new_id), Some(1));
        // One block allocated, one freed (net: same free count - 1 for new allocation)
        assert_eq!(pool.free_count(), 3);
    }

    #[test]
    fn copy_on_write_invalid_block_returns_error() {
        let pool = BlockPool::new(5, 16).unwrap();
        let invalid_id = BlockId(100);

        let result = pool.copy_on_write(invalid_id);
        assert!(result.is_err());
    }

    #[test]
    fn copy_on_write_fails_when_pool_exhausted() {
        let pool = BlockPool::new(2, 16).unwrap();

        let b1 = pool.allocate().unwrap();
        let _b2 = pool.allocate().unwrap();

        pool.share(b1).unwrap();

        // Pool exhausted, CoW should fail
        let result = pool.copy_on_write(b1);
        assert!(result.is_err());
    }
}
