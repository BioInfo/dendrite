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
}
