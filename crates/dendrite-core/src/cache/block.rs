//! Block types for KV cache.

use std::sync::atomic::{AtomicU32, Ordering};

/// Unique identifier for a block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

impl BlockId {
    /// Invalid/null block ID.
    pub const INVALID: BlockId = BlockId(u32::MAX);

    /// Check if this is a valid block ID.
    pub fn is_valid(&self) -> bool {
        *self != Self::INVALID
    }
}

impl From<u32> for BlockId {
    fn from(id: u32) -> Self {
        BlockId(id)
    }
}

impl From<BlockId> for u32 {
    fn from(id: BlockId) -> Self {
        id.0
    }
}

/// A single block in the KV cache.
///
/// Each block holds key-value tensors for a fixed number of tokens
/// (typically 16) across all layers.
#[derive(Debug)]
pub struct Block {
    /// Block identifier.
    id: BlockId,
    /// Reference count for copy-on-write.
    refcount: AtomicU32,
    /// Number of tokens currently stored in this block.
    num_tokens: u32,
    /// Maximum tokens per block.
    capacity: u32,
}

impl Block {
    /// Create a new block with the given ID and capacity.
    pub fn new(id: BlockId, capacity: u32) -> Self {
        Self {
            id,
            refcount: AtomicU32::new(1),
            num_tokens: 0,
            capacity,
        }
    }

    /// Get the block ID.
    pub fn id(&self) -> BlockId {
        self.id
    }

    /// Get current reference count.
    pub fn refcount(&self) -> u32 {
        self.refcount.load(Ordering::Acquire)
    }

    /// Increment reference count.
    pub fn inc_ref(&self) -> u32 {
        self.refcount.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// Decrement reference count, returns new count.
    pub fn dec_ref(&self) -> u32 {
        let prev = self.refcount.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(prev > 0, "decrement of zero refcount");
        prev - 1
    }

    /// Check if block is shared (refcount > 1).
    pub fn is_shared(&self) -> bool {
        self.refcount() > 1
    }

    /// Get number of tokens in this block.
    pub fn num_tokens(&self) -> u32 {
        self.num_tokens
    }

    /// Check if block is full.
    pub fn is_full(&self) -> bool {
        self.num_tokens >= self.capacity
    }

    /// Get remaining capacity.
    pub fn remaining(&self) -> u32 {
        self.capacity.saturating_sub(self.num_tokens)
    }

    /// Add tokens to this block.
    pub fn add_tokens(&mut self, count: u32) {
        self.num_tokens = (self.num_tokens + count).min(self.capacity);
    }

    /// Reset block for reuse.
    pub fn reset(&mut self) {
        self.refcount.store(1, Ordering::Release);
        self.num_tokens = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_id_validity() {
        let valid = BlockId(42);
        assert!(valid.is_valid());

        let invalid = BlockId::INVALID;
        assert!(!invalid.is_valid());
    }

    #[test]
    fn block_id_conversions() {
        let id: BlockId = 42u32.into();
        assert_eq!(id.0, 42);

        let val: u32 = id.into();
        assert_eq!(val, 42);
    }

    #[test]
    fn new_block_has_refcount_one() {
        let block = Block::new(BlockId(0), 16);
        assert_eq!(block.refcount(), 1);
        assert!(!block.is_shared());
    }

    #[test]
    fn inc_ref_increases_refcount() {
        let block = Block::new(BlockId(0), 16);
        assert_eq!(block.inc_ref(), 2);
        assert_eq!(block.refcount(), 2);
        assert!(block.is_shared());
    }

    #[test]
    fn dec_ref_decreases_refcount() {
        let block = Block::new(BlockId(0), 16);
        block.inc_ref();
        assert_eq!(block.dec_ref(), 1);
        assert_eq!(block.refcount(), 1);
        assert!(!block.is_shared());
    }

    #[test]
    fn new_block_is_empty() {
        let block = Block::new(BlockId(0), 16);
        assert_eq!(block.num_tokens(), 0);
        assert!(!block.is_full());
        assert_eq!(block.remaining(), 16);
    }

    #[test]
    fn add_tokens_updates_count() {
        let mut block = Block::new(BlockId(0), 16);
        block.add_tokens(5);
        assert_eq!(block.num_tokens(), 5);
        assert_eq!(block.remaining(), 11);
        assert!(!block.is_full());
    }

    #[test]
    fn block_becomes_full() {
        let mut block = Block::new(BlockId(0), 16);
        block.add_tokens(16);
        assert!(block.is_full());
        assert_eq!(block.remaining(), 0);
    }

    #[test]
    fn add_tokens_clamps_to_capacity() {
        let mut block = Block::new(BlockId(0), 16);
        block.add_tokens(100);
        assert_eq!(block.num_tokens(), 16);
    }

    #[test]
    fn reset_clears_block() {
        let mut block = Block::new(BlockId(0), 16);
        block.inc_ref();
        block.add_tokens(10);

        block.reset();

        assert_eq!(block.refcount(), 1);
        assert_eq!(block.num_tokens(), 0);
    }
}
