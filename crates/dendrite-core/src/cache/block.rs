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
