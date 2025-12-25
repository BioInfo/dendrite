//! Block table for logical-to-physical block mapping.

use super::BlockId;

/// Maps logical token positions to physical blocks.
///
/// Each sequence has its own block table that translates
/// logical block indices to physical block IDs in the pool.
#[derive(Debug, Clone)]
pub struct BlockTable {
    /// Physical block IDs, indexed by logical block index.
    blocks: Vec<BlockId>,
    /// Number of tokens in the sequence.
    num_tokens: usize,
    /// Tokens per block.
    tokens_per_block: usize,
}

impl BlockTable {
    /// Create a new empty block table.
    pub fn new(tokens_per_block: usize) -> Self {
        Self {
            blocks: Vec::new(),
            num_tokens: 0,
            tokens_per_block,
        }
    }

    /// Create a block table with pre-allocated capacity.
    pub fn with_capacity(tokens_per_block: usize, max_blocks: usize) -> Self {
        Self {
            blocks: Vec::with_capacity(max_blocks),
            num_tokens: 0,
            tokens_per_block,
        }
    }

    /// Get the number of logical blocks.
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Get the number of tokens.
    pub fn num_tokens(&self) -> usize {
        self.num_tokens
    }

    /// Get the physical block ID for a logical index.
    pub fn get(&self, logical_idx: usize) -> Option<BlockId> {
        self.blocks.get(logical_idx).copied()
    }

    /// Get the block containing a token position.
    pub fn block_for_token(&self, token_pos: usize) -> Option<BlockId> {
        let logical_idx = token_pos / self.tokens_per_block;
        self.get(logical_idx)
    }

    /// Append a new block to the table.
    pub fn push(&mut self, block_id: BlockId) {
        self.blocks.push(block_id);
    }

    /// Set the block at a logical index.
    pub fn set(&mut self, logical_idx: usize, block_id: BlockId) {
        if logical_idx >= self.blocks.len() {
            self.blocks.resize(logical_idx + 1, BlockId::INVALID);
        }
        self.blocks[logical_idx] = block_id;
    }

    /// Update token count.
    pub fn set_num_tokens(&mut self, count: usize) {
        self.num_tokens = count;
    }

    /// Increment token count.
    pub fn add_tokens(&mut self, count: usize) {
        self.num_tokens += count;
    }

    /// Get all block IDs as a slice.
    pub fn blocks(&self) -> &[BlockId] {
        &self.blocks
    }

    /// Check if we need a new block for more tokens.
    pub fn needs_new_block(&self) -> bool {
        if self.blocks.is_empty() {
            return true;
        }
        let tokens_in_last_block = self.num_tokens % self.tokens_per_block;
        tokens_in_last_block == 0 && self.num_tokens > 0
    }

    /// Fork this block table (shallow copy for CoW).
    pub fn fork(&self) -> Self {
        Self {
            blocks: self.blocks.clone(),
            num_tokens: self.num_tokens,
            tokens_per_block: self.tokens_per_block,
        }
    }

    /// Clear the block table.
    pub fn clear(&mut self) {
        self.blocks.clear();
        self.num_tokens = 0;
    }
}
