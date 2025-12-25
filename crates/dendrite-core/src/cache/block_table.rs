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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_creates_empty_table() {
        let table = BlockTable::new(16);
        assert_eq!(table.num_blocks(), 0);
        assert_eq!(table.num_tokens(), 0);
    }

    #[test]
    fn with_capacity_creates_empty_table_with_capacity() {
        let table = BlockTable::with_capacity(16, 10);
        assert_eq!(table.num_blocks(), 0);
        assert_eq!(table.num_tokens(), 0);
    }

    #[test]
    fn push_adds_blocks() {
        let mut table = BlockTable::new(16);

        table.push(BlockId(0));
        table.push(BlockId(5));
        table.push(BlockId(10));

        assert_eq!(table.num_blocks(), 3);
        assert_eq!(table.get(0), Some(BlockId(0)));
        assert_eq!(table.get(1), Some(BlockId(5)));
        assert_eq!(table.get(2), Some(BlockId(10)));
    }

    #[test]
    fn get_returns_none_for_invalid_index() {
        let table = BlockTable::new(16);
        assert_eq!(table.get(0), None);
        assert_eq!(table.get(100), None);
    }

    #[test]
    fn block_for_token_calculates_correct_index() {
        let mut table = BlockTable::new(16);
        table.push(BlockId(0));
        table.push(BlockId(1));
        table.push(BlockId(2));

        // Tokens 0-15 -> block 0
        assert_eq!(table.block_for_token(0), Some(BlockId(0)));
        assert_eq!(table.block_for_token(15), Some(BlockId(0)));

        // Tokens 16-31 -> block 1
        assert_eq!(table.block_for_token(16), Some(BlockId(1)));
        assert_eq!(table.block_for_token(31), Some(BlockId(1)));

        // Tokens 32-47 -> block 2
        assert_eq!(table.block_for_token(32), Some(BlockId(2)));

        // Token beyond blocks
        assert_eq!(table.block_for_token(48), None);
    }

    #[test]
    fn set_updates_existing_block() {
        let mut table = BlockTable::new(16);
        table.push(BlockId(0));
        table.push(BlockId(1));

        table.set(0, BlockId(99));
        assert_eq!(table.get(0), Some(BlockId(99)));
        assert_eq!(table.get(1), Some(BlockId(1)));
    }

    #[test]
    fn set_extends_table_if_needed() {
        let mut table = BlockTable::new(16);

        table.set(5, BlockId(42));

        assert_eq!(table.num_blocks(), 6);
        assert_eq!(table.get(0), Some(BlockId::INVALID));
        assert_eq!(table.get(4), Some(BlockId::INVALID));
        assert_eq!(table.get(5), Some(BlockId(42)));
    }

    #[test]
    fn add_tokens_increments_count() {
        let mut table = BlockTable::new(16);

        table.add_tokens(10);
        assert_eq!(table.num_tokens(), 10);

        table.add_tokens(5);
        assert_eq!(table.num_tokens(), 15);
    }

    #[test]
    fn set_num_tokens_updates_count() {
        let mut table = BlockTable::new(16);

        table.set_num_tokens(100);
        assert_eq!(table.num_tokens(), 100);

        table.set_num_tokens(50);
        assert_eq!(table.num_tokens(), 50);
    }

    #[test]
    fn needs_new_block_empty_table() {
        let table = BlockTable::new(16);
        assert!(table.needs_new_block());
    }

    #[test]
    fn needs_new_block_partial_block() {
        let mut table = BlockTable::new(16);
        table.push(BlockId(0));
        table.set_num_tokens(10); // 10 tokens in 16-token block

        assert!(!table.needs_new_block());
    }

    #[test]
    fn needs_new_block_exactly_full() {
        let mut table = BlockTable::new(16);
        table.push(BlockId(0));
        table.set_num_tokens(16); // Exactly full

        assert!(table.needs_new_block());
    }

    #[test]
    fn needs_new_block_multiple_blocks() {
        let mut table = BlockTable::new(16);
        table.push(BlockId(0));
        table.push(BlockId(1));
        table.set_num_tokens(20); // 16 + 4 tokens

        assert!(!table.needs_new_block());

        table.set_num_tokens(32); // Exactly 2 full blocks
        assert!(table.needs_new_block());
    }

    #[test]
    fn fork_creates_shallow_copy() {
        let mut table = BlockTable::new(16);
        table.push(BlockId(0));
        table.push(BlockId(1));
        table.set_num_tokens(20);

        let forked = table.fork();

        assert_eq!(forked.num_blocks(), 2);
        assert_eq!(forked.num_tokens(), 20);
        assert_eq!(forked.get(0), Some(BlockId(0)));
        assert_eq!(forked.get(1), Some(BlockId(1)));

        // Verify they share same block IDs (shallow copy)
        assert_eq!(forked.blocks(), table.blocks());
    }

    #[test]
    fn fork_is_independent() {
        let mut table = BlockTable::new(16);
        table.push(BlockId(0));
        table.set_num_tokens(10);

        let mut forked = table.fork();

        // Modify forked
        forked.push(BlockId(99));
        forked.set_num_tokens(20);

        // Original unchanged
        assert_eq!(table.num_blocks(), 1);
        assert_eq!(table.num_tokens(), 10);
    }

    #[test]
    fn clear_empties_table() {
        let mut table = BlockTable::new(16);
        table.push(BlockId(0));
        table.push(BlockId(1));
        table.set_num_tokens(32);

        table.clear();

        assert_eq!(table.num_blocks(), 0);
        assert_eq!(table.num_tokens(), 0);
    }

    #[test]
    fn blocks_returns_all_block_ids() {
        let mut table = BlockTable::new(16);
        table.push(BlockId(5));
        table.push(BlockId(10));
        table.push(BlockId(15));

        let blocks = table.blocks();
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0], BlockId(5));
        assert_eq!(blocks[1], BlockId(10));
        assert_eq!(blocks[2], BlockId(15));
    }
}
