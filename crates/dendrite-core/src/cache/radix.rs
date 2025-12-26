//! Radix tree for efficient prefix lookup.
//!
//! The RadixTree enables O(L) prefix matching where L is the length of the
//! query sequence. This allows efficient sharing of KV cache blocks when
//! multiple sequences share common prefixes.
//!
//! # Example
//!
//! ```
//! use dendrite_core::cache::RadixTree;
//!
//! let mut tree = RadixTree::new();
//!
//! // Insert sequences with their block indices
//! tree.insert(&[1, 2, 3, 4], 0);
//! tree.insert(&[1, 2, 3, 5], 1);
//! tree.insert(&[1, 2, 6, 7], 2);
//!
//! // Find longest matching prefix
//! let (match_len, block_idx) = tree.find_prefix(&[1, 2, 3, 4, 5]);
//! assert_eq!(match_len, 4); // Matches [1, 2, 3, 4]
//! assert_eq!(block_idx, Some(0));
//!
//! // Partial match
//! let (match_len, _) = tree.find_prefix(&[1, 2, 3]);
//! assert_eq!(match_len, 3); // Matches [1, 2, 3] but no exact entry
//! ```

use std::collections::HashMap;

/// A node in the radix tree.
#[derive(Debug, Clone)]
struct RadixNode {
    /// Children indexed by token.
    children: HashMap<u32, RadixNode>,
    /// Block index if this node represents a complete sequence.
    block_idx: Option<usize>,
    /// Number of sequences passing through this node.
    count: usize,
}

impl RadixNode {
    fn new() -> Self {
        Self {
            children: HashMap::new(),
            block_idx: None,
            count: 0,
        }
    }

    fn with_block(block_idx: usize) -> Self {
        Self {
            children: HashMap::new(),
            block_idx: Some(block_idx),
            count: 1,
        }
    }
}

/// Radix tree for efficient prefix lookup and sharing.
///
/// This data structure enables:
/// - O(L) insertion where L is sequence length
/// - O(L) prefix lookup
/// - O(1) common prefix identification
/// - Efficient memory sharing via block reuse
#[derive(Debug, Clone)]
pub struct RadixTree {
    /// Root node.
    root: RadixNode,
    /// Total number of sequences stored.
    num_sequences: usize,
    /// Total number of nodes in the tree.
    num_nodes: usize,
}

impl RadixTree {
    /// Create an empty radix tree.
    pub fn new() -> Self {
        Self {
            root: RadixNode::new(),
            num_sequences: 0,
            num_nodes: 1, // Root counts as one node
        }
    }

    /// Insert a token sequence with its associated block index.
    ///
    /// # Arguments
    ///
    /// * `tokens` - The token sequence to insert
    /// * `block_idx` - The block index associated with this sequence
    pub fn insert(&mut self, tokens: &[u32], block_idx: usize) {
        let mut node = &mut self.root;
        node.count += 1;

        for &token in tokens {
            node = node.children.entry(token).or_insert_with(|| {
                self.num_nodes += 1;
                RadixNode::new()
            });
            node.count += 1;
        }

        node.block_idx = Some(block_idx);
        self.num_sequences += 1;
    }

    /// Find the longest matching prefix for a query sequence.
    ///
    /// Returns the length of the longest match and the block index if found.
    ///
    /// # Arguments
    ///
    /// * `tokens` - The query token sequence
    ///
    /// # Returns
    ///
    /// A tuple of (match_length, Option<block_idx>).
    /// - `match_length` is how many tokens were matched
    /// - `block_idx` is the associated block if an exact prefix entry exists
    pub fn find_prefix(&self, tokens: &[u32]) -> (usize, Option<usize>) {
        let mut node = &self.root;
        let mut match_len = 0;
        let mut last_block = None;

        for &token in tokens {
            if let Some(next) = node.children.get(&token) {
                node = next;
                match_len += 1;
                if node.block_idx.is_some() {
                    last_block = node.block_idx;
                }
            } else {
                break;
            }
        }

        (match_len, last_block)
    }

    /// Find exact match for a token sequence.
    ///
    /// # Arguments
    ///
    /// * `tokens` - The token sequence to look up
    ///
    /// # Returns
    ///
    /// The block index if an exact match exists.
    pub fn find_exact(&self, tokens: &[u32]) -> Option<usize> {
        let mut node = &self.root;

        for &token in tokens {
            node = node.children.get(&token)?;
        }

        node.block_idx
    }

    /// Remove a token sequence from the tree.
    ///
    /// Returns the block index that was associated with the sequence.
    ///
    /// # Arguments
    ///
    /// * `tokens` - The token sequence to remove
    pub fn remove(&mut self, tokens: &[u32]) -> Option<usize> {
        if tokens.is_empty() {
            let result = self.root.block_idx.take();
            if result.is_some() {
                self.num_sequences -= 1;
                self.root.count -= 1;
            }
            return result;
        }

        // Navigate to the target node and collect the path
        let mut path: Vec<u32> = Vec::new();
        let mut node = &self.root;

        for &token in tokens {
            if node.children.contains_key(&token) {
                path.push(token);
                node = node.children.get(&token).unwrap();
            } else {
                return None;
            }
        }

        // Now perform the removal
        self.remove_at_path(&path)
    }

    fn remove_at_path(&mut self, path: &[u32]) -> Option<usize> {
        if path.is_empty() {
            return None;
        }

        // Navigate to parent of target
        let mut current = &mut self.root;
        for &token in &path[..path.len() - 1] {
            current.count -= 1;
            current = current.children.get_mut(&token).unwrap();
        }
        current.count -= 1;

        let last_token = path[path.len() - 1];
        let target = current.children.get_mut(&last_token)?;
        target.count -= 1;

        let result = target.block_idx.take();
        if result.is_some() {
            self.num_sequences -= 1;
        }

        // Clean up empty nodes (simplified - just remove the leaf if empty)
        if target.count == 0 && target.children.is_empty() {
            current.children.remove(&last_token);
            self.num_nodes -= 1;
        }

        result
    }

    /// Get statistics about the radix tree.
    pub fn stats(&self) -> RadixTreeStats {
        RadixTreeStats {
            num_sequences: self.num_sequences,
            num_nodes: self.num_nodes,
            memory_bytes: self.estimate_memory(),
        }
    }

    /// Estimate memory usage in bytes.
    fn estimate_memory(&self) -> usize {
        // Rough estimate: each node has HashMap overhead plus fields
        // HashMap: ~48 bytes + entries * (key + value + overhead)
        // RadixNode: ~80 bytes base
        self.num_nodes * 128
    }

    /// Number of sequences stored.
    pub fn len(&self) -> usize {
        self.num_sequences
    }

    /// Check if the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.num_sequences == 0
    }

    /// Clear all entries from the tree.
    pub fn clear(&mut self) {
        self.root = RadixNode::new();
        self.num_sequences = 0;
        self.num_nodes = 1;
    }

    /// Get all sequences that share a common prefix.
    ///
    /// Returns block indices for all sequences that start with the given prefix.
    pub fn find_all_with_prefix(&self, prefix: &[u32]) -> Vec<usize> {
        // Navigate to prefix node
        let mut node = &self.root;
        for &token in prefix {
            match node.children.get(&token) {
                Some(next) => node = next,
                None => return Vec::new(),
            }
        }

        // Collect all block indices under this node
        let mut results = Vec::new();
        self.collect_blocks(node, &mut results);
        results
    }

    fn collect_blocks(&self, node: &RadixNode, results: &mut Vec<usize>) {
        if let Some(idx) = node.block_idx {
            results.push(idx);
        }
        for child in node.children.values() {
            self.collect_blocks(child, results);
        }
    }
}

impl Default for RadixTree {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about a RadixTree.
#[derive(Debug, Clone, Copy)]
pub struct RadixTreeStats {
    /// Number of complete sequences stored.
    pub num_sequences: usize,
    /// Number of nodes in the tree.
    pub num_nodes: usize,
    /// Estimated memory usage in bytes.
    pub memory_bytes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_tree_is_empty() {
        let tree = RadixTree::new();
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
    }

    #[test]
    fn insert_and_find_exact() {
        let mut tree = RadixTree::new();
        tree.insert(&[1, 2, 3], 42);

        assert_eq!(tree.find_exact(&[1, 2, 3]), Some(42));
        assert_eq!(tree.find_exact(&[1, 2]), None);
        assert_eq!(tree.find_exact(&[1, 2, 3, 4]), None);
    }

    #[test]
    fn insert_multiple_sequences() {
        let mut tree = RadixTree::new();
        tree.insert(&[1, 2, 3], 0);
        tree.insert(&[1, 2, 4], 1);
        tree.insert(&[1, 5, 6], 2);

        assert_eq!(tree.len(), 3);
        assert_eq!(tree.find_exact(&[1, 2, 3]), Some(0));
        assert_eq!(tree.find_exact(&[1, 2, 4]), Some(1));
        assert_eq!(tree.find_exact(&[1, 5, 6]), Some(2));
    }

    #[test]
    fn find_prefix_returns_longest_match() {
        let mut tree = RadixTree::new();
        tree.insert(&[1, 2, 3], 0);
        tree.insert(&[1, 2, 3, 4, 5], 1);

        // Query matches both, should return info about longest prefix match
        let (match_len, block_idx) = tree.find_prefix(&[1, 2, 3, 4, 5, 6]);
        assert_eq!(match_len, 5);
        assert_eq!(block_idx, Some(1));

        // Partial query
        let (match_len, block_idx) = tree.find_prefix(&[1, 2, 3, 4]);
        assert_eq!(match_len, 4);
        assert_eq!(block_idx, Some(0)); // Last complete entry at [1,2,3]
    }

    #[test]
    fn find_prefix_with_no_match() {
        let mut tree = RadixTree::new();
        tree.insert(&[1, 2, 3], 0);

        let (match_len, block_idx) = tree.find_prefix(&[4, 5, 6]);
        assert_eq!(match_len, 0);
        assert_eq!(block_idx, None);
    }

    #[test]
    fn find_prefix_partial_match_no_block() {
        let mut tree = RadixTree::new();
        tree.insert(&[1, 2, 3, 4, 5], 0);

        // Query [1, 2, 3] - matches 3 tokens but no entry at that point
        let (match_len, block_idx) = tree.find_prefix(&[1, 2, 3]);
        assert_eq!(match_len, 3);
        assert_eq!(block_idx, None);
    }

    #[test]
    fn remove_sequence() {
        let mut tree = RadixTree::new();
        tree.insert(&[1, 2, 3], 42);
        assert_eq!(tree.len(), 1);

        let removed = tree.remove(&[1, 2, 3]);
        assert_eq!(removed, Some(42));
        assert_eq!(tree.len(), 0);
        assert_eq!(tree.find_exact(&[1, 2, 3]), None);
    }

    #[test]
    fn remove_preserves_other_sequences() {
        let mut tree = RadixTree::new();
        tree.insert(&[1, 2, 3], 0);
        tree.insert(&[1, 2, 4], 1);

        tree.remove(&[1, 2, 3]);

        assert_eq!(tree.find_exact(&[1, 2, 3]), None);
        assert_eq!(tree.find_exact(&[1, 2, 4]), Some(1));
    }

    #[test]
    fn remove_nonexistent_returns_none() {
        let mut tree = RadixTree::new();
        tree.insert(&[1, 2, 3], 42);

        let removed = tree.remove(&[4, 5, 6]);
        assert_eq!(removed, None);
    }

    #[test]
    fn find_all_with_prefix() {
        let mut tree = RadixTree::new();
        tree.insert(&[1, 2, 3], 0);
        tree.insert(&[1, 2, 4], 1);
        tree.insert(&[1, 2, 5], 2);
        tree.insert(&[1, 3, 6], 3);

        let results = tree.find_all_with_prefix(&[1, 2]);
        assert_eq!(results.len(), 3);
        assert!(results.contains(&0));
        assert!(results.contains(&1));
        assert!(results.contains(&2));
        assert!(!results.contains(&3));
    }

    #[test]
    fn find_all_with_prefix_no_match() {
        let mut tree = RadixTree::new();
        tree.insert(&[1, 2, 3], 0);

        let results = tree.find_all_with_prefix(&[4, 5]);
        assert!(results.is_empty());
    }

    #[test]
    fn clear_empties_tree() {
        let mut tree = RadixTree::new();
        tree.insert(&[1, 2, 3], 0);
        tree.insert(&[4, 5, 6], 1);
        assert_eq!(tree.len(), 2);

        tree.clear();
        assert!(tree.is_empty());
        assert_eq!(tree.find_exact(&[1, 2, 3]), None);
    }

    #[test]
    fn stats_reflect_tree_state() {
        let mut tree = RadixTree::new();
        tree.insert(&[1, 2, 3], 0);
        tree.insert(&[1, 2, 4], 1);

        let stats = tree.stats();
        assert_eq!(stats.num_sequences, 2);
        // Root + 1 + 2 + 3 + 4 = 5 nodes (1 and 2 are shared)
        assert!(stats.num_nodes >= 4); // At least 4 nodes
        assert!(stats.memory_bytes > 0);
    }

    #[test]
    fn empty_sequence() {
        let mut tree = RadixTree::new();
        tree.insert(&[], 0);

        // Empty sequence is stored at root
        let (match_len, _) = tree.find_prefix(&[1, 2]);
        assert_eq!(match_len, 0);
    }

    #[test]
    fn overwrite_existing_sequence() {
        let mut tree = RadixTree::new();
        tree.insert(&[1, 2, 3], 0);
        tree.insert(&[1, 2, 3], 1);

        // Last insert wins
        assert_eq!(tree.find_exact(&[1, 2, 3]), Some(1));
        // But count increases
        assert_eq!(tree.len(), 2);
    }

    #[test]
    fn deep_tree() {
        let mut tree = RadixTree::new();
        let long_seq: Vec<u32> = (0..100).collect();
        tree.insert(&long_seq, 42);

        assert_eq!(tree.find_exact(&long_seq), Some(42));

        let (match_len, _) = tree.find_prefix(&long_seq);
        assert_eq!(match_len, 100);
    }
}
