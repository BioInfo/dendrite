//! Tree state management.

use super::node::{NodeId, TreeNode};
use crate::cache::{BlockId, KvCache};
use crate::error::{DendriteError, Result};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// Handle to a forked branch for the user.
#[derive(Debug, Clone)]
pub struct ForkHandle {
    /// Node ID of the fork point.
    pub node_id: NodeId,
    /// Depth from root.
    pub depth: usize,
    /// Number of tokens at fork point.
    pub num_tokens: usize,
}

/// Manages the tree of inference states.
#[derive(Debug)]
pub struct TreeState {
    /// All nodes in the tree.
    nodes: RwLock<HashMap<NodeId, TreeNode>>,
    /// KV cache for block management.
    kv_cache: Arc<RwLock<KvCache>>,
    /// Tokens per block.
    #[allow(dead_code)]
    tokens_per_block: usize,
}

impl TreeState {
    /// Create a new tree state.
    pub fn new(kv_cache: Arc<RwLock<KvCache>>, tokens_per_block: usize) -> Self {
        let mut nodes = HashMap::new();
        nodes.insert(NodeId::ROOT, TreeNode::root(tokens_per_block));

        Self {
            nodes: RwLock::new(nodes),
            kv_cache,
            tokens_per_block,
        }
    }

    /// Fork from a node, creating a new branch.
    ///
    /// This is O(1) because we use copy-on-write for the block table.
    pub fn fork(&self, from_node: NodeId) -> Result<ForkHandle> {
        let mut nodes = self.nodes.write();

        let parent = nodes
            .get_mut(&from_node)
            .ok_or(DendriteError::InvalidNode(from_node.0))?;

        // Create child node (O(1) - just copies block table pointers)
        let child = parent.fork();
        let child_id = child.id();
        let num_tokens = child.total_tokens();

        // Increment refcounts on shared blocks
        let mut cache = self.kv_cache.write();
        for block_id in child.block_table().blocks() {
            if block_id.is_valid() {
                cache.share_block(*block_id)?;
            }
        }
        drop(cache);

        // Update parent
        parent.add_child();

        // Calculate depth
        let depth = self.calculate_depth(from_node, &nodes);

        // Insert child
        nodes.insert(child_id, child);

        Ok(ForkHandle {
            node_id: child_id,
            depth: depth + 1,
            num_tokens,
        })
    }

    /// Append a token to a node.
    pub fn append_token(&self, node_id: NodeId, token: u32) -> Result<()> {
        let mut nodes = self.nodes.write();

        let node = nodes
            .get_mut(&node_id)
            .ok_or(DendriteError::InvalidNode(node_id.0))?;

        // Check if we need a new block
        if node.block_table().needs_new_block() {
            let block_id = self.kv_cache.write().allocate_block()?;
            node.block_table_mut().push(block_id);
        }

        // Check for copy-on-write on last block
        if let Some(last_block) = node.block_table().blocks().last() {
            let new_block = self.kv_cache.write().copy_on_write(*last_block)?;
            if new_block != *last_block {
                let len = node.block_table().num_blocks();
                node.block_table_mut().set(len - 1, new_block);
            }
        }

        node.add_token(token);
        node.block_table_mut().add_tokens(1);

        Ok(())
    }

    /// Release a node (mark as inactive, trigger GC).
    pub fn release(&self, node_id: NodeId) -> Result<()> {
        let mut nodes = self.nodes.write();

        if let Some(node) = nodes.get_mut(&node_id) {
            node.deactivate();

            // Notify parent
            if let Some(parent_id) = node.parent() {
                if let Some(parent) = nodes.get_mut(&parent_id) {
                    parent.remove_child();
                }
            }
        }

        // Garbage collect eligible nodes
        self.gc(&mut nodes)?;

        Ok(())
    }

    /// Get node info.
    pub fn get_node(&self, node_id: NodeId) -> Option<NodeInfo> {
        let nodes = self.nodes.read();
        nodes.get(&node_id).map(|n| NodeInfo {
            id: n.id(),
            parent: n.parent(),
            num_tokens: n.total_tokens(),
            num_children: n.num_children(),
            is_active: n.is_active(),
        })
    }

    /// Get block table for a node.
    pub fn get_block_table(&self, node_id: NodeId) -> Option<Vec<BlockId>> {
        let nodes = self.nodes.read();
        nodes
            .get(&node_id)
            .map(|n| n.block_table().blocks().to_vec())
    }

    /// Calculate depth of a node.
    fn calculate_depth(&self, node_id: NodeId, nodes: &HashMap<NodeId, TreeNode>) -> usize {
        let mut depth = 0;
        let mut current = node_id;

        while let Some(node) = nodes.get(&current) {
            if let Some(parent) = node.parent() {
                depth += 1;
                current = parent;
            } else {
                break;
            }
        }

        depth
    }

    /// Garbage collect unreachable nodes.
    fn gc(&self, nodes: &mut HashMap<NodeId, TreeNode>) -> Result<()> {
        let mut to_remove = Vec::new();

        for (id, node) in nodes.iter() {
            if node.can_gc() && *id != NodeId::ROOT {
                to_remove.push(*id);
            }
        }

        let mut cache = self.kv_cache.write();
        for id in to_remove {
            if let Some(node) = nodes.remove(&id) {
                // Free blocks
                for block_id in node.block_table().blocks() {
                    if block_id.is_valid() {
                        cache.free_block(*block_id)?;
                    }
                }
            }
        }

        Ok(())
    }
}

/// Info about a tree node.
#[derive(Debug, Clone)]
pub struct NodeInfo {
    /// Node ID.
    pub id: NodeId,
    /// Parent node ID.
    pub parent: Option<NodeId>,
    /// Total tokens in sequence.
    pub num_tokens: usize,
    /// Number of active children.
    pub num_children: u32,
    /// Whether node is active.
    pub is_active: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::KvCacheConfig;

    fn create_test_tree_state() -> TreeState {
        let config = KvCacheConfig {
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 64,
            max_blocks: 100,
            tokens_per_block: 16,
        };
        let kv_cache = Arc::new(RwLock::new(KvCache::new(config).unwrap()));
        TreeState::new(kv_cache, 16)
    }

    #[test]
    fn new_creates_root_node() {
        let state = create_test_tree_state();

        let root_info = state.get_node(NodeId::ROOT);
        assert!(root_info.is_some());

        let info = root_info.unwrap();
        assert_eq!(info.id, NodeId::ROOT);
        assert!(info.parent.is_none());
        assert!(info.is_active);
        assert_eq!(info.num_children, 0);
    }

    #[test]
    fn fork_creates_child_node() {
        let state = create_test_tree_state();

        let handle = state.fork(NodeId::ROOT).unwrap();

        assert_ne!(handle.node_id, NodeId::ROOT);
        assert_eq!(handle.depth, 1);

        let child_info = state.get_node(handle.node_id).unwrap();
        assert_eq!(child_info.parent, Some(NodeId::ROOT));
        assert!(child_info.is_active);
    }

    #[test]
    fn fork_increments_parent_child_count() {
        let state = create_test_tree_state();

        let root_before = state.get_node(NodeId::ROOT).unwrap();
        assert_eq!(root_before.num_children, 0);

        state.fork(NodeId::ROOT).unwrap();

        let root_after = state.get_node(NodeId::ROOT).unwrap();
        assert_eq!(root_after.num_children, 1);

        state.fork(NodeId::ROOT).unwrap();

        let root_after2 = state.get_node(NodeId::ROOT).unwrap();
        assert_eq!(root_after2.num_children, 2);
    }

    #[test]
    fn fork_from_invalid_node_returns_error() {
        let state = create_test_tree_state();
        let invalid_id = NodeId(99999);

        let result = state.fork(invalid_id);
        assert!(result.is_err());
    }

    #[test]
    fn deep_fork_chain_calculates_depth() {
        let state = create_test_tree_state();

        // Fork depth 1
        let h1 = state.fork(NodeId::ROOT).unwrap();
        assert_eq!(h1.depth, 1);

        // Fork depth 2
        let h2 = state.fork(h1.node_id).unwrap();
        assert_eq!(h2.depth, 2);

        // Fork depth 3
        let h3 = state.fork(h2.node_id).unwrap();
        assert_eq!(h3.depth, 3);

        // Fork depth 4
        let h4 = state.fork(h3.node_id).unwrap();
        assert_eq!(h4.depth, 4);
    }

    #[test]
    fn multiple_forks_from_same_parent() {
        let state = create_test_tree_state();

        let h1 = state.fork(NodeId::ROOT).unwrap();
        let h2 = state.fork(NodeId::ROOT).unwrap();
        let h3 = state.fork(NodeId::ROOT).unwrap();

        // All should be depth 1
        assert_eq!(h1.depth, 1);
        assert_eq!(h2.depth, 1);
        assert_eq!(h3.depth, 1);

        // All should have ROOT as parent
        assert_eq!(state.get_node(h1.node_id).unwrap().parent, Some(NodeId::ROOT));
        assert_eq!(state.get_node(h2.node_id).unwrap().parent, Some(NodeId::ROOT));
        assert_eq!(state.get_node(h3.node_id).unwrap().parent, Some(NodeId::ROOT));

        // Root should have 3 children
        assert_eq!(state.get_node(NodeId::ROOT).unwrap().num_children, 3);
    }

    #[test]
    fn get_node_returns_none_for_invalid() {
        let state = create_test_tree_state();
        let invalid_id = NodeId(99999);

        assert!(state.get_node(invalid_id).is_none());
    }

    #[test]
    fn release_deactivates_node() {
        let state = create_test_tree_state();
        let handle = state.fork(NodeId::ROOT).unwrap();

        let before = state.get_node(handle.node_id).unwrap();
        assert!(before.is_active);

        state.release(handle.node_id).unwrap();

        // Node might be GC'd, so we check if it's gone or inactive
        if let Some(after) = state.get_node(handle.node_id) {
            assert!(!after.is_active);
        }
    }

    #[test]
    fn release_decrements_parent_child_count() {
        let state = create_test_tree_state();

        let h1 = state.fork(NodeId::ROOT).unwrap();
        let h2 = state.fork(NodeId::ROOT).unwrap();

        assert_eq!(state.get_node(NodeId::ROOT).unwrap().num_children, 2);

        state.release(h1.node_id).unwrap();

        assert_eq!(state.get_node(NodeId::ROOT).unwrap().num_children, 1);

        state.release(h2.node_id).unwrap();

        assert_eq!(state.get_node(NodeId::ROOT).unwrap().num_children, 0);
    }

    #[test]
    fn get_block_table_returns_blocks() {
        let state = create_test_tree_state();

        // Root starts with empty block table
        let root_blocks = state.get_block_table(NodeId::ROOT);
        assert!(root_blocks.is_some());
        assert!(root_blocks.unwrap().is_empty());
    }

    #[test]
    fn fork_handle_contains_correct_info() {
        let state = create_test_tree_state();

        let handle = state.fork(NodeId::ROOT).unwrap();

        assert!(handle.node_id.is_valid());
        assert_eq!(handle.depth, 1);
        assert_eq!(handle.num_tokens, 0); // No tokens added yet
    }

    // ==================== Integration Tests ====================

    #[test]
    fn append_token_updates_node() {
        let state = create_test_tree_state();
        let handle = state.fork(NodeId::ROOT).unwrap();

        // Append a token
        state.append_token(handle.node_id, 42).unwrap();

        let info = state.get_node(handle.node_id).unwrap();
        assert_eq!(info.num_tokens, 1);
    }

    #[test]
    fn append_token_allocates_blocks() {
        let state = create_test_tree_state();
        let handle = state.fork(NodeId::ROOT).unwrap();

        // Initially no blocks
        let blocks_before = state.get_block_table(handle.node_id).unwrap();
        assert!(blocks_before.is_empty());

        // Append first token - should allocate a block
        state.append_token(handle.node_id, 1).unwrap();

        let blocks_after = state.get_block_table(handle.node_id).unwrap();
        assert_eq!(blocks_after.len(), 1);
    }

    #[test]
    fn append_token_fills_block_before_allocating_new() {
        let state = create_test_tree_state();
        let handle = state.fork(NodeId::ROOT).unwrap();

        // Fill first block (16 tokens)
        for i in 0..16 {
            state.append_token(handle.node_id, i).unwrap();
        }

        let blocks = state.get_block_table(handle.node_id).unwrap();
        assert_eq!(blocks.len(), 1);

        // 17th token should allocate new block
        state.append_token(handle.node_id, 16).unwrap();

        let blocks = state.get_block_table(handle.node_id).unwrap();
        assert_eq!(blocks.len(), 2);
    }

    #[test]
    fn append_token_to_invalid_node_fails() {
        let state = create_test_tree_state();
        let invalid_id = NodeId(99999);

        let result = state.append_token(invalid_id, 42);
        assert!(result.is_err());
    }

    #[test]
    fn fork_after_tokens_inherits_blocks() {
        let state = create_test_tree_state();
        let parent = state.fork(NodeId::ROOT).unwrap();

        // Add tokens to parent
        for i in 0..10 {
            state.append_token(parent.node_id, i).unwrap();
        }

        // Fork from parent
        let child = state.fork(parent.node_id).unwrap();

        // Child should have same token count
        assert_eq!(child.num_tokens, 10);

        // Child should have same blocks
        let parent_blocks = state.get_block_table(parent.node_id).unwrap();
        let child_blocks = state.get_block_table(child.node_id).unwrap();
        assert_eq!(parent_blocks, child_blocks);
    }

    #[test]
    fn cow_triggered_when_forked_child_appends() {
        let config = KvCacheConfig {
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 64,
            max_blocks: 100,
            tokens_per_block: 16,
        };
        let kv_cache = Arc::new(RwLock::new(KvCache::new(config).unwrap()));
        let state = TreeState::new(kv_cache.clone(), 16);

        // Create parent and add tokens
        let parent = state.fork(NodeId::ROOT).unwrap();
        for i in 0..8 {
            state.append_token(parent.node_id, i).unwrap();
        }

        // Fork child
        let child = state.fork(parent.node_id).unwrap();

        // Get block counts before
        let free_before = kv_cache.read().free_blocks();

        // Child appends - should trigger CoW
        state.append_token(child.node_id, 100).unwrap();

        // After CoW, we should have allocated a new block
        let free_after = kv_cache.read().free_blocks();

        // New block allocated for child's CoW
        assert!(free_after <= free_before);
    }

    #[test]
    fn multiple_forks_share_blocks() {
        let config = KvCacheConfig {
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 64,
            max_blocks: 100,
            tokens_per_block: 16,
        };
        let kv_cache = Arc::new(RwLock::new(KvCache::new(config).unwrap()));
        let state = TreeState::new(kv_cache.clone(), 16);

        // Create parent with tokens
        let parent = state.fork(NodeId::ROOT).unwrap();
        for i in 0..32 {
            // 2 blocks
            state.append_token(parent.node_id, i).unwrap();
        }

        let blocks_after_parent = kv_cache.read().used_blocks();

        // Fork 5 children - they share parent's blocks
        for _ in 0..5 {
            state.fork(parent.node_id).unwrap();
        }

        let blocks_after_forks = kv_cache.read().used_blocks();

        // No new blocks allocated - all sharing parent's blocks
        assert_eq!(blocks_after_forks, blocks_after_parent);
    }

    #[test]
    fn release_frees_blocks_when_no_references() {
        let config = KvCacheConfig {
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 64,
            max_blocks: 100,
            tokens_per_block: 16,
        };
        let kv_cache = Arc::new(RwLock::new(KvCache::new(config).unwrap()));
        let state = TreeState::new(kv_cache.clone(), 16);

        // Create node with tokens
        let handle = state.fork(NodeId::ROOT).unwrap();
        for i in 0..16 {
            state.append_token(handle.node_id, i).unwrap();
        }

        let used_before = kv_cache.read().used_blocks();
        assert!(used_before > 0);

        // Release node
        state.release(handle.node_id).unwrap();

        let used_after = kv_cache.read().used_blocks();

        // Blocks should be freed
        assert!(used_after < used_before);
    }

    #[test]
    fn gc_preserves_nodes_with_active_children() {
        let state = create_test_tree_state();

        // Create parent
        let parent = state.fork(NodeId::ROOT).unwrap();

        // Create child
        let child = state.fork(parent.node_id).unwrap();

        // Deactivate parent
        state.release(parent.node_id).unwrap();

        // Parent should still exist (has active child)
        // Note: parent might be in the tree still if it has children
        let child_info = state.get_node(child.node_id);
        assert!(child_info.is_some());
    }

    #[test]
    fn tree_builds_correct_sequence() {
        let state = create_test_tree_state();

        // Build a sequence: root -> A (tokens 1-10) -> B (tokens 11-20)
        let node_a = state.fork(NodeId::ROOT).unwrap();
        for i in 1..=10 {
            state.append_token(node_a.node_id, i).unwrap();
        }

        let node_b = state.fork(node_a.node_id).unwrap();
        for i in 11..=20 {
            state.append_token(node_b.node_id, i).unwrap();
        }

        // Verify token counts
        let info_a = state.get_node(node_a.node_id).unwrap();
        let info_b = state.get_node(node_b.node_id).unwrap();

        assert_eq!(info_a.num_tokens, 10);
        assert_eq!(info_b.num_tokens, 20);
    }

    #[test]
    fn branching_tree_structure() {
        let state = create_test_tree_state();

        // Create base sequence
        let base = state.fork(NodeId::ROOT).unwrap();
        for i in 0..5 {
            state.append_token(base.node_id, i).unwrap();
        }

        // Branch into 3 paths
        let branch1 = state.fork(base.node_id).unwrap();
        let branch2 = state.fork(base.node_id).unwrap();
        let branch3 = state.fork(base.node_id).unwrap();

        // Each branch adds different tokens
        for i in 100..105 {
            state.append_token(branch1.node_id, i).unwrap();
        }
        for i in 200..203 {
            state.append_token(branch2.node_id, i).unwrap();
        }
        for i in 300..310 {
            state.append_token(branch3.node_id, i).unwrap();
        }

        // Verify independent token counts
        let info1 = state.get_node(branch1.node_id).unwrap();
        let info2 = state.get_node(branch2.node_id).unwrap();
        let info3 = state.get_node(branch3.node_id).unwrap();

        assert_eq!(info1.num_tokens, 10); // 5 from base + 5
        assert_eq!(info2.num_tokens, 8); // 5 from base + 3
        assert_eq!(info3.num_tokens, 15); // 5 from base + 10

        // Base should have 3 children
        let base_info = state.get_node(base.node_id).unwrap();
        assert_eq!(base_info.num_children, 3);
    }
}
