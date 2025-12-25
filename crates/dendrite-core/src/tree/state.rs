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
