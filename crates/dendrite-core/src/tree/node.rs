//! Tree node representation.

use crate::cache::BlockTable;
use std::sync::atomic::{AtomicU64, Ordering};

/// Unique identifier for a tree node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u64);

impl NodeId {
    /// Root node ID.
    pub const ROOT: NodeId = NodeId(0);

    /// Invalid/null node ID.
    pub const INVALID: NodeId = NodeId(u64::MAX);

    /// Check if this is a valid node ID.
    pub fn is_valid(&self) -> bool {
        *self != Self::INVALID
    }
}

static NEXT_NODE_ID: AtomicU64 = AtomicU64::new(1);

impl NodeId {
    /// Generate a new unique node ID.
    pub fn new() -> Self {
        NodeId(NEXT_NODE_ID.fetch_add(1, Ordering::Relaxed))
    }
}

impl Default for NodeId {
    fn default() -> Self {
        Self::new()
    }
}

/// A node in the inference tree.
///
/// Each node represents a point in the generation where branching
/// can occur. Nodes share KV cache blocks with ancestors via CoW.
#[derive(Debug)]
pub struct TreeNode {
    /// Unique node identifier.
    id: NodeId,
    /// Parent node (None for root).
    parent: Option<NodeId>,
    /// Block table for this node's KV cache.
    block_table: BlockTable,
    /// Token IDs generated at this node.
    tokens: Vec<u32>,
    /// Number of active children (for garbage collection).
    num_children: u32,
    /// Whether this node is still active.
    is_active: bool,
}

impl TreeNode {
    /// Create a new root node.
    pub fn root(tokens_per_block: usize) -> Self {
        Self {
            id: NodeId::ROOT,
            parent: None,
            block_table: BlockTable::new(tokens_per_block),
            tokens: Vec::new(),
            num_children: 0,
            is_active: true,
        }
    }

    /// Create a child node (fork).
    pub fn fork(&self) -> Self {
        Self {
            id: NodeId::new(),
            parent: Some(self.id),
            block_table: self.block_table.fork(),
            tokens: Vec::new(),
            num_children: 0,
            is_active: true,
        }
    }

    /// Get node ID.
    pub fn id(&self) -> NodeId {
        self.id
    }

    /// Get parent node ID.
    pub fn parent(&self) -> Option<NodeId> {
        self.parent
    }

    /// Get the block table.
    pub fn block_table(&self) -> &BlockTable {
        &self.block_table
    }

    /// Get mutable block table.
    pub fn block_table_mut(&mut self) -> &mut BlockTable {
        &mut self.block_table
    }

    /// Get tokens generated at this node.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Add a token to this node.
    pub fn add_token(&mut self, token: u32) {
        self.tokens.push(token);
    }

    /// Get number of children.
    pub fn num_children(&self) -> u32 {
        self.num_children
    }

    /// Increment child count.
    pub fn add_child(&mut self) {
        self.num_children += 1;
    }

    /// Decrement child count.
    pub fn remove_child(&mut self) {
        self.num_children = self.num_children.saturating_sub(1);
    }

    /// Check if node is active.
    pub fn is_active(&self) -> bool {
        self.is_active
    }

    /// Deactivate node.
    pub fn deactivate(&mut self) {
        self.is_active = false;
    }

    /// Check if node can be garbage collected.
    pub fn can_gc(&self) -> bool {
        !self.is_active && self.num_children == 0
    }

    /// Total sequence length (tokens from root to this node).
    pub fn total_tokens(&self) -> usize {
        self.block_table.num_tokens()
    }
}
