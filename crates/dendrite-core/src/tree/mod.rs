//! Tree-structured state management for branching inference.
//!
//! This module provides:
//! - Tree nodes with parent/child relationships
//! - O(1) fork operations via copy-on-write
//! - Reference counting for shared KV cache blocks

mod node;
mod state;

pub use node::{TreeNode, NodeId};
pub use state::{TreeState, ForkHandle};
