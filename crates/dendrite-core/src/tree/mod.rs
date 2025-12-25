//! Tree-structured state management for branching inference.
//!
//! This module provides:
//! - Tree nodes with parent/child relationships
//! - O(1) fork operations via copy-on-write
//! - Reference counting for shared KV cache blocks
//!
//! # Overview
//!
//! The tree structure enables efficient branching for algorithms like:
//! - Tree-of-Thought (ToT)
//! - Monte Carlo Tree Search (MCTS)
//! - Beam Search
//! - Speculative Decoding
//!
//! # O(1) Fork Latency
//!
//! Forking is O(1) with respect to sequence length because we don't copy
//! the KV cache data. Instead, we:
//! 1. Clone the block table (just pointers)
//! 2. Increment refcounts on shared blocks
//!
//! # Example
//!
//! ```rust
//! use dendrite_core::tree::{TreeState, NodeId};
//! use dendrite_core::cache::{KvCache, KvCacheConfig};
//! use std::sync::Arc;
//! use parking_lot::RwLock;
//!
//! // Create cache and tree
//! let config = KvCacheConfig::default();
//! let cache = Arc::new(RwLock::new(KvCache::new(config).unwrap()));
//! let tree = TreeState::new(cache, 16);
//!
//! // Fork from root
//! let handle = tree.fork(NodeId::ROOT).unwrap();
//!
//! // Add tokens to the sequence
//! tree.append_token(handle.node_id, 42).unwrap();
//!
//! // Fork to create branches
//! let branch1 = tree.fork(handle.node_id).unwrap();
//! let branch2 = tree.fork(handle.node_id).unwrap();
//!
//! // Each branch can now generate independently
//! tree.append_token(branch1.node_id, 100).unwrap();
//! tree.append_token(branch2.node_id, 200).unwrap();
//!
//! // Release when done
//! tree.release(branch1.node_id).unwrap();
//! tree.release(branch2.node_id).unwrap();
//! ```

mod node;
mod state;

pub use node::{NodeId, TreeNode};
pub use state::{ForkHandle, TreeState};
