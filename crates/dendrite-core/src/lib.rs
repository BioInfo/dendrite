//! # Dendrite Core
//!
//! Core engine for tree-structured KV cache management with O(1) fork latency.
//!
//! This crate provides the fundamental building blocks for Dendrite:
//! - **Tree-structured KV cache** with copy-on-write semantics
//! - **PagedAttention** implementation with block tables
//! - **FlashInfer** kernel integration for efficient attention
//! - **Scheduler** for batched prefill/decode operations
//! - **Grammar-constrained decoding** via llguidance
//!
//! ## Core Concepts
//!
//! ### O(1) Fork Latency
//!
//! The key innovation in Dendrite is O(1) fork latency. When you fork a sequence
//! (e.g., for tree search), we don't copy the KV cache data. Instead, we use
//! copy-on-write semantics with reference counting on cache blocks.
//!
//! ```rust,ignore
//! use dendrite_core::prelude::*;
//!
//! // Create cache and tree state
//! let kv_cache = Arc::new(RwLock::new(KvCache::new(config)?));
//! let tree = TreeState::new(kv_cache, 16);
//!
//! // Process some tokens on the base sequence
//! let base = tree.fork(NodeId::ROOT)?;
//! for token in input_tokens {
//!     tree.append_token(base.node_id, token)?;
//! }
//!
//! // Fork for tree search - O(1) operation!
//! let branch1 = tree.fork(base.node_id)?;  // ~200ns
//! let branch2 = tree.fork(base.node_id)?;  // ~200ns
//! let branch3 = tree.fork(base.node_id)?;  // ~200ns
//!
//! // Each branch can now generate independently
//! // They share the parent's KV cache via copy-on-write
//! ```
//!
//! ### PagedAttention with Blocks
//!
//! The KV cache is organized into fixed-size blocks (default: 16 tokens).
//! This enables efficient memory management and sharing between sequences.
//!
//! ```rust,ignore
//! use dendrite_core::cache::{KvCache, KvCacheConfig};
//!
//! let config = KvCacheConfig {
//!     num_layers: 32,        // Transformer layers
//!     num_kv_heads: 8,       // KV attention heads
//!     head_dim: 128,         // Dimension per head
//!     max_blocks: 65536,     // Pool size
//!     tokens_per_block: 16,  // Tokens per block
//! };
//!
//! let cache = KvCache::new(config)?;
//!
//! // Allocate blocks as needed
//! let block = cache.allocate_block()?;
//!
//! // Share blocks for copy-on-write
//! cache.share_block(block)?;
//!
//! // Copy-on-write when modifying shared block
//! let new_block = cache.copy_on_write(block)?;
//! ```
//!
//! ### Continuous Batching Scheduler
//!
//! The scheduler manages request lifecycle with continuous batching:
//!
//! ```rust,ignore
//! use dendrite_core::scheduler::{Scheduler, Request, BatchConfig};
//!
//! let scheduler = Scheduler::new(BatchConfig::default(), tree_state);
//!
//! // Add requests
//! let request = Request::new(input_tokens, max_tokens);
//! scheduler.add_request(request)?;
//!
//! // Schedule execution
//! loop {
//!     match scheduler.schedule()? {
//!         Some(batch) => {
//!             // Execute batch (prefill or decode)
//!             execute_batch(&batch)?;
//!         }
//!         None => break, // No more work
//!     }
//! }
//! ```
//!
//! ## Module Overview
//!
//! - [`cache`] - KV cache with block pool and copy-on-write
//! - [`tree`] - Tree state management for branching inference
//! - [`scheduler`] - Request scheduling with continuous batching
//! - [`attention`] - Attention backend trait and implementations
//! - [`grammar`] - Grammar constraints for structured output
//! - [`model`] - Transformer model definitions
//! - [`error`] - Error types and Result alias

#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]

pub mod attention;
pub mod cache;
pub mod error;
pub mod grammar;
pub mod model;
pub mod scheduler;
pub mod tree;

pub use error::{DendriteError, Result};

/// Re-export commonly used types
pub mod prelude {
    pub use crate::attention::AttentionBackend;
    pub use crate::cache::{Block, BlockTable, KvCache};
    pub use crate::error::{DendriteError, Result};
    pub use crate::scheduler::{BatchConfig, Request, Scheduler};
    pub use crate::tree::{ForkHandle, TreeNode, TreeState};
}
