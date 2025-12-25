//! # Dendrite Core
//!
//! Core engine for tree-structured KV cache management with O(1) fork latency.
//!
//! This crate provides:
//! - **Tree-structured KV cache** with copy-on-write semantics
//! - **PagedAttention** implementation with block tables
//! - **FlashInfer** kernel integration for efficient attention
//! - **Scheduler** for batched prefill/decode operations
//! - **Grammar-constrained decoding** via llguidance

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
