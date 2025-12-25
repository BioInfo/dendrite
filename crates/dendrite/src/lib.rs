//! # Dendrite
//!
//! Agent-native inference engine with O(1) fork latency for tree-structured reasoning.
//!
//! Dendrite provides efficient LLM inference optimized for agentic workloads:
//! - **O(1) Fork**: Create reasoning branches without copying the KV cache
//! - **Tree-of-Thought**: Native support for MCTS and beam search
//! - **Grammar Constraints**: Structured output via llguidance integration
//! - **PagedAttention**: Memory-efficient KV cache management
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use dendrite::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Create engine
//!     let engine = Engine::builder()
//!         .model_path("model.safetensors")
//!         .build()
//!         .await?;
//!
//!     // Generate with tree search
//!     let result = engine
//!         .generate("Solve step by step:")
//!         .with_tree_search(TreeSearchConfig::default())
//!         .await?;
//!
//!     println!("{}", result.text);
//!     Ok(())
//! }
//! ```

#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]

// Re-export core crate
pub use dendrite_core::*;

// Re-export FFI crate (for advanced users)
pub use dendrite_ffi as ffi;

mod engine;

pub use engine::{
    Engine, EngineBuilder, EngineConfig, GenerateRequest, GenerateResult, ScoringMethod,
    TreeSearchConfig,
};

/// Commonly used types.
pub mod prelude {
    pub use crate::engine::{
        Engine, EngineBuilder, EngineConfig, GenerateRequest, GenerateResult, ScoringMethod,
        TreeSearchConfig,
    };
    pub use crate::{
        attention::AttentionBackend,
        cache::{KvCache, KvCacheConfig},
        error::{DendriteError, Result},
        grammar::{Grammar, GrammarConstraint},
        model::{ModelConfig, Transformer},
        scheduler::{BatchConfig, Request, Scheduler},
        tree::{ForkHandle, TreeNode, TreeState},
    };

    // Re-export useful external types
    pub use anyhow;
    pub use tokio;
    pub use tracing;
}
