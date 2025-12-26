//! Error types for Dendrite Core.

use thiserror::Error;

/// Result type alias for Dendrite operations.
pub type Result<T> = std::result::Result<T, DendriteError>;

/// Errors that can occur in Dendrite operations.
#[derive(Error, Debug)]
pub enum DendriteError {
    /// Out of memory for KV cache blocks.
    #[error("out of memory: {0}")]
    OutOfMemory(String),

    /// Invalid block reference.
    #[error("invalid block id: {0}")]
    InvalidBlock(u32),

    /// Invalid tree node reference.
    #[error("invalid node id: {0}")]
    InvalidNode(u64),

    /// Fork operation failed.
    #[error("fork failed: {0}")]
    ForkFailed(String),

    /// Scheduler error.
    #[error("scheduler error: {0}")]
    SchedulerError(String),

    /// Cache operation error.
    #[error("cache error: {0}")]
    CacheError(String),

    /// Attention computation error.
    #[error("attention error: {0}")]
    AttentionError(String),

    /// Model loading error.
    #[error("model error: {0}")]
    ModelError(String),

    /// Grammar constraint error.
    #[error("grammar error: {0}")]
    GrammarError(String),

    /// Shape mismatch error.
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),

    /// CUDA/GPU error.
    #[error("cuda error: {0}")]
    CudaError(String),

    /// I/O error.
    #[error("io error: {0}")]
    IoError(#[from] std::io::Error),

    /// Serialization error.
    #[error("serialization error: {0}")]
    SerdeError(#[from] serde_json::Error),

    /// Candle tensor error.
    #[error("tensor error: {0}")]
    TensorError(#[from] candle_core::Error),
}
