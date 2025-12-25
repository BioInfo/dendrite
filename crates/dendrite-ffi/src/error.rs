//! Error types for FFI operations.

use thiserror::Error;

/// Result type for FFI operations.
pub type Result<T> = std::result::Result<T, FfiError>;

/// Errors from FFI operations.
#[derive(Error, Debug)]
pub enum FfiError {
    /// CUDA runtime error.
    #[error("CUDA error: {0}")]
    CudaError(String),

    /// FlashInfer kernel error.
    #[error("FlashInfer error: {0}")]
    FlashInferError(String),

    /// Invalid argument.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// Memory allocation failed.
    #[error("memory allocation failed: {0}")]
    AllocationFailed(String),

    /// Feature not available.
    #[error("feature not available: {0}")]
    NotAvailable(String),
}
