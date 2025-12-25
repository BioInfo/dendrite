//! # Dendrite FFI
//!
//! FFI bindings for FlashInfer and CUDA kernels.
//!
//! This crate provides safe Rust wrappers around:
//! - FlashInfer attention kernels
//! - Custom CUDA kernels for paged attention
//! - Memory management utilities

#![warn(missing_docs)]

pub mod flashinfer;
pub mod error;

#[cfg(feature = "cuda")]
pub mod cuda;

pub use error::{FfiError, Result};

/// FlashInfer version information.
pub const FLASHINFER_VERSION: &str = "0.2.x";

/// Check if CUDA is available at runtime.
#[cfg(feature = "cuda")]
pub fn cuda_available() -> bool {
    cuda::is_available()
}

#[cfg(not(feature = "cuda"))]
pub fn cuda_available() -> bool {
    false
}
