//! Attention computation backends.
//!
//! Provides abstractions over different attention implementations:
//! - FlashInfer (production)
//! - Reference implementation (testing)

mod backend;
mod paged;

pub use backend::{AttentionBackend, AttentionConfig, ReferenceBackend};
pub use paged::PagedAttention;

#[cfg(feature = "cuda")]
mod flashinfer;

#[cfg(feature = "cuda")]
pub use flashinfer::FlashInferBackend;
