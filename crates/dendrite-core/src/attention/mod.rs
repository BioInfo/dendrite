//! Attention computation backends.
//!
//! Provides abstractions over different attention implementations:
//! - [`FlashAttnBackend`] - GPU-accelerated via candle-flash-attn
//! - [`FlashInferBackend`] - Paged attention via FlashInfer (requires FFI)
//! - [`ReferenceBackend`] - CPU reference implementation for testing

mod backend;
mod paged;

pub use backend::{AttentionBackend, AttentionConfig, ReferenceBackend};
pub use paged::PagedAttention;

#[cfg(feature = "cuda")]
mod flashinfer;

#[cfg(feature = "cuda")]
pub use flashinfer::{FlashAttnBackend, FlashInferBackend};
