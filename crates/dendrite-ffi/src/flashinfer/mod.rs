//! FlashInfer kernel bindings.
//!
//! Safe Rust wrappers for FlashInfer attention kernels.

mod batch_decode;
mod batch_prefill;
mod cascade;

pub use batch_decode::BatchDecodeKernel;
pub use batch_prefill::BatchPrefillKernel;
pub use cascade::CascadeAttention;

/// Page size for paged attention (in tokens).
pub const PAGE_SIZE: usize = 16;

/// Maximum sequence length supported.
pub const MAX_SEQ_LEN: usize = 131072;

/// Supported head dimensions.
pub const SUPPORTED_HEAD_DIMS: &[usize] = &[64, 128, 256];
