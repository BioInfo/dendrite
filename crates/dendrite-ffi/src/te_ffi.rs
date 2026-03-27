//! Transformer Engine (TE) FFI bindings — stub layer.
//!
//! NVIDIA Transformer Engine provides fused FP8 GEMM kernels that are
//! 2x faster than unfused dequant + matmul on Hopper/Blackwell hardware.
//!
//! # Current Status
//!
//! This module defines the interface Dendrite will use once TE bindings are
//! wired. The current implementation falls back to the pure-Rust MXFP8
//! path (quantize weights once, dequantize per forward pass).
//!
//! # Planned TE Integration
//!
//! When `DENDRITE_USE_TE=1` is set and TE is installed:
//! 1. `te_fp8_gemm` will call `transformer_engine::gemm_fp8` directly
//! 2. No dequantization step — the GEMM consumes FP8 natively
//! 3. Expected speedup: 1.8–2.2x over the dequant path on H100/GB10
//!
//! # Usage
//!
//! ```rust,ignore
//! use dendrite_ffi::te_ffi::{is_te_available, te_fp8_gemm};
//!
//! if is_te_available() {
//!     let out = te_fp8_gemm(&q_fp8, &k_fp8, scale_a, scale_b)?;
//! } else {
//!     // Fall back to pure-Rust dequant path
//!     let out = layer.forward(&input)?;
//! }
//! ```
//!
//! # References
//!
//! - https://github.com/NVIDIA/TransformerEngine
//! - TE Python API: `transformer_engine.pytorch.fp8_autocast`

/// Check whether NVIDIA Transformer Engine is available at runtime.
///
/// Currently always returns `false` (stub). When TE bindings land,
/// this will probe for `libte_ffi.so` / the TE Python wheel.
#[inline]
pub fn is_te_available() -> bool {
    // TODO: check for TE shared library via dlopen or env gate
    false
}

/// Fused FP8 GEMM via Transformer Engine.
///
/// This is a stub — returns `Err(TeNotAvailable)` until TE bindings
/// are implemented. The caller should fall back to the Dendrite
/// pure-Rust MXFP8 path.
///
/// # Arguments
///
/// * `a_data` - FP8 matrix A (row-major bytes)
/// * `b_data` - FP8 matrix B (row-major bytes)
/// * `m`, `n`, `k` - Matrix dimensions: (M×K) @ (K×N) → (M×N)
/// * `scale_a` - Per-tensor scale for A (f32)
/// * `scale_b` - Per-tensor scale for B (f32)
///
/// # Returns
///
/// `Ok(output_bytes)` — FP32 output flattened to bytes — or `Err` if TE unavailable.
pub fn te_fp8_gemm(
    _a_data: &[u8],
    _b_data: &[u8],
    _m: usize,
    _n: usize,
    _k: usize,
    _scale_a: f32,
    _scale_b: f32,
) -> Result<Vec<f32>, TeError> {
    Err(TeError::NotAvailable)
}

/// Errors from the TE FFI layer.
#[derive(Debug, thiserror::Error)]
pub enum TeError {
    /// Transformer Engine is not installed or not detected.
    #[error("Transformer Engine is not available (set DENDRITE_USE_TE=1 and install TE)")]
    NotAvailable,

    /// TE call succeeded but returned an unexpected result.
    #[error("Transformer Engine GEMM failed: {0}")]
    GemmFailed(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn te_not_available_by_default() {
        // TE is not installed in the test environment; stub returns false
        assert!(!is_te_available());
    }

    #[test]
    fn te_gemm_returns_not_available() {
        let result = te_fp8_gemm(&[], &[], 4, 4, 4, 1.0, 1.0);
        assert!(matches!(result, Err(TeError::NotAvailable)));
    }
}
