//! Quantization support for efficient inference.
//!
//! This module provides quantization schemes for reducing memory usage and
//! improving throughput on supported hardware.
//!
//! # Supported Formats
//!
//! - **FP8 (E4M3)**: 8-bit floating point for NVIDIA Hopper/Ada GPUs
//! - **MXFP8**: Microscaling FP8 with shared exponents for GB10/Blackwell
//! - **INT8**: 8-bit integer quantization with scale factors
//!
//! # GB10 Optimization
//!
//! The NVIDIA GB10 (DGX Spark) supports:
//! - FP8 compute with 2x throughput vs FP16
//! - MXFP8 for memory-bound inference
//! - 128GB unified memory via NVLink-C2C
//!
//! # Usage
//!
//! ```rust,ignore
//! use dendrite_core::quantization::{QuantConfig, QuantFormat};
//!
//! let config = QuantConfig {
//!     format: QuantFormat::FP8E4M3,
//!     per_channel: true,
//!     dynamic: false,
//! };
//!
//! // Quantize weights
//! let quantized = quantize_tensor(&weights, &config)?;
//! ```
//!
//! # Roadmap
//!
//! - [x] Module structure and configuration
//! - [ ] FP8 E4M3/E5M2 conversion
//! - [ ] MXFP8 with block-wise scaling
//! - [ ] INT8 symmetric/asymmetric
//! - [ ] Integration with FlashInfer FP8 kernels

mod fp8;

pub use fp8::{Fp8Config, Fp8Format, QuantizedTensor};

/// Quantization format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantFormat {
    /// FP8 E4M3 (4-bit exponent, 3-bit mantissa)
    /// Range: ±448, good for weights
    FP8E4M3,

    /// FP8 E5M2 (5-bit exponent, 2-bit mantissa)
    /// Range: ±57344, good for activations
    FP8E5M2,

    /// MXFP8 with shared 8-bit exponent per block
    /// Reduces memory further with minimal accuracy loss
    MXFP8,

    /// INT8 symmetric quantization
    INT8Sym,

    /// INT8 asymmetric quantization
    INT8Asym,
}

impl QuantFormat {
    /// Get bits per element.
    pub fn bits(&self) -> usize {
        match self {
            Self::FP8E4M3 | Self::FP8E5M2 => 8,
            Self::MXFP8 => 8, // Plus shared exponent overhead
            Self::INT8Sym | Self::INT8Asym => 8,
        }
    }

    /// Check if this format requires scale factors.
    pub fn needs_scale(&self) -> bool {
        matches!(self, Self::MXFP8 | Self::INT8Sym | Self::INT8Asym)
    }
}

/// Quantization configuration.
#[derive(Debug, Clone)]
pub struct QuantConfig {
    /// Quantization format.
    pub format: QuantFormat,

    /// Per-channel vs per-tensor quantization.
    pub per_channel: bool,

    /// Dynamic quantization (compute scales at runtime).
    pub dynamic: bool,

    /// Block size for MXFP8 shared exponents.
    pub block_size: usize,
}

impl Default for QuantConfig {
    fn default() -> Self {
        Self {
            format: QuantFormat::FP8E4M3,
            per_channel: true,
            dynamic: false,
            block_size: 32,
        }
    }
}

impl QuantConfig {
    /// Create FP8 E4M3 config (best for weights).
    pub fn fp8_weights() -> Self {
        Self {
            format: QuantFormat::FP8E4M3,
            per_channel: true,
            dynamic: false,
            block_size: 32,
        }
    }

    /// Create FP8 E5M2 config (best for activations).
    pub fn fp8_activations() -> Self {
        Self {
            format: QuantFormat::FP8E5M2,
            per_channel: false,
            dynamic: true,
            block_size: 32,
        }
    }

    /// Create MXFP8 config for maximum memory efficiency.
    pub fn mxfp8() -> Self {
        Self {
            format: QuantFormat::MXFP8,
            per_channel: true,
            dynamic: false,
            block_size: 32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quant_format_bits() {
        assert_eq!(QuantFormat::FP8E4M3.bits(), 8);
        assert_eq!(QuantFormat::FP8E5M2.bits(), 8);
        assert_eq!(QuantFormat::MXFP8.bits(), 8);
        assert_eq!(QuantFormat::INT8Sym.bits(), 8);
    }

    #[test]
    fn quant_format_needs_scale() {
        assert!(!QuantFormat::FP8E4M3.needs_scale());
        assert!(QuantFormat::MXFP8.needs_scale());
        assert!(QuantFormat::INT8Sym.needs_scale());
    }

    #[test]
    fn quant_config_presets() {
        let weights = QuantConfig::fp8_weights();
        assert_eq!(weights.format, QuantFormat::FP8E4M3);
        assert!(weights.per_channel);

        let activations = QuantConfig::fp8_activations();
        assert_eq!(activations.format, QuantFormat::FP8E5M2);
        assert!(activations.dynamic);
    }
}
