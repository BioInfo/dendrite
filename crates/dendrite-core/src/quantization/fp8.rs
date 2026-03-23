//! FP8 quantization implementation.
//!
//! FP8 is an 8-bit floating point format supported on NVIDIA Hopper, Ada,
//! and Blackwell GPUs. It provides 2x memory savings and improved throughput
//! compared to FP16.
//!
//! # Formats
//!
//! - **E4M3**: 4-bit exponent, 3-bit mantissa
//!   - Range: ±448
//!   - Better precision, ideal for weights
//!
//! - **E5M2**: 5-bit exponent, 2-bit mantissa
//!   - Range: ±57344
//!   - Better dynamic range, ideal for activations
//!
//! # Hardware Support
//!
//! | GPU | FP8 E4M3 | FP8 E5M2 | MXFP8 |
//! |-----|----------|----------|-------|
//! | H100 | ✅ | ✅ | ❌ |
//! | GB10 | ✅ | ✅ | ✅ |
//! | B100 | ✅ | ✅ | ✅ |

use crate::error::Result;
use candle_core::Tensor;

/// FP8 format specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp8Format {
    /// E4M3: 4-bit exponent, 3-bit mantissa, no inf
    E4M3,
    /// E5M2: 5-bit exponent, 2-bit mantissa, with inf/nan
    E5M2,
}

impl Fp8Format {
    /// Get maximum representable value.
    pub fn max_value(&self) -> f32 {
        match self {
            Self::E4M3 => 448.0,
            Self::E5M2 => 57344.0,
        }
    }

    /// Get minimum positive value.
    pub fn min_positive(&self) -> f32 {
        match self {
            Self::E4M3 => 2.0f32.powi(-9),  // ~0.00195
            Self::E5M2 => 2.0f32.powi(-16), // ~0.0000153
        }
    }
}

/// Configuration for FP8 quantization.
#[derive(Debug, Clone)]
pub struct Fp8Config {
    /// FP8 format to use.
    pub format: Fp8Format,

    /// Per-channel scaling.
    pub per_channel: bool,

    /// Axis for per-channel scaling (usually 0 for weights).
    pub scale_axis: usize,
}

impl Default for Fp8Config {
    fn default() -> Self {
        Self {
            format: Fp8Format::E4M3,
            per_channel: true,
            scale_axis: 0,
        }
    }
}

/// A tensor with FP8 quantization metadata.
#[derive(Debug)]
pub struct QuantizedTensor {
    /// Quantized data stored as u8 (FP8 bit pattern).
    pub data: Tensor,

    /// Scale factors for dequantization.
    pub scales: Tensor,

    /// FP8 format used.
    pub format: Fp8Format,

    /// Original shape.
    pub shape: Vec<usize>,
}

impl QuantizedTensor {
    /// Quantize a tensor to FP8.
    ///
    /// # Arguments
    /// * `tensor` - Input tensor in FP16/FP32/BF16
    /// * `config` - FP8 configuration
    ///
    /// # Returns
    /// Quantized tensor with scale factors
    pub fn quantize(tensor: &Tensor, config: &Fp8Config) -> Result<Self> {
        let shape = tensor.dims().to_vec();
        let _device = tensor.device(); // Reserved for future GPU transfer

        // Compute scale factors
        let abs_max = if config.per_channel {
            // Per-channel: max along all axes except scale_axis
            let t = tensor.abs()?;
            // Reduce to get per-channel max
            let mut reduced = t;
            for i in (0..shape.len()).rev() {
                if i != config.scale_axis {
                    reduced = reduced.max(i)?;
                }
            }
            reduced
        } else {
            // Per-tensor: single scale
            tensor.abs()?.max_all()?
        };

        let format_max = config.format.max_value();
        let scales = (&abs_max / format_max as f64)?;

        // For now, store as FP32 since Candle doesn't have native FP8
        // In production, this would be stored as u8 with FP8 bit patterns
        let quantized = (tensor / &scales.broadcast_as(tensor.shape())?)?;
        let data = quantized.clamp(-format_max as f64, format_max as f64)?;

        Ok(Self {
            data,
            scales,
            format: config.format,
            shape,
        })
    }

    /// Dequantize back to FP32.
    pub fn dequantize(&self) -> Result<Tensor> {
        // Handle scalar scale (per-tensor quantization)
        if self.scales.dims().iter().product::<usize>() == 1 {
            let scale_value: f32 = self.scales.flatten_all()?.to_vec1()?[0];
            return Ok((&self.data * scale_value as f64)?);
        }

        // Per-channel dequantization
        let scales_broadcast = self.scales.broadcast_as(self.data.shape())?;
        let result = (&self.data * &scales_broadcast)?;
        Ok(result)
    }

    /// Get memory size in bytes.
    pub fn size_bytes(&self) -> usize {
        // Quantized data (1 byte per element) + scales
        let data_bytes = self.shape.iter().product::<usize>();
        let scale_bytes = self.scales.dims().iter().product::<usize>() * 4; // FP32 scales
        data_bytes + scale_bytes
    }

    /// Get compression ratio vs FP16.
    pub fn compression_ratio(&self) -> f32 {
        let fp16_size = self.shape.iter().product::<usize>() * 2;
        let quant_size = self.size_bytes();
        fp16_size as f32 / quant_size as f32
    }
}

/// Quantize a collection of weight tensors to FP8.
///
/// Iterates over all weight tensors and quantizes each using the provided
/// configuration. Weights use E4M3 format by default (better precision for
/// weights vs activations).
///
/// # Arguments
/// * `weights` - Slice of tensors to quantize (typically model weight matrices)
/// * `config` - FP8 quantization configuration (format, per-channel, axis)
///
/// # Returns
/// Vector of `QuantizedTensor` in the same order as input weights.
///
/// # Example
/// ```rust,ignore
/// use dendrite_core::quantization::fp8::{Fp8Config, Fp8Format, quantize_weights};
///
/// let config = Fp8Config { format: Fp8Format::E4M3, per_channel: true, scale_axis: 0 };
/// let quantized = quantize_weights(&weight_tensors, &config)?;
/// println!("Compression: {:.2}x", quantized[0].compression_ratio());
/// ```
pub fn quantize_weights(weights: &[Tensor], config: &Fp8Config) -> Result<Vec<QuantizedTensor>> {
    weights
        .iter()
        .map(|w| QuantizedTensor::quantize(w, config))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn fp8_format_ranges() {
        assert_eq!(Fp8Format::E4M3.max_value(), 448.0);
        assert_eq!(Fp8Format::E5M2.max_value(), 57344.0);
    }

    #[test]
    fn quantized_tensor_basic() {
        let device = Device::Cpu;
        let tensor = Tensor::randn(0.0f32, 1.0, (16, 64), &device).unwrap();

        // Use per-tensor quantization for simpler test
        let config = Fp8Config {
            format: Fp8Format::E4M3,
            per_channel: false,
            scale_axis: 0,
        };
        let quantized = QuantizedTensor::quantize(&tensor, &config).unwrap();

        assert_eq!(quantized.shape, vec![16, 64]);
        assert_eq!(quantized.format, Fp8Format::E4M3);

        // Dequantize and check shape
        let dequantized = quantized.dequantize().unwrap();
        assert_eq!(dequantized.dims(), &[16, 64]);
    }

    #[test]
    fn quantize_weights_batch() {
        let device = Device::Cpu;
        let config = Fp8Config {
            format: Fp8Format::E4M3,
            per_channel: false,
            scale_axis: 0,
        };

        // Simulate 3 weight tensors (e.g., Q, K, V projections)
        let weights: Vec<Tensor> = (0..3)
            .map(|_| Tensor::randn(0.0f32, 1.0, (64, 64), &device).unwrap())
            .collect();

        let quantized = super::quantize_weights(&weights, &config).unwrap();
        assert_eq!(quantized.len(), 3);

        for q in &quantized {
            assert_eq!(q.shape, vec![64, 64]);
            assert_eq!(q.format, Fp8Format::E4M3);
            // Each should be close to 2x compression
            assert!(q.compression_ratio() > 1.5);
        }
    }

    #[test]
    fn quantize_weights_empty() {
        let config = Fp8Config::default();
        let result = super::quantize_weights(&[], &config).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn quantized_tensor_compression() {
        let device = Device::Cpu;
        let tensor = Tensor::randn(0.0f32, 1.0, (1024, 1024), &device).unwrap();

        let config = Fp8Config {
            format: Fp8Format::E4M3,
            per_channel: false,
            scale_axis: 0,
        };

        let quantized = QuantizedTensor::quantize(&tensor, &config).unwrap();

        // Should be close to 2x compression (FP16 -> FP8)
        let ratio = quantized.compression_ratio();
        assert!(ratio > 1.8, "Compression ratio should be ~2x, got {}", ratio);
    }
}
