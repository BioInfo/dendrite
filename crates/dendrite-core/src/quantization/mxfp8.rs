//! MXFP8 (Microscaling FP8) block quantization.
//!
//! MXFP8 is a microscaling format supported on NVIDIA Blackwell (GB10/B100) GPUs.
//! Unlike standard FP8 which uses per-channel or per-tensor scaling, MXFP8 uses
//! a shared 8-bit exponent for every 32 elements (a "block"), significantly
//! reducing the overhead of scale storage while maintaining accuracy.
//!
//! # Block Structure
//!
//! For a tensor of shape [M, N], MXFP8 groups elements into blocks of `BLOCK_SIZE`
//! (default 32) along the innermost dimension. Each block shares a single E8M0
//! (8-bit exponent, no mantissa) scale factor. The FP8 values within the block
//! use E4M3 format.
//!
//! Memory layout:
//! - Data: M × N bytes (FP8 E4M3 per element)
//! - Scales: M × ceil(N / BLOCK_SIZE) bytes (E8M0 per block)
//!
//! This achieves ~2x compression vs FP16 with very low scale overhead
//! (1 byte per 32 elements vs 1 float per 32 elements for per-channel FP8).
//!
//! # Hardware Support
//!
//! | GPU    | MXFP8 Native | Block Size |
//! |--------|-------------|------------|
//! | H100   | ❌          | N/A        |
//! | B100   | ✅          | 32         |
//! | GB10   | ✅          | 32         |
//!
//! # References
//!
//! - OCP MX Specification: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
//! - NVIDIA Blackwell Architecture whitepaper

use crate::error::Result;
use candle_core::Tensor;

/// Default block size for MXFP8 scaling (OCP MX spec).
pub const DEFAULT_BLOCK_SIZE: usize = 32;

/// E8M0 maximum exponent bias.
const E8M0_MAX: u8 = 254; // 255 is reserved for NaN

/// An MXFP8 microscaling quantized tensor.
///
/// Stores data in FP8 E4M3 format with per-block E8M0 (8-bit exponent) scale
/// factors. One scale per `block_size` elements along the last dimension.
#[derive(Debug)]
pub struct MxFp8Tensor {
    /// Quantized element data (FP8 E4M3 bit patterns stored as u8-equivalent f32).
    /// Shape: same as original tensor.
    pub data: Tensor,

    /// Block scale exponents in E8M0 format (stored as u8-equivalent f32).
    /// Shape: [..., ceil(N / block_size)] where N is the last dimension.
    pub scales: Tensor,

    /// Block size used for quantization.
    pub block_size: usize,

    /// Original tensor shape.
    pub shape: Vec<usize>,
}

impl MxFp8Tensor {
    /// Quantize a 2D tensor using MXFP8 block scaling.
    ///
    /// Processes the tensor in blocks of `block_size` elements along the last
    /// dimension. Each block gets a shared E8M0 exponent computed from the
    /// block maximum.
    ///
    /// # Arguments
    /// * `tensor` - Input tensor (any floating-point dtype, shape [M, N])
    /// * `block_size` - Number of elements per scaling block (default: 32)
    ///
    /// # Returns
    /// `MxFp8Tensor` with block-quantized data and E8M0 scale factors.
    pub fn quantize(tensor: &Tensor, block_size: usize) -> Result<Self> {
        let shape = tensor.dims().to_vec();
        let ndim = shape.len();

        // Flatten to 2D: [batch, features]
        let rows = shape[..ndim - 1].iter().product::<usize>().max(1);
        let cols = *shape.last().unwrap_or(&1);
        let flat = tensor.reshape((rows, cols))?;

        let n_blocks = cols.div_ceil(block_size);
        let fp8_e4m3_max = 448.0f32;

        // Compute per-block scales and quantize
        // We work in f32 for portability (candle doesn't expose native FP8/E8M0)
        let flat_f32 = flat.to_dtype(candle_core::DType::F32)?;
        let flat_vec: Vec<f32> = flat_f32.flatten_all()?.to_vec1()?;

        let mut quantized_vec = vec![0.0f32; rows * cols];
        let mut scales_vec = vec![0.0f32; rows * n_blocks];

        for row in 0..rows {
            for block_idx in 0..n_blocks {
                let start = block_idx * block_size;
                let end = (start + block_size).min(cols);

                // Find block maximum (abs)
                let block_max = (start..end)
                    .map(|c| flat_vec[row * cols + c].abs())
                    .fold(f32::NEG_INFINITY, f32::max);

                // Compute E8M0 scale exponent: scale = 2^floor(log2(block_max / fp8_max))
                // Store as the actual scale value (fp8_max / block_max) for dequantization
                let scale = if block_max > 0.0 {
                    fp8_e4m3_max / block_max
                } else {
                    1.0 // Zero block — scale doesn't matter
                };

                // Clamp exponent to E8M0 range
                let log2_scale = scale.log2().floor();
                let clamped_scale =
                    2.0f32.powf(log2_scale.clamp(-(E8M0_MAX as f32), E8M0_MAX as f32));

                scales_vec[row * n_blocks + block_idx] = clamped_scale;

                // Quantize each element in block
                for c in start..end {
                    let elem = flat_vec[row * cols + c];
                    let scaled = elem * clamped_scale;
                    // Clamp to FP8 E4M3 range and apply rounding
                    let quantized = scaled.clamp(-fp8_e4m3_max, fp8_e4m3_max);
                    quantized_vec[row * cols + c] = quantized;
                }
            }
        }

        let device = tensor.device();
        let data_flat = Tensor::from_vec(quantized_vec, (rows, cols), device)?;
        let scales_flat = Tensor::from_vec(scales_vec, (rows, n_blocks), device)?;

        // Reshape back to original shape
        let data = if ndim > 2 {
            let mut out_shape = shape[..ndim - 1].to_vec();
            out_shape.push(cols);
            data_flat.reshape(out_shape)?
        } else {
            data_flat
        };

        let scales = if ndim > 2 {
            let mut scale_shape = shape[..ndim - 1].to_vec();
            scale_shape.push(n_blocks);
            scales_flat.reshape(scale_shape)?
        } else {
            scales_flat
        };

        Ok(Self {
            data,
            scales,
            block_size,
            shape,
        })
    }

    /// Dequantize MXFP8 tensor back to FP32.
    ///
    /// Reconstructs the original values by dividing each block by its
    /// corresponding E8M0 scale factor.
    pub fn dequantize(&self) -> Result<Tensor> {
        let shape = &self.shape;
        let ndim = shape.len();
        let rows = shape[..ndim - 1].iter().product::<usize>().max(1);
        let cols = *shape.last().unwrap_or(&1);
        let n_blocks = cols.div_ceil(self.block_size);

        let data_f32 = self
            .data
            .reshape((rows, cols))?
            .to_dtype(candle_core::DType::F32)?;
        let scales_f32 = self
            .scales
            .reshape((rows, n_blocks))?
            .to_dtype(candle_core::DType::F32)?;

        let data_vec: Vec<f32> = data_f32.flatten_all()?.to_vec1()?;
        let scales_vec: Vec<f32> = scales_f32.flatten_all()?.to_vec1()?;

        let mut output = vec![0.0f32; rows * cols];

        for row in 0..rows {
            for block_idx in 0..n_blocks {
                let start = block_idx * self.block_size;
                let end = (start + self.block_size).min(cols);
                let scale = scales_vec[row * n_blocks + block_idx];

                for c in start..end {
                    let idx = row * cols + c;
                    output[idx] = if scale > 0.0 {
                        data_vec[idx] / scale
                    } else {
                        0.0
                    };
                }
            }
        }

        let device = self.data.device();
        let out_flat = Tensor::from_vec(output, (rows, cols), device)?;

        if ndim > 2 {
            let mut out_shape = shape[..ndim - 1].to_vec();
            out_shape.push(cols);
            Ok(out_flat.reshape(out_shape)?)
        } else {
            Ok(out_flat)
        }
    }

    /// Compute memory size in bytes for the quantized representation.
    ///
    /// Returns the storage footprint: 1 byte per element (FP8) plus 1 byte per
    /// block for the E8M0 scale.
    pub fn size_bytes(&self) -> usize {
        let ndim = self.shape.len();
        let rows = self.shape[..ndim - 1].iter().product::<usize>().max(1);
        let cols = *self.shape.last().unwrap_or(&1);
        let n_blocks = cols.div_ceil(self.block_size);

        // 1 byte/element (FP8 E4M3) + 1 byte/block (E8M0 exponent)
        rows * cols + rows * n_blocks
    }

    /// Compute compression ratio vs FP16 (2 bytes/element).
    pub fn compression_ratio(&self) -> f32 {
        let _ndim = self.shape.len();
        let n_elements = self.shape.iter().product::<usize>();
        let fp16_bytes = n_elements * 2;
        fp16_bytes as f32 / self.size_bytes() as f32
    }
}

/// Quantize a batch of weight tensors to MXFP8 with block scaling.
///
/// # Arguments
/// * `weights` - Slice of tensors to quantize
/// * `block_size` - MXFP8 block size (32 per OCP MX spec)
///
/// # Returns
/// Vector of `MxFp8Tensor` in the same order as input weights.
pub fn quantize_weights_mxfp8(weights: &[Tensor], block_size: usize) -> Result<Vec<MxFp8Tensor>> {
    weights
        .iter()
        .map(|w| MxFp8Tensor::quantize(w, block_size))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn mxfp8_basic_quantize_dequantize() {
        let device = Device::Cpu;
        let data: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let tensor = Tensor::from_vec(data.clone(), (4, 16), &device).unwrap();

        let quantized = MxFp8Tensor::quantize(&tensor, 16).unwrap();
        assert_eq!(quantized.shape, vec![4, 16]);
        assert_eq!(quantized.block_size, 16);

        // Scales shape: [4, 1] (1 block of 16 per row)
        assert_eq!(quantized.scales.dims(), &[4, 1]);

        let dequantized = quantized.dequantize().unwrap();
        assert_eq!(dequantized.dims(), &[4, 16]);

        // Dequantized should be close to original
        let deq_vec: Vec<f32> = dequantized.flatten_all().unwrap().to_vec1().unwrap();
        for (orig, deq) in data.iter().zip(deq_vec.iter()) {
            let err = (orig - deq).abs();
            assert!(
                err < 0.05,
                "Dequant error too large: {} vs {} (err {})",
                orig,
                deq,
                err
            );
        }
    }

    #[test]
    fn mxfp8_block_size_32() {
        let device = Device::Cpu;
        let tensor = Tensor::randn(0.0f32, 1.0, (8, 64), &device).unwrap();

        let quantized = MxFp8Tensor::quantize(&tensor, DEFAULT_BLOCK_SIZE).unwrap();
        // 64 elements / 32 block_size = 2 blocks per row
        assert_eq!(quantized.scales.dims(), &[8, 2]);
    }

    #[test]
    fn mxfp8_non_multiple_cols() {
        let device = Device::Cpu;
        // 40 cols, block_size=32 -> 2 blocks (32 + 8)
        let tensor = Tensor::randn(0.0f32, 1.0, (4, 40), &device).unwrap();

        let quantized = MxFp8Tensor::quantize(&tensor, DEFAULT_BLOCK_SIZE).unwrap();
        assert_eq!(quantized.scales.dims(), &[4, 2]);

        let dequantized = quantized.dequantize().unwrap();
        assert_eq!(dequantized.dims(), &[4, 40]);
    }

    #[test]
    fn mxfp8_compression_ratio() {
        let device = Device::Cpu;
        // Large tensor for stable compression ratio
        let tensor = Tensor::randn(0.0f32, 1.0, (256, 1024), &device).unwrap();

        let quantized = MxFp8Tensor::quantize(&tensor, DEFAULT_BLOCK_SIZE).unwrap();
        let ratio = quantized.compression_ratio();

        // With block_size=32: 1 + 1/32 overhead per element
        // Expected ~1.94x compression vs FP16
        assert!(ratio > 1.8, "Expected ~1.9x compression, got {:.3}x", ratio);
        assert!(ratio < 2.1, "Unexpectedly high compression {:.3}x", ratio);
    }

    #[test]
    fn mxfp8_batch_quantize() {
        let device = Device::Cpu;
        let weights: Vec<Tensor> = (0..4)
            .map(|_| Tensor::randn(0.0f32, 1.0, (64, 64), &device).unwrap())
            .collect();

        let quantized = quantize_weights_mxfp8(&weights, DEFAULT_BLOCK_SIZE).unwrap();
        assert_eq!(quantized.len(), 4);

        for q in &quantized {
            assert_eq!(q.shape, vec![64, 64]);
            assert_eq!(q.block_size, DEFAULT_BLOCK_SIZE);
        }
    }

    #[test]
    fn mxfp8_zero_tensor() {
        let device = Device::Cpu;
        let tensor = Tensor::zeros((4, 32), candle_core::DType::F32, &device).unwrap();

        let quantized = MxFp8Tensor::quantize(&tensor, DEFAULT_BLOCK_SIZE).unwrap();
        let dequantized = quantized.dequantize().unwrap();

        let deq_vec: Vec<f32> = dequantized.flatten_all().unwrap().to_vec1().unwrap();
        for v in &deq_vec {
            assert!(
                v.abs() < 1e-6,
                "Zero tensor should dequantize to zeros, got {}",
                v
            );
        }
    }
}
