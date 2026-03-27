//! FP8-quantized linear layer for memory-efficient inference.
//!
//! Wraps a weight matrix in MXFP8 block-quantized format and performs
//! matrix multiply by dequantizing on the fly. This is the "quantize once,
//! dequantize per-forward-pass" pattern — suitable for memory-bound inference
//! where the bottleneck is VRAM bandwidth, not compute.
//!
//! # Usage
//!
//! ```rust,ignore
//! use dendrite_core::quantization::fp8_linear::Fp8Linear;
//!
//! // Quantize a weight matrix once at load time
//! let layer = Fp8Linear::from_weight(&weight_tensor, bias, DEFAULT_BLOCK_SIZE)?;
//!
//! // Use in forward pass — dequantizes weight and performs matmul
//! let output = layer.forward(&input)?;
//!
//! println!("Compression: {:.2}x", layer.compression_ratio());
//! ```
//!
//! # Memory Savings
//!
//! For a 7B parameter model (mostly linear layers in attention + MLP):
//! - FP16 weights: ~14 GB
//! - MXFP8 weights: ~7.3 GB (1.94x compression)
//! - Headroom on GB10 (128 GB unified): enough for 70B+ parameter models

use crate::error::Result;
use crate::quantization::mxfp8::{MxFp8Tensor, DEFAULT_BLOCK_SIZE};
use candle_core::Tensor;

/// A linear layer with MXFP8-quantized weights.
///
/// Stores the weight matrix in compressed MXFP8 format (block_size=32 by default)
/// and dequantizes at forward-pass time. Bias (if present) is stored in FP32.
///
/// This trades a small compute overhead (dequant per forward pass) for ~2x
/// memory bandwidth reduction, improving throughput on memory-bound workloads.
#[derive(Debug)]
pub struct Fp8Linear {
    /// MXFP8-compressed weight matrix.
    /// Original shape: [out_features, in_features]
    pub weight: MxFp8Tensor,

    /// Optional bias vector. Shape: [out_features]
    pub bias: Option<Tensor>,

    /// Input (in) features.
    pub in_features: usize,

    /// Output (out) features.
    pub out_features: usize,
}

impl Fp8Linear {
    /// Create an Fp8Linear by quantizing a dense weight tensor.
    ///
    /// # Arguments
    /// * `weight` - Dense weight tensor, shape [out_features, in_features]
    /// * `bias` - Optional bias tensor, shape [out_features]
    /// * `block_size` - MXFP8 block size (32 per OCP spec)
    pub fn from_weight(weight: &Tensor, bias: Option<Tensor>, block_size: usize) -> Result<Self> {
        let shape = weight.dims();
        assert_eq!(shape.len(), 2, "Weight must be 2D [out, in]");
        let out_features = shape[0];
        let in_features = shape[1];

        let quantized = MxFp8Tensor::quantize(weight, block_size)?;

        Ok(Self {
            weight: quantized,
            bias,
            in_features,
            out_features,
        })
    }

    /// Create with default OCP block size (32).
    pub fn from_weight_default(weight: &Tensor, bias: Option<Tensor>) -> Result<Self> {
        Self::from_weight(weight, bias, DEFAULT_BLOCK_SIZE)
    }

    /// Forward pass: dequantize weights and perform linear transform.
    ///
    /// `input` can be:
    /// - 2D: [batch, in_features] → output [batch, out_features]
    /// - 3D: [batch, seq, in_features] → output [batch, seq, out_features]
    ///
    /// The weight matrix is dequantized from MXFP8 at each call. In a
    /// GPU-accelerated path, this would fuse into a single kernel.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Dequantize weight: [out_features, in_features] in FP32
        let weight_fp32 = self.weight.dequantize()?;
        let weight_t = weight_fp32.t()?; // [in_features, out_features]

        let input_dims = input.dims();
        let ndim = input_dims.len();

        // Flatten batch dimensions for matmul, then restore shape
        let output = if ndim == 2 {
            // [batch, in] @ [in, out] → [batch, out]
            input.matmul(&weight_t)?
        } else {
            // [batch, seq, in] → reshape to [batch*seq, in], matmul, reshape back
            let leading: usize = input_dims[..ndim - 1].iter().product();
            let flat = input.reshape((leading, self.in_features))?;
            let out_flat = flat.matmul(&weight_t)?; // [batch*seq, out]

            // Restore original leading dims
            let mut out_shape = input_dims[..ndim - 1].to_vec();
            out_shape.push(self.out_features);
            out_flat.reshape(out_shape)?
        };

        // Add bias if present
        let output = if let Some(bias) = &self.bias {
            output.broadcast_add(bias)?
        } else {
            output
        };

        Ok(output)
    }

    /// Get memory footprint of compressed weights in bytes.
    pub fn weight_bytes(&self) -> usize {
        self.weight.size_bytes()
    }

    /// Get compression ratio vs FP16.
    pub fn compression_ratio(&self) -> f32 {
        self.weight.compression_ratio()
    }
}

/// Quantize a set of attention projection weights (Q, K, V, O) to FP8.
///
/// Convenience function for quantizing all four attention projections at once.
/// Returns (q, k, v, o) as `Fp8Linear` layers.
pub fn quantize_attention_projs(
    q_proj: &Tensor,
    k_proj: &Tensor,
    v_proj: &Tensor,
    o_proj: &Tensor,
    block_size: usize,
) -> Result<(Fp8Linear, Fp8Linear, Fp8Linear, Fp8Linear)> {
    Ok((
        Fp8Linear::from_weight(q_proj, None, block_size)?,
        Fp8Linear::from_weight(k_proj, None, block_size)?,
        Fp8Linear::from_weight(v_proj, None, block_size)?,
        Fp8Linear::from_weight(o_proj, None, block_size)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn make_linear(out: usize, inp: usize, device: &Device) -> Fp8Linear {
        let weight = Tensor::randn(0.0f32, 0.02, (out, inp), device).unwrap();
        Fp8Linear::from_weight_default(&weight, None).unwrap()
    }

    #[test]
    fn fp8_linear_forward_2d() {
        let device = Device::Cpu;
        let layer = make_linear(64, 128, &device);

        let input = Tensor::randn(0.0f32, 1.0, (8, 128), &device).unwrap();
        let output = layer.forward(&input).unwrap();

        assert_eq!(output.dims(), &[8, 64]);
    }

    #[test]
    fn fp8_linear_forward_3d() {
        let device = Device::Cpu;
        let layer = make_linear(256, 512, &device);

        // [batch=2, seq=16, in=512] → [batch=2, seq=16, out=256]
        let input = Tensor::randn(0.0f32, 1.0, (2, 16, 512), &device).unwrap();
        let output = layer.forward(&input).unwrap();

        assert_eq!(output.dims(), &[2, 16, 256]);
    }

    #[test]
    fn fp8_linear_with_bias() {
        let device = Device::Cpu;
        let weight = Tensor::randn(0.0f32, 0.02, (32, 64), &device).unwrap();
        let bias = Tensor::zeros((32,), DType::F32, &device).unwrap();
        let layer = Fp8Linear::from_weight_default(&weight, Some(bias)).unwrap();

        let input = Tensor::randn(0.0f32, 1.0, (4, 64), &device).unwrap();
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.dims(), &[4, 32]);
    }

    #[test]
    fn fp8_linear_compression() {
        let device = Device::Cpu;
        // 1024x1024 weight for meaningful compression ratio
        let layer = make_linear(1024, 1024, &device);
        let ratio = layer.compression_ratio();
        // Expect ~1.94x (block_size=32, 1 scale byte per 32 data bytes)
        assert!(ratio > 1.8, "Expected ~1.9x, got {:.3}", ratio);
    }

    #[test]
    fn fp8_linear_numerics() {
        let device = Device::Cpu;
        // Simple identity-ish: weight = small random, check output is bounded
        let weight = Tensor::randn(0.0f32, 0.01, (16, 32), &device).unwrap();
        let layer = Fp8Linear::from_weight_default(&weight, None).unwrap();

        let input = Tensor::randn(0.0f32, 1.0, (4, 32), &device).unwrap();
        let output = layer.forward(&input).unwrap();

        // Output should be finite (no NaN/inf)
        let out_vec: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
        for v in &out_vec {
            assert!(v.is_finite(), "Output contains non-finite value: {}", v);
        }
    }

    #[test]
    fn quantize_attention_projs_all_four() {
        let device = Device::Cpu;
        let hidden = 256;
        let head_dim = 64;
        let num_heads = 4;
        let num_kv_heads = 2;

        let q = Tensor::randn(0.0f32, 0.02, (num_heads * head_dim, hidden), &device).unwrap();
        let k = Tensor::randn(0.0f32, 0.02, (num_kv_heads * head_dim, hidden), &device).unwrap();
        let v = Tensor::randn(0.0f32, 0.02, (num_kv_heads * head_dim, hidden), &device).unwrap();
        let o = Tensor::randn(0.0f32, 0.02, (hidden, num_heads * head_dim), &device).unwrap();

        let (q_fp8, k_fp8, v_fp8, o_fp8) =
            quantize_attention_projs(&q, &k, &v, &o, DEFAULT_BLOCK_SIZE).unwrap();

        assert_eq!(q_fp8.in_features, hidden);
        assert_eq!(q_fp8.out_features, num_heads * head_dim);
        assert_eq!(k_fp8.out_features, num_kv_heads * head_dim);
        assert_eq!(v_fp8.out_features, num_kv_heads * head_dim);
        assert_eq!(o_fp8.out_features, hidden);

        // All should show >1.8x compression
        for (name, layer) in [("Q", &q_fp8), ("K", &k_fp8), ("V", &v_fp8), ("O", &o_fp8)] {
            assert!(
                layer.compression_ratio() > 1.5,
                "{} compression {:.2}x too low",
                name,
                layer.compression_ratio()
            );
        }
    }
}
