//! FP8-quantized transformer layer components.
//!
//! Provides drop-in FP8 replacements for [`Attention`] and [`SwiGluMlp`]
//! that store weights as MXFP8 and dequantize on the fly during the forward pass.
//!
//! # Design
//!
//! These are additive wrappers — the FP8 variants have the **exact same
//! forward-pass interface** as the FP16 originals. You can swap them in
//! by calling `.quantize()` on an existing layer.
//!
//! # Memory Savings
//!
//! A 7B-parameter Llama model (mostly attention + MLP projections):
//! - FP16: ~14 GB
//! - MXFP8 quantized: ~7.3 GB (~1.94x compression)
//!
//! On GB10 (128 GB unified), this enables fitting 70B+ models that wouldn't
//! otherwise fit.

use crate::error::Result;
use crate::quantization::fp8_linear::{quantize_attention_projs, Fp8Linear};
use crate::quantization::DEFAULT_BLOCK_SIZE;
use candle_core::Tensor;

/// FP8-quantized self-attention module.
///
/// Same interface as [`Attention`](super::Attention) but stores Q/K/V/O
/// projections as MXFP8-compressed `Fp8Linear` layers.
#[derive(Debug)]
pub struct Fp8Attention {
    q_proj: Fp8Linear,
    k_proj: Fp8Linear,
    v_proj: Fp8Linear,
    o_proj: Fp8Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    hidden_size: usize,
}

impl Fp8Attention {
    /// Quantize dense attention weight tensors into FP8.
    ///
    /// Takes the raw weight matrices (same shapes as [`Attention::new`]) and
    /// produces an FP8 attention block at the given block size.
    pub fn from_weights(
        q_proj: &Tensor,
        k_proj: &Tensor,
        v_proj: &Tensor,
        o_proj: &Tensor,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
    ) -> Result<Self> {
        let hidden_size = q_proj.dims()[1];

        let (q, k, v, o) = quantize_attention_projs(q_proj, k_proj, v_proj, o_proj, block_size)?;

        Ok(Self {
            q_proj: q,
            k_proj: k,
            v_proj: v,
            o_proj: o,
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_size,
        })
    }

    /// Create using the default OCP MXFP8 block size (32).
    pub fn from_weights_default(
        q_proj: &Tensor,
        k_proj: &Tensor,
        v_proj: &Tensor,
        o_proj: &Tensor,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        Self::from_weights(
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            DEFAULT_BLOCK_SIZE,
        )
    }

    /// Project input into Q, K, V using FP8 weights.
    ///
    /// Returns (query, key, value) shaped for attention:
    /// - query: [batch, num_heads, seq_len, head_dim]
    /// - key:   [batch, num_kv_heads, seq_len, head_dim]
    /// - value: [batch, num_kv_heads, seq_len, head_dim]
    pub fn project(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let dims = x.dims();
        let batch = dims[0];
        let seq_len = dims[1];

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        Ok((q, k, v))
    }

    /// Apply FP8 output projection after attention.
    ///
    /// Input: [batch, num_heads, seq_len, head_dim]
    /// Output: [batch, seq_len, hidden_size]
    pub fn output(&self, attn_output: &Tensor) -> Result<Tensor> {
        let dims = attn_output.dims();
        let batch = dims[0];
        let seq_len = dims[2];

        let x = attn_output.transpose(1, 2)?;
        let x = x.reshape((batch, seq_len, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&x)
    }

    /// Total compressed weight size in bytes.
    pub fn weight_bytes(&self) -> usize {
        self.q_proj.weight_bytes()
            + self.k_proj.weight_bytes()
            + self.v_proj.weight_bytes()
            + self.o_proj.weight_bytes()
    }

    /// Average compression ratio across all four projections vs FP16.
    pub fn compression_ratio(&self) -> f32 {
        let ratios = [
            self.q_proj.compression_ratio(),
            self.k_proj.compression_ratio(),
            self.v_proj.compression_ratio(),
            self.o_proj.compression_ratio(),
        ];
        ratios.iter().sum::<f32>() / ratios.len() as f32
    }

    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
}

// ─────────────────────────────────────────────
// FP8 SwiGLU MLP
// ─────────────────────────────────────────────

/// FP8-quantized SwiGLU MLP.
///
/// Same interface as [`SwiGluMlp`](super::SwiGluMlp) but stores
/// gate/up/down projections as MXFP8 `Fp8Linear` layers.
#[derive(Debug)]
pub struct Fp8SwiGluMlp {
    gate_proj: Fp8Linear,
    up_proj: Fp8Linear,
    down_proj: Fp8Linear,
    hidden_size: usize,
    intermediate_size: usize,
}

impl Fp8SwiGluMlp {
    /// Quantize dense MLP weights into FP8.
    pub fn from_weights(
        gate_proj: &Tensor,
        up_proj: &Tensor,
        down_proj: &Tensor,
        block_size: usize,
    ) -> Result<Self> {
        let hidden_size = gate_proj.dims()[1];
        let intermediate_size = gate_proj.dims()[0];

        Ok(Self {
            gate_proj: Fp8Linear::from_weight(gate_proj, None, block_size)?,
            up_proj: Fp8Linear::from_weight(up_proj, None, block_size)?,
            down_proj: Fp8Linear::from_weight(down_proj, None, block_size)?,
            hidden_size,
            intermediate_size,
        })
    }

    /// Create using the default OCP MXFP8 block size (32).
    pub fn from_weights_default(
        gate_proj: &Tensor,
        up_proj: &Tensor,
        down_proj: &Tensor,
    ) -> Result<Self> {
        Self::from_weights(gate_proj, up_proj, down_proj, DEFAULT_BLOCK_SIZE)
    }

    /// Forward pass: dequantize-and-compute SwiGLU.
    ///
    /// `x` can be 2D `[batch, hidden]` or 3D `[batch, seq, hidden]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // gate_proj(x) and up_proj(x) — Fp8Linear handles 2D/3D natively
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;

        // SwiGLU: silu(up) * gate
        let activated = (candle_nn::ops::silu(&up) * gate)?;

        // down_proj
        self.down_proj.forward(&activated)
    }

    /// Total compressed weight size in bytes.
    pub fn weight_bytes(&self) -> usize {
        self.gate_proj.weight_bytes() + self.up_proj.weight_bytes() + self.down_proj.weight_bytes()
    }

    /// Compression ratio vs FP16 (averaged across three projections).
    pub fn compression_ratio(&self) -> f32 {
        let ratios = [
            self.gate_proj.compression_ratio(),
            self.up_proj.compression_ratio(),
            self.down_proj.compression_ratio(),
        ];
        ratios.iter().sum::<f32>() / ratios.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    fn make_fp8_attention(
        hidden: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        device: &Device,
    ) -> Fp8Attention {
        let q = Tensor::randn(0.0f32, 0.02, (num_heads * head_dim, hidden), device).unwrap();
        let k = Tensor::randn(0.0f32, 0.02, (num_kv_heads * head_dim, hidden), device).unwrap();
        let v = Tensor::randn(0.0f32, 0.02, (num_kv_heads * head_dim, hidden), device).unwrap();
        let o = Tensor::randn(0.0f32, 0.02, (hidden, num_heads * head_dim), device).unwrap();
        Fp8Attention::from_weights_default(&q, &k, &v, &o, num_heads, num_kv_heads, head_dim)
            .unwrap()
    }

    fn make_fp8_mlp(hidden: usize, intermediate: usize, device: &Device) -> Fp8SwiGluMlp {
        let gate = Tensor::randn(0.0f32, 0.02, (intermediate, hidden), device).unwrap();
        let up = Tensor::randn(0.0f32, 0.02, (intermediate, hidden), device).unwrap();
        let down = Tensor::randn(0.0f32, 0.02, (hidden, intermediate), device).unwrap();
        Fp8SwiGluMlp::from_weights_default(&gate, &up, &down).unwrap()
    }

    // ── Attention tests ──────────────────────────────────

    #[test]
    fn fp8_attention_project_shapes() {
        let device = Device::Cpu;
        let attn = make_fp8_attention(256, 4, 2, 64, &device);

        let x = Tensor::randn(0.0f32, 1.0, (1, 8, 256), &device).unwrap();
        let (q, k, v) = attn.project(&x).unwrap();

        assert_eq!(q.dims(), &[1, 4, 8, 64]); // [batch, heads, seq, head_dim]
        assert_eq!(k.dims(), &[1, 2, 8, 64]);
        assert_eq!(v.dims(), &[1, 2, 8, 64]);
    }

    #[test]
    fn fp8_attention_output_shape() {
        let device = Device::Cpu;
        let attn = make_fp8_attention(256, 4, 2, 64, &device);

        // Simulated post-attention output: [batch, num_heads, seq, head_dim]
        let attn_out = Tensor::randn(0.0f32, 1.0, (2, 4, 8, 64), &device).unwrap();
        let out = attn.output(&attn_out).unwrap();

        assert_eq!(out.dims(), &[2, 8, 256]);
    }

    #[test]
    fn fp8_attention_compression() {
        let device = Device::Cpu;
        let attn = make_fp8_attention(256, 4, 2, 64, &device);
        assert!(
            attn.compression_ratio() > 1.5,
            "Expected >1.5x compression, got {:.2}",
            attn.compression_ratio()
        );
    }

    #[test]
    fn fp8_attention_output_finite() {
        let device = Device::Cpu;
        let attn = make_fp8_attention(256, 4, 4, 64, &device);

        let x = Tensor::randn(0.0f32, 1.0, (1, 4, 256), &device).unwrap();
        let (q, k, v) = attn.project(&x).unwrap();

        // Simulate sdp manually: Q @ K^T / sqrt(d)
        let scale = 1.0 / (64.0f64).sqrt();
        let scores = q.matmul(&k.transpose(2, 3).unwrap()).unwrap();
        let scores = (scores * scale).unwrap();
        let weights = candle_nn::ops::softmax(&scores, candle_core::D::Minus1).unwrap();
        let context = weights.matmul(&v).unwrap();

        let out = attn.output(&context).unwrap();
        let vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "Output has non-finite values"
        );
    }

    // ── MLP tests ─────────────────────────────────────────

    #[test]
    fn fp8_mlp_forward_2d() {
        let device = Device::Cpu;
        let mlp = make_fp8_mlp(256, 512, &device);

        let x = Tensor::randn(0.0f32, 1.0, (4, 256), &device).unwrap();
        let out = mlp.forward(&x).unwrap();
        assert_eq!(out.dims(), &[4, 256]);
    }

    #[test]
    fn fp8_mlp_forward_3d() {
        let device = Device::Cpu;
        let mlp = make_fp8_mlp(256, 512, &device);

        let x = Tensor::randn(0.0f32, 1.0, (2, 16, 256), &device).unwrap();
        let out = mlp.forward(&x).unwrap();
        assert_eq!(out.dims(), &[2, 16, 256]);
    }

    #[test]
    fn fp8_mlp_compression() {
        let device = Device::Cpu;
        let mlp = make_fp8_mlp(1024, 4096, &device);
        assert!(
            mlp.compression_ratio() > 1.8,
            "Expected >1.8x, got {:.2}",
            mlp.compression_ratio()
        );
    }

    #[test]
    fn fp8_mlp_output_finite() {
        let device = Device::Cpu;
        let mlp = make_fp8_mlp(256, 512, &device);

        let x = Tensor::randn(0.0f32, 1.0, (2, 8, 256), &device).unwrap();
        let out = mlp.forward(&x).unwrap();
        let vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert!(vals.iter().all(|v| v.is_finite()));
    }

    // ── End-to-end forward pass ───────────────────────────

    /// Full FP8 transformer layer: norm → Fp8Attention → residual → norm → Fp8MLP → residual
    #[test]
    fn fp8_end_to_end_forward() {
        use crate::model::RmsNorm;
        use crate::model::RotaryEmbedding;

        let device = Device::Cpu;
        let hidden = 256;
        let intermediate = 512;
        let num_heads = 4;
        let num_kv_heads = 4;
        let head_dim = 64;
        let batch = 2;
        let seq = 8;

        // Build FP8 attention
        let attn = make_fp8_attention(hidden, num_heads, num_kv_heads, head_dim, &device);
        // Build FP8 MLP
        let mlp = make_fp8_mlp(hidden, intermediate, &device);
        // Layer norms
        let norm1 = RmsNorm::ones(hidden, 1e-5, &device).unwrap();
        let norm2 = RmsNorm::ones(hidden, 1e-5, &device).unwrap();
        // RoPE
        let rope = RotaryEmbedding::new(head_dim, 2048, 10000.0, &device).unwrap();

        // Input hidden states [batch, seq, hidden]
        let hidden_states = Tensor::randn(0.0f32, 1.0, (batch, seq, hidden), &device).unwrap();

        // ── Attention sublayer ─────────────────────────────
        let normed = norm1.forward(&hidden_states).unwrap();
        let (q, k, v) = attn.project(&normed).unwrap();
        let (q, k) = rope.apply(&q, &k, 0).unwrap();

        // Scaled dot-product attention
        let scale = 1.0 / (head_dim as f64).sqrt();
        let k_cont = k.contiguous().unwrap();
        let scores = q.matmul(&k_cont.transpose(2, 3).unwrap()).unwrap();
        let scores = (scores * scale).unwrap();
        let attn_weights = candle_nn::ops::softmax(&scores, candle_core::D::Minus1).unwrap();
        let v_cont = v.contiguous().unwrap();
        let context = attn_weights.matmul(&v_cont).unwrap();
        let attn_out = attn.output(&context).unwrap();

        // Residual
        let hidden_states = (&hidden_states + attn_out).unwrap();

        // ── MLP sublayer ───────────────────────────────────
        let normed = norm2.forward(&hidden_states).unwrap();
        let mlp_out = mlp.forward(&normed).unwrap();
        let output = (&hidden_states + mlp_out).unwrap();

        // Shape check
        assert_eq!(output.dims(), &[batch, seq, hidden]);

        // Finite check
        let vals: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "FP8 end-to-end output contains non-finite values"
        );

        println!(
            "FP8 end-to-end: attn {:.2}x compression, mlp {:.2}x compression",
            attn.compression_ratio(),
            mlp.compression_ratio(),
        );
    }

    /// Verify that FP8 output stays numerically close to FP16 baseline.
    ///
    /// Uses identical random weights converted to both formats,
    /// then checks that the mean absolute error is within tolerance.
    #[test]
    fn fp8_vs_fp16_numeric_agreement() {
        use crate::model::SwiGluMlp;

        let device = Device::Cpu;
        let hidden = 512;
        let intermediate = 1024;

        // Shared weight tensors
        let gate = Tensor::randn(0.0f32, 0.02, (intermediate, hidden), &device).unwrap();
        let up = Tensor::randn(0.0f32, 0.02, (intermediate, hidden), &device).unwrap();
        let down = Tensor::randn(0.0f32, 0.02, (hidden, intermediate), &device).unwrap();

        // FP16 (actually FP32 in candle CPU) baseline
        let fp16_mlp = SwiGluMlp::new(gate.clone(), up.clone(), down.clone()).unwrap();
        // FP8 quantized
        let fp8_mlp = Fp8SwiGluMlp::from_weights_default(&gate, &up, &down).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, (4, 16, hidden), &device).unwrap();

        let fp16_out = fp16_mlp.forward(&x).unwrap();
        let fp8_out = fp8_mlp.forward(&x).unwrap();

        // Compute mean absolute error
        let diff = (fp8_out - fp16_out).unwrap().abs().unwrap();
        let mae: f32 = diff.mean_all().unwrap().to_scalar().unwrap();

        // FP8 introduces quantization noise; typical MAE should be < 1% of signal
        let fp16_mean_abs: f32 = fp16_mlp
            .forward(&x)
            .unwrap()
            .abs()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar()
            .unwrap();

        let relative_error = if fp16_mean_abs > 1e-8 {
            mae / fp16_mean_abs
        } else {
            mae
        };

        println!(
            "FP8 vs FP16 relative MAE: {:.4} ({:.2}%)",
            relative_error,
            relative_error * 100.0
        );

        assert!(
            relative_error < 0.05,
            "FP8 relative error {:.4} exceeds 5% tolerance (MAE={:.6}, FP16 mean={:.6})",
            relative_error,
            mae,
            fp16_mean_abs
        );
    }
}
