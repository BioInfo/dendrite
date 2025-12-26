//! Transformer layer implementation.
//!
//! A single transformer decoder layer combining:
//! - Pre-attention RMSNorm
//! - Grouped Query Attention
//! - Post-attention RMSNorm
//! - SwiGLU MLP

use super::{LayerCache, RmsNorm, RotaryEmbedding, SwiGluMlp};
use crate::error::Result;
use candle_core::{Device, Tensor};

/// Self-attention module for transformer layers.
#[derive(Debug, Clone)]
pub struct Attention {
    /// Query projection: [num_heads * head_dim, hidden_size]
    q_proj: Tensor,
    /// Key projection: [num_kv_heads * head_dim, hidden_size]
    k_proj: Tensor,
    /// Value projection: [num_kv_heads * head_dim, hidden_size]
    v_proj: Tensor,
    /// Output projection: [hidden_size, num_heads * head_dim]
    o_proj: Tensor,
    /// Number of attention heads.
    num_heads: usize,
    /// Number of key-value heads (for GQA).
    num_kv_heads: usize,
    /// Head dimension.
    head_dim: usize,
    /// Hidden size.
    hidden_size: usize,
}

impl Attention {
    /// Create attention with random weights (for testing).
    pub fn random(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        device: &Device,
    ) -> Result<Self> {
        let q_proj = Tensor::randn(
            0.0f32,
            0.02,
            &[num_heads * head_dim, hidden_size],
            device,
        )?;
        let k_proj = Tensor::randn(
            0.0f32,
            0.02,
            &[num_kv_heads * head_dim, hidden_size],
            device,
        )?;
        let v_proj = Tensor::randn(
            0.0f32,
            0.02,
            &[num_kv_heads * head_dim, hidden_size],
            device,
        )?;
        let o_proj = Tensor::randn(
            0.0f32,
            0.02,
            &[hidden_size, num_heads * head_dim],
            device,
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_size,
        })
    }

    /// Create attention with provided weights.
    pub fn new(
        q_proj: Tensor,
        k_proj: Tensor,
        v_proj: Tensor,
        o_proj: Tensor,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        let hidden_size = q_proj.dims()[1];
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_size,
        })
    }

    /// Forward pass computing Q, K, V projections.
    ///
    /// Returns (query, key, value) tensors reshaped for attention:
    /// - query: [batch, num_heads, seq_len, head_dim]
    /// - key: [batch, num_kv_heads, seq_len, head_dim]
    /// - value: [batch, num_kv_heads, seq_len, head_dim]
    pub fn project(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let dims = x.dims();
        let batch = dims[0];
        let seq_len = dims[1];

        // Reshape to 2D for matmul if needed
        let x_2d = x.reshape((batch * seq_len, self.hidden_size))?;

        // Project Q, K, V
        let q = x_2d.matmul(&self.q_proj.t()?)?;
        let k = x_2d.matmul(&self.k_proj.t()?)?;
        let v = x_2d.matmul(&self.v_proj.t()?)?;

        // Reshape to [batch, seq, num_heads, head_dim] then transpose
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

    /// Apply output projection after attention.
    ///
    /// Input: [batch, num_heads, seq_len, head_dim]
    /// Output: [batch, seq_len, hidden_size]
    pub fn output(&self, attn_output: &Tensor) -> Result<Tensor> {
        let dims = attn_output.dims();
        let batch = dims[0];
        let seq_len = dims[2];

        // Transpose and reshape: [batch, seq, num_heads * head_dim]
        let x = attn_output.transpose(1, 2)?;
        let x = x.reshape((batch, seq_len, self.num_heads * self.head_dim))?;

        // Reshape to 2D for matmul
        let x_2d = x.reshape((batch * seq_len, self.num_heads * self.head_dim))?;
        let out = x_2d.matmul(&self.o_proj.t()?)?;

        // Reshape back to 3D
        let out = out.reshape((batch, seq_len, self.hidden_size))?;
        Ok(out)
    }

    /// Get the number of heads.
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Get the number of KV heads.
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    /// Get the head dimension.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
}

/// A single transformer decoder layer.
#[derive(Debug, Clone)]
pub struct TransformerLayer {
    /// Pre-attention layer norm.
    input_layernorm: RmsNorm,
    /// Self-attention.
    attention: Attention,
    /// Post-attention layer norm.
    post_attention_layernorm: RmsNorm,
    /// MLP.
    mlp: SwiGluMlp,
    /// Layer index (for debugging).
    layer_idx: usize,
}

impl TransformerLayer {
    /// Create a new transformer layer.
    pub fn new(
        input_layernorm: RmsNorm,
        attention: Attention,
        post_attention_layernorm: RmsNorm,
        mlp: SwiGluMlp,
        layer_idx: usize,
    ) -> Self {
        Self {
            input_layernorm,
            attention,
            post_attention_layernorm,
            mlp,
            layer_idx,
        }
    }

    /// Create a transformer layer with random weights (for testing).
    pub fn random(
        hidden_size: usize,
        intermediate_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        eps: f64,
        layer_idx: usize,
        device: &Device,
    ) -> Result<Self> {
        let input_layernorm = RmsNorm::ones(hidden_size, eps, device)?;
        let attention = Attention::random(hidden_size, num_heads, num_kv_heads, head_dim, device)?;
        let post_attention_layernorm = RmsNorm::ones(hidden_size, eps, device)?;
        let mlp = SwiGluMlp::random(hidden_size, intermediate_size, device)?;

        Ok(Self {
            input_layernorm,
            attention,
            post_attention_layernorm,
            mlp,
            layer_idx,
        })
    }

    /// Forward pass without KV cache (for prefill/testing).
    ///
    /// # Arguments
    ///
    /// * `hidden_states` - Input tensor [batch, seq_len, hidden_size]
    /// * `rope` - Rotary position embeddings
    /// * `position` - Starting position for RoPE
    /// * `attention_mask` - Optional causal mask
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        rope: &RotaryEmbedding,
        position: usize,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Pre-attention norm
        let normed = self.input_layernorm.forward(hidden_states)?;

        // Self-attention
        let (q, k, v) = self.attention.project(&normed)?;

        // Apply RoPE
        let (q, k) = rope.apply(&q, &k, position)?;

        // Compute attention scores
        let attn_output = self.compute_attention(&q, &k, &v, attention_mask)?;

        // Output projection
        let attn_output = self.attention.output(&attn_output)?;

        // Residual connection
        let hidden_states = (hidden_states + attn_output)?;

        // Post-attention norm
        let normed = self.post_attention_layernorm.forward(&hidden_states)?;

        // MLP
        let mlp_output = self.mlp.forward(&normed)?;

        // Residual connection
        let output = (hidden_states + mlp_output)?;

        Ok(output)
    }

    /// Compute scaled dot-product attention.
    fn compute_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let head_dim = self.attention.head_dim() as f64;
        let scale = 1.0 / head_dim.sqrt();

        // Handle GQA: expand K, V to match Q heads
        let num_heads = self.attention.num_heads();
        let num_kv_heads = self.attention.num_kv_heads();
        let (k, v) = if num_heads != num_kv_heads {
            let repeat = num_heads / num_kv_heads;
            let k = Self::repeat_kv(k, repeat)?;
            let v = Self::repeat_kv(v, repeat)?;
            (k, v)
        } else {
            (k.clone(), v.clone())
        };

        // Attention scores: Q @ K^T / sqrt(d)
        let scores = q.matmul(&k.transpose(2, 3)?)?;
        let scores = (scores * scale)?;

        // Apply mask if provided
        let scores = if let Some(mask) = mask {
            scores.broadcast_add(mask)?
        } else {
            scores
        };

        // Softmax
        let attn_weights = candle_nn::ops::softmax(&scores, candle_core::D::Minus1)?;

        // Attention output: weights @ V
        let output = attn_weights.matmul(&v)?;

        Ok(output)
    }

    /// Repeat KV heads for GQA.
    fn repeat_kv(x: &Tensor, repeat: usize) -> Result<Tensor> {
        if repeat == 1 {
            return Ok(x.clone());
        }

        let dims = x.dims();
        let batch = dims[0];
        let num_kv_heads = dims[1];
        let seq_len = dims[2];
        let head_dim = dims[3];

        // [batch, num_kv_heads, seq, head_dim] -> [batch, num_kv_heads, 1, seq, head_dim]
        let x = x.unsqueeze(2)?;
        // Expand to [batch, num_kv_heads, repeat, seq, head_dim]
        let x = x.expand(&[batch, num_kv_heads, repeat, seq_len, head_dim])?;
        // Reshape to [batch, num_heads, seq, head_dim]
        let x = x.reshape((batch, num_kv_heads * repeat, seq_len, head_dim))?;

        Ok(x)
    }

    /// Get the layer index.
    pub fn layer_idx(&self) -> usize {
        self.layer_idx
    }

    /// Get the attention module.
    pub fn attention(&self) -> &Attention {
        &self.attention
    }

    /// Get the MLP module.
    pub fn mlp(&self) -> &SwiGluMlp {
        &self.mlp
    }

    /// Forward pass with KV cache for autoregressive generation.
    ///
    /// This method:
    /// 1. Computes Q, K, V projections for the new tokens
    /// 2. Appends K, V to the cache
    /// 3. Uses the full cached K, V for attention
    ///
    /// # Arguments
    ///
    /// * `hidden_states` - Input tensor [batch, seq_len, hidden_size]
    /// * `rope` - Rotary position embeddings
    /// * `cache` - Layer KV cache to read from and update
    /// * `attention_mask` - Optional causal mask
    ///
    /// # Returns
    ///
    /// Output hidden states [batch, seq_len, hidden_size]
    pub fn forward_with_cache(
        &self,
        hidden_states: &Tensor,
        rope: &RotaryEmbedding,
        cache: &mut LayerCache,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Pre-attention norm
        let normed = self.input_layernorm.forward(hidden_states)?;

        // Get Q, K, V projections
        let (q, k, v) = self.attention.project(&normed)?;

        // Get position offset from cache
        let position = cache.seq_len();

        // Apply RoPE to new tokens
        let (q, k) = rope.apply(&q, &k, position)?;

        // Append K, V to cache and get full K, V
        let (full_k, full_v) = cache.append(&k, &v)?;

        // Compute attention with full cached K, V
        let attn_output = self.compute_attention(&q, &full_k, &full_v, attention_mask)?;

        // Output projection
        let attn_output = self.attention.output(&attn_output)?;

        // Residual connection
        let hidden_states = (hidden_states + attn_output)?;

        // Post-attention norm
        let normed = self.post_attention_layernorm.forward(&hidden_states)?;

        // MLP
        let mlp_output = self.mlp.forward(&normed)?;

        // Residual connection
        let output = (hidden_states + mlp_output)?;

        Ok(output)
    }
}

/// Create a causal attention mask.
pub fn create_causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
    // Create causal mask manually
    // mask[i][j] = 0 if j <= i, -inf otherwise
    let neg_inf = f32::NEG_INFINITY;
    let mut mask_data = vec![0.0f32; seq_len * seq_len];

    for i in 0..seq_len {
        for j in 0..seq_len {
            if j > i {
                mask_data[i * seq_len + j] = neg_inf;
            }
        }
    }

    let mask = Tensor::from_slice(&mask_data, (seq_len, seq_len), device)?;

    // Add batch and head dimensions: [1, 1, seq_len, seq_len]
    let mask = mask.unsqueeze(0)?.unsqueeze(0)?;

    Ok(mask)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_layer() -> (TransformerLayer, RotaryEmbedding) {
        let hidden_size = 256;
        let intermediate_size = 512;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 64;
        let eps = 1e-5;

        let layer = TransformerLayer::random(
            hidden_size,
            intermediate_size,
            num_heads,
            num_kv_heads,
            head_dim,
            eps,
            0,
            &Device::Cpu,
        )
        .unwrap();

        let rope = RotaryEmbedding::new(head_dim, 2048, 10000.0, &Device::Cpu).unwrap();

        (layer, rope)
    }

    #[test]
    fn attention_projection_shapes() {
        let hidden_size = 256;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 64;

        let attn =
            Attention::random(hidden_size, num_heads, num_kv_heads, head_dim, &Device::Cpu)
                .unwrap();

        // [batch=2, seq=8, hidden=256]
        let x = Tensor::randn(0.0f32, 1.0, &[2, 8, 256], &Device::Cpu).unwrap();
        let (q, k, v) = attn.project(&x).unwrap();

        assert_eq!(q.dims(), &[2, 4, 8, 64]); // [batch, num_heads, seq, head_dim]
        assert_eq!(k.dims(), &[2, 2, 8, 64]); // [batch, num_kv_heads, seq, head_dim]
        assert_eq!(v.dims(), &[2, 2, 8, 64]);
    }

    #[test]
    fn attention_output_shape() {
        let hidden_size = 256;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 64;

        let attn =
            Attention::random(hidden_size, num_heads, num_kv_heads, head_dim, &Device::Cpu)
                .unwrap();

        // Simulated attention output: [batch, num_heads, seq, head_dim]
        let attn_output = Tensor::randn(0.0f32, 1.0, &[2, 4, 8, 64], &Device::Cpu).unwrap();
        let output = attn.output(&attn_output).unwrap();

        assert_eq!(output.dims(), &[2, 8, 256]); // [batch, seq, hidden]
    }

    #[test]
    fn layer_forward_shape() {
        let (layer, rope) = create_test_layer();

        // [batch=1, seq=16, hidden=256]
        let x = Tensor::randn(0.0f32, 1.0, &[1, 16, 256], &Device::Cpu).unwrap();
        let output = layer.forward(&x, &rope, 0, None).unwrap();

        assert_eq!(output.dims(), &[1, 16, 256]);
    }

    #[test]
    fn layer_forward_with_mask() {
        let (layer, rope) = create_test_layer();

        let x = Tensor::randn(0.0f32, 1.0, &[1, 8, 256], &Device::Cpu).unwrap();
        let mask = create_causal_mask(8, &Device::Cpu).unwrap();
        let output = layer.forward(&x, &rope, 0, Some(&mask)).unwrap();

        assert_eq!(output.dims(), &[1, 8, 256]);
    }

    #[test]
    fn causal_mask_shape() {
        let mask = create_causal_mask(16, &Device::Cpu).unwrap();
        assert_eq!(mask.dims(), &[1, 1, 16, 16]);
    }

    #[test]
    fn causal_mask_values() {
        let mask = create_causal_mask(4, &Device::Cpu).unwrap();
        let mask_2d = mask.squeeze(0).unwrap().squeeze(0).unwrap();
        let vals: Vec<Vec<f32>> = mask_2d.to_vec2().unwrap();

        // First row: only position 0 is valid
        assert!(vals[0][0].is_finite()); // 0.0
        assert!(vals[0][1].is_infinite()); // -inf

        // Last row: all positions valid
        assert!(vals[3][0].is_finite());
        assert!(vals[3][1].is_finite());
        assert!(vals[3][2].is_finite());
        assert!(vals[3][3].is_finite());
    }

    #[test]
    fn repeat_kv_identity() {
        let x = Tensor::randn(0.0f32, 1.0, &[1, 8, 4, 64], &Device::Cpu).unwrap();
        let repeated = TransformerLayer::repeat_kv(&x, 1).unwrap();
        assert_eq!(repeated.dims(), x.dims());
    }

    #[test]
    fn repeat_kv_expansion() {
        // [batch=1, num_kv_heads=2, seq=4, head_dim=64]
        let x = Tensor::randn(0.0f32, 1.0, &[1, 2, 4, 64], &Device::Cpu).unwrap();
        let repeated = TransformerLayer::repeat_kv(&x, 4).unwrap();

        // Should expand to [1, 8, 4, 64]
        assert_eq!(repeated.dims(), &[1, 8, 4, 64]);
    }

    #[test]
    fn gqa_attention_works() {
        let (layer, rope) = create_test_layer();

        // GQA with 4 query heads and 2 KV heads
        assert_eq!(layer.attention.num_heads(), 4);
        assert_eq!(layer.attention.num_kv_heads(), 2);

        let x = Tensor::randn(0.0f32, 1.0, &[1, 8, 256], &Device::Cpu).unwrap();
        let output = layer.forward(&x, &rope, 0, None).unwrap();

        assert_eq!(output.dims(), &[1, 8, 256]);
    }
}
