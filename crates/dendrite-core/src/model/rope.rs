//! Rotary Position Embeddings (RoPE).
//!
//! RoPE encodes position information by rotating query and key vectors
//! in the complex plane. This allows the model to understand relative
//! positions between tokens.
//!
//! # Reference
//!
//! [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)

use crate::error::Result;
use candle_core::{Device, Tensor};

/// Rotary Position Embedding implementation.
#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    /// Cosine cache for positions.
    cos_cache: Tensor,
    /// Sine cache for positions.
    sin_cache: Tensor,
    /// Head dimension.
    head_dim: usize,
    /// Maximum sequence length cached.
    max_seq_len: usize,
}

impl RotaryEmbedding {
    /// Create a new rotary embedding.
    ///
    /// # Arguments
    ///
    /// * `head_dim` - Dimension of each attention head
    /// * `max_seq_len` - Maximum sequence length to cache
    /// * `theta` - Base frequency (default 10000.0, Llama-3 uses 500000.0)
    /// * `device` - Device for tensors
    pub fn new(head_dim: usize, max_seq_len: usize, theta: f64, device: &Device) -> Result<Self> {
        // Compute inverse frequencies: 1 / (theta^(2i/d)) for i in [0, d/2)
        let half_dim = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64) as f32)
            .collect();
        let inv_freq = Tensor::from_slice(&inv_freq, (1, half_dim), device)?;

        // Compute position indices
        let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();
        let positions = Tensor::from_slice(&positions, (max_seq_len, 1), device)?;

        // Compute freqs = positions * inv_freq -> [max_seq_len, half_dim]
        let freqs = positions.matmul(&inv_freq)?;

        // Cache cos and sin
        let cos_cache = freqs.cos()?;
        let sin_cache = freqs.sin()?;

        Ok(Self {
            cos_cache,
            sin_cache,
            head_dim,
            max_seq_len,
        })
    }

    /// Apply rotary embeddings to query and key tensors.
    ///
    /// # Arguments
    ///
    /// * `query` - Query tensor [batch, num_heads, seq_len, head_dim]
    /// * `key` - Key tensor [batch, num_kv_heads, seq_len, head_dim]
    /// * `position_ids` - Starting position for each sequence
    pub fn apply(
        &self,
        query: &Tensor,
        key: &Tensor,
        position_ids: usize,
    ) -> Result<(Tensor, Tensor)> {
        let seq_len = query.dims()[2];

        // Get cos/sin for these positions
        let cos = self.cos_cache.narrow(0, position_ids, seq_len)?;
        let sin = self.sin_cache.narrow(0, position_ids, seq_len)?;

        // Apply RoPE
        let query_rot = self.rotate_half(query, &cos, &sin)?;
        let key_rot = self.rotate_half(key, &cos, &sin)?;

        Ok((query_rot, key_rot))
    }

    /// Rotate tensor using cos/sin embeddings.
    fn rotate_half(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let half_dim = self.head_dim / 2;
        let dims = x.dims();

        // Split into first and second halves
        let x1 = x.narrow(3, 0, half_dim)?;
        let x2 = x.narrow(3, half_dim, half_dim)?;

        // Reshape cos/sin for broadcasting: [1, 1, seq_len, half_dim]
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

        // Broadcast to match x dimensions
        let cos = cos.broadcast_as(&[dims[0], dims[1], dims[2], half_dim])?;
        let sin = sin.broadcast_as(&[dims[0], dims[1], dims[2], half_dim])?;

        // Apply rotation: [x1, x2] -> [x1*cos - x2*sin, x2*cos + x1*sin]
        let out1 = ((&x1 * &cos)? - (&x2 * &sin)?)?;
        let out2 = ((&x2 * &cos)? + (&x1 * &sin)?)?;

        // Concatenate back
        Ok(Tensor::cat(&[out1, out2], 3)?)
    }

    /// Get the maximum sequence length this embedding supports.
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Get the head dimension.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_creation() {
        let rope = RotaryEmbedding::new(64, 2048, 10000.0, &Device::Cpu).unwrap();
        assert_eq!(rope.head_dim(), 64);
        assert_eq!(rope.max_seq_len(), 2048);
    }

    #[test]
    fn rope_cache_shapes() {
        let rope = RotaryEmbedding::new(64, 2048, 10000.0, &Device::Cpu).unwrap();

        // Caches should be [max_seq_len, half_dim]
        assert_eq!(rope.cos_cache.dims(), &[2048, 32]);
        assert_eq!(rope.sin_cache.dims(), &[2048, 32]);
    }

    #[test]
    fn rope_apply_shape() {
        let rope = RotaryEmbedding::new(64, 2048, 10000.0, &Device::Cpu).unwrap();

        // [batch=1, heads=8, seq_len=16, head_dim=64]
        let query = Tensor::randn(0.0f32, 1.0, &[1, 8, 16, 64], &Device::Cpu).unwrap();
        let key = Tensor::randn(0.0f32, 1.0, &[1, 8, 16, 64], &Device::Cpu).unwrap();

        let (q_rot, k_rot) = rope.apply(&query, &key, 0).unwrap();

        // Output shapes should match input shapes
        assert_eq!(q_rot.dims(), query.dims());
        assert_eq!(k_rot.dims(), key.dims());
    }

    #[test]
    fn rope_apply_with_offset() {
        let rope = RotaryEmbedding::new(64, 2048, 10000.0, &Device::Cpu).unwrap();

        let query = Tensor::randn(0.0f32, 1.0, &[1, 8, 1, 64], &Device::Cpu).unwrap();
        let key = Tensor::randn(0.0f32, 1.0, &[1, 8, 1, 64], &Device::Cpu).unwrap();

        // Apply at position 100
        let (q_rot, k_rot) = rope.apply(&query, &key, 100).unwrap();

        assert_eq!(q_rot.dims(), query.dims());
        assert_eq!(k_rot.dims(), key.dims());
    }

    #[test]
    fn rope_llama3_theta() {
        // Llama-3 uses theta=500000
        let rope = RotaryEmbedding::new(128, 8192, 500000.0, &Device::Cpu).unwrap();
        assert_eq!(rope.head_dim(), 128);
        assert_eq!(rope.max_seq_len(), 8192);
    }

    #[test]
    fn rope_gqa_different_heads() {
        let rope = RotaryEmbedding::new(64, 2048, 10000.0, &Device::Cpu).unwrap();

        // GQA: 32 query heads, 8 KV heads
        let query = Tensor::randn(0.0f32, 1.0, &[1, 32, 16, 64], &Device::Cpu).unwrap();
        let key = Tensor::randn(0.0f32, 1.0, &[1, 8, 16, 64], &Device::Cpu).unwrap();

        let (q_rot, k_rot) = rope.apply(&query, &key, 0).unwrap();

        // Shapes should be preserved
        assert_eq!(q_rot.dims(), &[1, 32, 16, 64]);
        assert_eq!(k_rot.dims(), &[1, 8, 16, 64]);
    }
}
