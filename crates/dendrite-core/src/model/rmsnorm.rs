//! Root Mean Square Layer Normalization.
//!
//! RMSNorm is a simpler alternative to LayerNorm that only
//! normalizes by the root mean square, without centering.
//!
//! # Formula
//!
//! `RMSNorm(x) = x * weight / sqrt(mean(x^2) + eps)`
//!
//! # Reference
//!
//! [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)

use crate::error::Result;
use candle_core::{DType, Device, Tensor};

/// RMS Layer Normalization.
#[derive(Debug, Clone)]
pub struct RmsNorm {
    /// Learnable scale parameter.
    weight: Tensor,
    /// Small constant for numerical stability.
    eps: f64,
    /// Hidden dimension.
    hidden_size: usize,
}

impl RmsNorm {
    /// Create a new RMSNorm layer with given weight.
    pub fn new(weight: Tensor, eps: f64) -> Result<Self> {
        let hidden_size = weight.dims()[0];
        Ok(Self {
            weight,
            eps,
            hidden_size,
        })
    }

    /// Create a new RMSNorm layer with ones (for testing).
    pub fn ones(hidden_size: usize, eps: f64, device: &Device) -> Result<Self> {
        let weight = Tensor::ones(hidden_size, DType::F32, device)?;
        Ok(Self {
            weight,
            eps,
            hidden_size,
        })
    }

    /// Forward pass.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor [..., hidden_size]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Compute x^2
        let x_sq = x.sqr()?;

        // Mean over last dimension
        let mean_sq = x_sq.mean_keepdim(candle_core::D::Minus1)?;

        // 1/sqrt(mean + eps)
        let rsqrt = (mean_sq + self.eps)?.sqrt()?.recip()?;

        // x * rsqrt * weight
        let normalized = x.broadcast_mul(&rsqrt)?;
        let output = normalized.broadcast_mul(&self.weight)?;

        Ok(output)
    }

    /// Get the hidden size.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Get epsilon value.
    pub fn eps(&self) -> f64 {
        self.eps
    }

    /// Get the weight tensor (for loading weights).
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Set weights from another tensor.
    pub fn load_weight(&mut self, weight: Tensor) -> Result<()> {
        if weight.dims() != self.weight.dims() {
            return Err(crate::error::DendriteError::ShapeMismatch(format!(
                "Expected shape {:?}, got {:?}",
                self.weight.dims(),
                weight.dims()
            )));
        }
        self.weight = weight;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_rmsnorm(hidden_size: usize) -> RmsNorm {
        RmsNorm::ones(hidden_size, 1e-5, &Device::Cpu).unwrap()
    }

    #[test]
    fn rmsnorm_creation() {
        let norm = create_test_rmsnorm(4096);
        assert_eq!(norm.hidden_size(), 4096);
        assert!((norm.eps() - 1e-5).abs() < 1e-10);
    }

    #[test]
    fn rmsnorm_forward_2d() {
        let norm = create_test_rmsnorm(64);

        // [batch, hidden]
        let x = Tensor::randn(0.0f32, 1.0, &[4, 64], &Device::Cpu).unwrap();
        let output = norm.forward(&x).unwrap();

        assert_eq!(output.dims(), x.dims());
    }

    #[test]
    fn rmsnorm_forward_3d() {
        let norm = create_test_rmsnorm(64);

        // [batch, seq, hidden]
        let x = Tensor::randn(0.0f32, 1.0, &[2, 16, 64], &Device::Cpu).unwrap();
        let output = norm.forward(&x).unwrap();

        assert_eq!(output.dims(), x.dims());
    }

    #[test]
    fn rmsnorm_normalized_magnitude() {
        let norm = create_test_rmsnorm(64);

        // Create input with known magnitude
        let x = Tensor::ones(&[1, 64], DType::F32, &Device::Cpu).unwrap();
        let x = (&x * 2.0).unwrap(); // All values = 2.0

        let output = norm.forward(&x).unwrap();

        // For constant input with weight=1, output should be close to 1
        // RMS of 2.0 repeated 64 times = 2.0
        // So output = 2.0 / 2.0 * 1.0 = 1.0
        let output_vec: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
        for val in output_vec {
            assert!((val - 1.0).abs() < 1e-4, "Expected ~1.0, got {}", val);
        }
    }

    #[test]
    fn rmsnorm_weight_loading() {
        let mut norm = create_test_rmsnorm(64);

        let new_weight = Tensor::randn(0.0f32, 1.0, 64, &Device::Cpu).unwrap();
        norm.load_weight(new_weight).unwrap();

        // Should have updated
        assert_eq!(norm.weight().dims(), &[64]);
    }

    #[test]
    fn rmsnorm_weight_mismatch_error() {
        let mut norm = create_test_rmsnorm(64);

        let wrong_weight = Tensor::ones(128, DType::F32, &Device::Cpu).unwrap();
        let result = norm.load_weight(wrong_weight);

        assert!(result.is_err());
    }

    #[test]
    fn rmsnorm_llama3_config() {
        // Llama-3-8B uses hidden_size=4096, eps=1e-5
        let norm = RmsNorm::ones(4096, 1e-5, &Device::Cpu).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, &[1, 128, 4096], &Device::Cpu).unwrap();
        let output = norm.forward(&x).unwrap();

        assert_eq!(output.dims(), &[1, 128, 4096]);
    }
}
