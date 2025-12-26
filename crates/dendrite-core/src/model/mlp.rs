//! SwiGLU MLP (Swish-Gated Linear Unit).
//!
//! SwiGLU is a gated activation function used in modern transformers
//! like Llama and PaLM. It combines SiLU (Swish) with a gating mechanism.
//!
//! # Formula
//!
//! `SwiGLU(x) = (x @ gate_proj) * silu(x @ up_proj) @ down_proj`
//!
//! Where `silu(x) = x * sigmoid(x)`
//!
//! # Reference
//!
//! [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)

use crate::error::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::ops::silu;

/// SwiGLU MLP block.
#[derive(Debug, Clone)]
pub struct SwiGluMlp {
    /// Gate projection: hidden -> intermediate
    gate_proj: Tensor,
    /// Up projection: hidden -> intermediate
    up_proj: Tensor,
    /// Down projection: intermediate -> hidden
    down_proj: Tensor,
    /// Hidden dimension.
    hidden_size: usize,
    /// Intermediate dimension.
    intermediate_size: usize,
}

impl SwiGluMlp {
    /// Create a new SwiGLU MLP with given weights.
    pub fn new(gate_proj: Tensor, up_proj: Tensor, down_proj: Tensor) -> Result<Self> {
        let hidden_size = gate_proj.dims()[1];
        let intermediate_size = gate_proj.dims()[0];

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            hidden_size,
            intermediate_size,
        })
    }

    /// Create a new SwiGLU MLP with random weights (for testing).
    pub fn random(hidden_size: usize, intermediate_size: usize, device: &Device) -> Result<Self> {
        // Weight shapes: [out_features, in_features] for matmul with x @ W^T
        let gate_proj =
            Tensor::randn(0.0f32, 0.02, &[intermediate_size, hidden_size], device)?;
        let up_proj =
            Tensor::randn(0.0f32, 0.02, &[intermediate_size, hidden_size], device)?;
        let down_proj =
            Tensor::randn(0.0f32, 0.02, &[hidden_size, intermediate_size], device)?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            hidden_size,
            intermediate_size,
        })
    }

    /// Create with zeros (for testing specific behaviors).
    pub fn zeros(hidden_size: usize, intermediate_size: usize, device: &Device) -> Result<Self> {
        let gate_proj = Tensor::zeros(&[intermediate_size, hidden_size], DType::F32, device)?;
        let up_proj = Tensor::zeros(&[intermediate_size, hidden_size], DType::F32, device)?;
        let down_proj = Tensor::zeros(&[hidden_size, intermediate_size], DType::F32, device)?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            hidden_size,
            intermediate_size,
        })
    }

    /// Forward pass.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor [..., hidden_size]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims();
        let hidden = dims[dims.len() - 1];

        // Handle 3D inputs by reshaping to 2D
        let (x_2d, batch_seq) = if dims.len() == 3 {
            let batch = dims[0];
            let seq = dims[1];
            let batch_seq = batch * seq;
            (x.reshape((batch_seq, hidden))?, Some((batch, seq)))
        } else {
            (x.clone(), None)
        };

        // x @ gate_proj^T -> [batch_seq, intermediate_size]
        let gate = x_2d.matmul(&self.gate_proj.t()?)?;

        // x @ up_proj^T -> [batch_seq, intermediate_size]
        let up = x_2d.matmul(&self.up_proj.t()?)?;

        // silu(up) * gate
        let activated = (silu(&up)? * gate)?;

        // activated @ down_proj^T -> [batch_seq, hidden_size]
        let output = activated.matmul(&self.down_proj.t()?)?;

        // Reshape back to 3D if input was 3D
        let output = if let Some((batch, seq)) = batch_seq {
            output.reshape((batch, seq, self.hidden_size))?
        } else {
            output
        };

        Ok(output)
    }

    /// Get the hidden size.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Get the intermediate size.
    pub fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }

    /// Load weights from tensors.
    pub fn load_weights(
        &mut self,
        gate_proj: Tensor,
        up_proj: Tensor,
        down_proj: Tensor,
    ) -> Result<()> {
        // Verify shapes
        if gate_proj.dims() != self.gate_proj.dims() {
            return Err(crate::error::DendriteError::ShapeMismatch(format!(
                "gate_proj: expected {:?}, got {:?}",
                self.gate_proj.dims(),
                gate_proj.dims()
            )));
        }
        if up_proj.dims() != self.up_proj.dims() {
            return Err(crate::error::DendriteError::ShapeMismatch(format!(
                "up_proj: expected {:?}, got {:?}",
                self.up_proj.dims(),
                up_proj.dims()
            )));
        }
        if down_proj.dims() != self.down_proj.dims() {
            return Err(crate::error::DendriteError::ShapeMismatch(format!(
                "down_proj: expected {:?}, got {:?}",
                self.down_proj.dims(),
                down_proj.dims()
            )));
        }

        self.gate_proj = gate_proj;
        self.up_proj = up_proj;
        self.down_proj = down_proj;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_mlp() -> SwiGluMlp {
        SwiGluMlp::random(256, 512, &Device::Cpu).unwrap()
    }

    #[test]
    fn mlp_creation() {
        let mlp = create_test_mlp();
        assert_eq!(mlp.hidden_size(), 256);
        assert_eq!(mlp.intermediate_size(), 512);
    }

    #[test]
    fn mlp_forward_2d() {
        let mlp = create_test_mlp();

        // [batch, hidden]
        let x = Tensor::randn(0.0f32, 1.0, &[4, 256], &Device::Cpu).unwrap();
        let output = mlp.forward(&x).unwrap();

        assert_eq!(output.dims(), &[4, 256]);
    }

    #[test]
    fn mlp_forward_3d() {
        let mlp = create_test_mlp();

        // [batch, seq, hidden]
        let x = Tensor::randn(0.0f32, 1.0, &[2, 16, 256], &Device::Cpu).unwrap();
        let output = mlp.forward(&x).unwrap();

        assert_eq!(output.dims(), &[2, 16, 256]);
    }

    #[test]
    fn mlp_zeros_gives_zeros() {
        let mlp = SwiGluMlp::zeros(64, 128, &Device::Cpu).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, &[1, 64], &Device::Cpu).unwrap();
        let output = mlp.forward(&x).unwrap();

        // Zero weights should give zero output
        let sum: f32 = output.abs().unwrap().sum_all().unwrap().to_scalar().unwrap();
        assert!(sum < 1e-6, "Expected near-zero output, got sum={}", sum);
    }

    #[test]
    fn mlp_weight_shapes() {
        let mlp = create_test_mlp();

        assert_eq!(mlp.gate_proj.dims(), &[512, 256]);
        assert_eq!(mlp.up_proj.dims(), &[512, 256]);
        assert_eq!(mlp.down_proj.dims(), &[256, 512]);
    }

    #[test]
    fn mlp_llama3_config() {
        // Llama-3-8B: hidden=4096, intermediate=14336
        let mlp = SwiGluMlp::random(4096, 14336, &Device::Cpu).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, &[1, 32, 4096], &Device::Cpu).unwrap();
        let output = mlp.forward(&x).unwrap();

        assert_eq!(output.dims(), &[1, 32, 4096]);
    }

    #[test]
    fn mlp_load_weights() {
        let mut mlp = SwiGluMlp::zeros(64, 128, &Device::Cpu).unwrap();

        let gate = Tensor::randn(0.0f32, 0.02, &[128, 64], &Device::Cpu).unwrap();
        let up = Tensor::randn(0.0f32, 0.02, &[128, 64], &Device::Cpu).unwrap();
        let down = Tensor::randn(0.0f32, 0.02, &[64, 128], &Device::Cpu).unwrap();

        mlp.load_weights(gate, up, down).unwrap();

        // After loading non-zero weights, output should be non-zero
        let x = Tensor::randn(0.0f32, 1.0, &[1, 64], &Device::Cpu).unwrap();
        let output = mlp.forward(&x).unwrap();
        let sum: f32 = output.abs().unwrap().sum_all().unwrap().to_scalar().unwrap();
        assert!(sum > 0.0);
    }

    #[test]
    fn mlp_silu_activation() {
        // Verify SiLU is applied correctly
        // silu(x) = x * sigmoid(x)
        // For x=0, silu(0) = 0
        // For x>0, silu(x) > 0
        // For x<0, silu(x) < 0 but smaller magnitude

        let x = Tensor::new(&[0.0f32, 1.0, -1.0], &Device::Cpu).unwrap();
        let result = silu(&x).unwrap();
        let vals: Vec<f32> = result.to_vec1().unwrap();

        assert!(vals[0].abs() < 1e-6); // silu(0) ≈ 0
        assert!(vals[1] > 0.0); // silu(1) > 0
        assert!(vals[2] < 0.0); // silu(-1) < 0
        assert!(vals[1].abs() > vals[2].abs()); // |silu(1)| > |silu(-1)|
    }
}
