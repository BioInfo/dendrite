//! Quantization support for KV cache compression.
//!
//! This module provides support for TurboQuant-style KV cache quantization,
//! including packed 4-bit and 2-bit formats with per-token scaling norms.
//!
//! # TurboQuant Format
//!
//! - **4-bit**: Pack 2 indices per byte via (high_nibble << 4) | low_nibble
//! - **2-bit**: Pack 4 indices per byte via (a << 6) | (b << 4) | (c << 2) | d
//! - **Norms**: FP16 scaling factors per token per head

use crate::error::Result;
use candle_core::{Device, Tensor};

/// Unpack 4-bit indices from uint8 bytes.
///
/// Reverses the packing: (high_nibble << 4) | low_nibble
///
/// # Arguments
/// * `packed` - Uint8 tensor with shape [..., packed_dim]
/// * `original_dim` - Original dimension before packing (packed_dim * 2)
///
/// # Returns
/// Uint8 tensor with shape [..., original_dim], values in [0, 15]
pub fn unpack_4bit(packed: &Tensor, original_dim: usize) -> Result<Tensor> {
    let device = packed.device();
    let shape = packed.shape();

    // Flatten to 1D for processing, then reshape
    let flat = packed.flatten_all()?;
    let packed_data = flat.to_vec1::<u8>()?;

    let mut unpacked = Vec::new();
    for &byte in &packed_data {
        let high = (byte >> 4) & 0x0F;
        let low = byte & 0x0F;
        unpacked.push(high);
        unpacked.push(low);
    }

    // Reshape to original dimension
    let mut new_shape = shape.dims().to_vec();
    let last = new_shape.len() - 1;
    new_shape[last] = original_dim;

    Tensor::from_slice(&unpacked, new_shape.as_slice(), device).map_err(|e| e.into())
}

/// Unpack 2-bit indices from uint8 bytes.
///
/// Reverses the packing: (a << 6) | (b << 4) | (c << 2) | d
///
/// # Arguments
/// * `packed` - Uint8 tensor with shape [..., packed_dim]
/// * `original_dim` - Original dimension before packing (packed_dim * 4)
///
/// # Returns
/// Uint8 tensor with shape [..., original_dim], values in [0, 3]
pub fn unpack_2bit(packed: &Tensor, original_dim: usize) -> Result<Tensor> {
    let device = packed.device();
    let shape = packed.shape();

    let flat = packed.flatten_all()?;
    let packed_data = flat.to_vec1::<u8>()?;

    let mut unpacked = Vec::new();
    for &byte in &packed_data {
        let a = (byte >> 6) & 0x03;
        let b = (byte >> 4) & 0x03;
        let c = (byte >> 2) & 0x03;
        let d = byte & 0x03;
        unpacked.push(a);
        unpacked.push(b);
        unpacked.push(c);
        unpacked.push(d);
    }

    // Reshape to original dimension
    let mut new_shape = shape.dims().to_vec();
    let last = new_shape.len() - 1;
    new_shape[last] = original_dim;

    Tensor::from_slice(&unpacked, new_shape.as_slice(), device).map_err(|e| e.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unpack_4bit() -> Result<()> {
        let device = Device::Cpu;

        // Pack: [0, 1, 2, 3] -> [(0 << 4) | 1, (2 << 4) | 3] = [0x01, 0x23]
        let packed = Tensor::from_slice(&[0x01u8, 0x23], (2,), &device)?;
        let unpacked = unpack_4bit(&packed, 4)?;

        let data: Vec<u8> = unpacked.to_vec1()?;
        assert_eq!(data, vec![0, 1, 2, 3]);

        Ok(())
    }

    #[test]
    fn test_unpack_2bit() -> Result<()> {
        let device = Device::Cpu;

        // Pack: [0, 1, 2, 3] -> (0 << 6) | (1 << 4) | (2 << 2) | 3 = 0b00_01_10_11 = 0x1B
        let packed = Tensor::from_slice(&[0x1Bu8], (1,), &device)?;
        let unpacked = unpack_2bit(&packed, 4)?;

        let data: Vec<u8> = unpacked.to_vec1()?;
        assert_eq!(data, vec![0, 1, 2, 3]);

        Ok(())
    }

    #[test]
    #[test]
    fn test_unpack_4bit_multidim() -> Result<()> {
        let device = Device::Cpu;

        // 2x2 packed -> 2x4 unpacked
        let packed = Tensor::from_slice(&[0x01u8, 0x23, 0x45, 0x67], (2, 2), &device)?;
        let unpacked = unpack_4bit(&packed, 4)?;

        assert_eq!(unpacked.dims(), [2, 4]);
        let data: Vec<Vec<u8>> = unpacked.to_vec2()?;
        assert_eq!(data, vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]]);

        Ok(())
    }
}
