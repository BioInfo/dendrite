//! SafeTensors weight loading utilities.
//!
//! This module provides utilities for loading model weights from
//! SafeTensors format, which is the standard format for HuggingFace models.
//!
//! # Example
//!
//! ```ignore
//! use dendrite_core::model::WeightLoader;
//!
//! let loader = WeightLoader::new("/path/to/model")?;
//! let tensor = loader.load_tensor("model.embed_tokens.weight")?;
//! ```

use crate::error::{DendriteError, Result};
use candle_core::{DType, Device, Tensor};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Weight loader for SafeTensors format.
pub struct WeightLoader {
    /// Loaded tensors indexed by name.
    tensors: HashMap<String, Tensor>,
    /// Device for loaded tensors.
    device: Device,
}

impl WeightLoader {
    /// Create a new weight loader from a model directory.
    ///
    /// Loads all .safetensors files in the directory.
    pub fn from_dir(dir: &Path, device: &Device) -> Result<Self> {
        let mut tensors = HashMap::new();

        // Find all safetensors files
        let mut safetensor_files: Vec<PathBuf> = Vec::new();
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().map_or(false, |e| e == "safetensors") {
                safetensor_files.push(path);
            }
        }

        // Sort for deterministic loading order
        safetensor_files.sort();

        if safetensor_files.is_empty() {
            return Err(DendriteError::ModelError(format!(
                "No .safetensors files found in {}",
                dir.display()
            )));
        }

        // Load each file
        for path in &safetensor_files {
            let file_tensors = Self::load_safetensors_file(path, device)?;
            tensors.extend(file_tensors);
        }

        Ok(Self {
            tensors,
            device: device.clone(),
        })
    }

    /// Create a weight loader from a single file.
    pub fn from_file(path: &Path, device: &Device) -> Result<Self> {
        let tensors = Self::load_safetensors_file(path, device)?;
        Ok(Self {
            tensors,
            device: device.clone(),
        })
    }

    /// Load tensors from a single safetensors file.
    fn load_safetensors_file(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
        let data = fs::read(path)?;
        let safetensors = SafeTensors::deserialize(&data).map_err(|e| {
            DendriteError::ModelError(format!("Failed to deserialize {}: {}", path.display(), e))
        })?;

        let mut tensors = HashMap::new();

        for (name, view) in safetensors.tensors() {
            let tensor = Self::view_to_tensor(&view, device)?;
            tensors.insert(name.to_string(), tensor);
        }

        Ok(tensors)
    }

    /// Convert a SafeTensors view to a Candle tensor.
    fn view_to_tensor(view: &safetensors::tensor::TensorView, device: &Device) -> Result<Tensor> {
        let shape: Vec<usize> = view.shape().to_vec();
        let dtype = match view.dtype() {
            safetensors::Dtype::F32 => DType::F32,
            safetensors::Dtype::F16 => DType::F16,
            safetensors::Dtype::BF16 => DType::BF16,
            safetensors::Dtype::I64 => DType::I64,
            safetensors::Dtype::I32 => DType::I64, // Candle doesn't have I32, upcast
            safetensors::Dtype::U32 => DType::U32,
            safetensors::Dtype::U8 => DType::U8,
            other => {
                return Err(DendriteError::ModelError(format!(
                    "Unsupported dtype: {:?}",
                    other
                )));
            }
        };

        let data = view.data();

        // Create tensor from raw bytes
        let tensor = match dtype {
            DType::F32 => {
                let values: &[f32] = bytemuck::cast_slice(data);
                Tensor::from_slice(values, shape.as_slice(), device)?
            }
            DType::F16 => {
                let values: &[half::f16] = bytemuck::cast_slice(data);
                Tensor::from_slice(values, shape.as_slice(), device)?
            }
            DType::BF16 => {
                let values: &[half::bf16] = bytemuck::cast_slice(data);
                Tensor::from_slice(values, shape.as_slice(), device)?
            }
            DType::I64 => {
                if view.dtype() == safetensors::Dtype::I32 {
                    // Upcast I32 to I64
                    let values: &[i32] = bytemuck::cast_slice(data);
                    let values: Vec<i64> = values.iter().map(|&x| x as i64).collect();
                    Tensor::from_slice(&values, shape.as_slice(), device)?
                } else {
                    let values: &[i64] = bytemuck::cast_slice(data);
                    Tensor::from_slice(values, shape.as_slice(), device)?
                }
            }
            DType::U32 => {
                let values: &[u32] = bytemuck::cast_slice(data);
                Tensor::from_slice(values, shape.as_slice(), device)?
            }
            DType::U8 => {
                let values: &[u8] = data;
                Tensor::from_slice(values, shape.as_slice(), device)?
            }
            _ => {
                return Err(DendriteError::ModelError(format!(
                    "Unsupported dtype for tensor creation: {:?}",
                    dtype
                )));
            }
        };

        Ok(tensor)
    }

    /// Get a tensor by name.
    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    /// Get a tensor by name, returning an error if not found.
    pub fn get_tensor(&self, name: &str) -> Result<Tensor> {
        self.tensors
            .get(name)
            .cloned()
            .ok_or_else(|| DendriteError::ModelError(format!("Tensor not found: {}", name)))
    }

    /// Check if a tensor exists.
    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Get all tensor names.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }

    /// Get the number of loaded tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Check if no tensors are loaded.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Get the device tensors are loaded to.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get tensors matching a prefix.
    pub fn get_with_prefix(&self, prefix: &str) -> HashMap<&str, &Tensor> {
        self.tensors
            .iter()
            .filter(|(name, _)| name.starts_with(prefix))
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect()
    }
}

/// Map HuggingFace tensor names to Dendrite names.
///
/// Llama-style naming conventions:
/// - `model.embed_tokens.weight` -> `embed_tokens`
/// - `model.layers.0.self_attn.q_proj.weight` -> `layers.0.attn.q_proj`
/// - `model.layers.0.mlp.gate_proj.weight` -> `layers.0.mlp.gate_proj`
/// - `model.norm.weight` -> `norm`
/// - `lm_head.weight` -> `lm_head`
pub fn map_hf_name(hf_name: &str) -> String {
    let name = hf_name
        .strip_prefix("model.")
        .unwrap_or(hf_name)
        .strip_suffix(".weight")
        .unwrap_or(hf_name);

    // Map attention names
    let name = name.replace("self_attn.", "attn.");

    name.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hf_name_mapping() {
        assert_eq!(
            map_hf_name("model.embed_tokens.weight"),
            "embed_tokens"
        );
        assert_eq!(
            map_hf_name("model.layers.0.self_attn.q_proj.weight"),
            "layers.0.attn.q_proj"
        );
        assert_eq!(
            map_hf_name("model.layers.0.mlp.gate_proj.weight"),
            "layers.0.mlp.gate_proj"
        );
        assert_eq!(map_hf_name("model.norm.weight"), "norm");
        assert_eq!(map_hf_name("lm_head.weight"), "lm_head");
    }

    #[test]
    fn loader_from_nonexistent_dir() {
        let result = WeightLoader::from_dir(Path::new("/nonexistent/path"), &Device::Cpu);
        assert!(result.is_err());
    }

    #[test]
    fn loader_empty_dir() {
        let temp_dir = std::env::temp_dir().join("dendrite_test_empty");
        let _ = fs::create_dir_all(&temp_dir);

        let result = WeightLoader::from_dir(&temp_dir, &Device::Cpu);
        assert!(result.is_err());

        let _ = fs::remove_dir_all(&temp_dir);
    }
}
