//! KV cache for transformer inference.
//!
//! This module provides a simple KV cache that stores key-value tensors
//! for each layer during autoregressive generation.

use crate::error::Result;
use candle_core::{Device, Tensor};

/// KV cache for a single layer.
#[derive(Debug, Clone)]
pub struct LayerCache {
    /// Cached keys: [batch, num_kv_heads, seq_len, head_dim]
    key: Option<Tensor>,
    /// Cached values: [batch, num_kv_heads, seq_len, head_dim]
    value: Option<Tensor>,
}

impl LayerCache {
    /// Create a new empty layer cache.
    pub fn new() -> Self {
        Self {
            key: None,
            value: None,
        }
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.key.is_none()
    }

    /// Get the current sequence length in cache.
    pub fn seq_len(&self) -> usize {
        self.key.as_ref().map(|k| k.dims()[2]).unwrap_or(0)
    }

    /// Append new KV to cache and return concatenated KV.
    ///
    /// Returns (key, value) tensors that include all cached + new tokens.
    pub fn append(&mut self, key: &Tensor, value: &Tensor) -> Result<(Tensor, Tensor)> {
        let (new_key, new_value) = if let (Some(cached_k), Some(cached_v)) =
            (&self.key, &self.value)
        {
            // Concatenate along sequence dimension (dim 2)
            let k = Tensor::cat(&[cached_k, key], 2)?;
            let v = Tensor::cat(&[cached_v, value], 2)?;
            (k, v)
        } else {
            // First tokens - just clone
            (key.clone(), value.clone())
        };

        // Update cache
        self.key = Some(new_key.clone());
        self.value = Some(new_value.clone());

        Ok((new_key, new_value))
    }

    /// Get cached KV without modification.
    pub fn get(&self) -> Option<(&Tensor, &Tensor)> {
        match (&self.key, &self.value) {
            (Some(k), Some(v)) => Some((k, v)),
            _ => None,
        }
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.key = None;
        self.value = None;
    }
}

impl Default for LayerCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Full KV cache for all layers.
#[derive(Debug)]
pub struct KvCache {
    /// Per-layer caches.
    layers: Vec<LayerCache>,
    /// Device for tensors.
    #[allow(dead_code)]
    device: Device,
}

impl KvCache {
    /// Create a new KV cache for the given number of layers.
    pub fn new(num_layers: usize, device: Device) -> Self {
        let layers = (0..num_layers).map(|_| LayerCache::new()).collect();
        Self { layers, device }
    }

    /// Get mutable reference to layer cache.
    pub fn layer_mut(&mut self, layer_idx: usize) -> &mut LayerCache {
        &mut self.layers[layer_idx]
    }

    /// Get reference to layer cache.
    pub fn layer(&self, layer_idx: usize) -> &LayerCache {
        &self.layers[layer_idx]
    }

    /// Get the current sequence length (from first layer).
    pub fn seq_len(&self) -> usize {
        self.layers.first().map(|l| l.seq_len()).unwrap_or(0)
    }

    /// Clear all layer caches.
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.layers.first().map(|l| l.is_empty()).unwrap_or(true)
    }

    /// Get number of layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_cache_empty() {
        let cache = LayerCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.seq_len(), 0);
    }

    #[test]
    fn layer_cache_append() {
        let mut cache = LayerCache::new();
        let device = Device::Cpu;

        // First append: 4 tokens
        let k1 = Tensor::zeros((1, 4, 4, 32), candle_core::DType::F32, &device).unwrap();
        let v1 = Tensor::zeros((1, 4, 4, 32), candle_core::DType::F32, &device).unwrap();

        let (k, v) = cache.append(&k1, &v1).unwrap();
        assert_eq!(k.dims(), &[1, 4, 4, 32]);
        assert_eq!(v.dims(), &[1, 4, 4, 32]);
        assert_eq!(cache.seq_len(), 4);

        // Second append: 1 more token
        let k2 = Tensor::zeros((1, 4, 1, 32), candle_core::DType::F32, &device).unwrap();
        let v2 = Tensor::zeros((1, 4, 1, 32), candle_core::DType::F32, &device).unwrap();

        let (k, v) = cache.append(&k2, &v2).unwrap();
        assert_eq!(k.dims(), &[1, 4, 5, 32]);
        assert_eq!(v.dims(), &[1, 4, 5, 32]);
        assert_eq!(cache.seq_len(), 5);
    }

    #[test]
    fn kv_cache_creation() {
        let cache = KvCache::new(32, Device::Cpu);
        assert_eq!(cache.num_layers(), 32);
        assert!(cache.is_empty());
        assert_eq!(cache.seq_len(), 0);
    }

    #[test]
    fn kv_cache_clear() {
        let mut cache = KvCache::new(2, Device::Cpu);
        let device = Device::Cpu;

        // Add some data
        let k = Tensor::zeros((1, 4, 4, 32), candle_core::DType::F32, &device).unwrap();
        let v = Tensor::zeros((1, 4, 4, 32), candle_core::DType::F32, &device).unwrap();
        cache.layer_mut(0).append(&k, &v).unwrap();

        assert!(!cache.is_empty());
        assert_eq!(cache.seq_len(), 4);

        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.seq_len(), 0);
    }
}
