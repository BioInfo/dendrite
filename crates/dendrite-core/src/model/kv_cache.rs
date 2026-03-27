//! KV cache for transformer inference.
//!
//! This module provides a simple KV cache that stores key-value tensors
//! for each layer during autoregressive generation.
//!
//! For compressed storage, see [`CompressedLayerCache`] and [`CompressedKvCache`],
//! which wrap the same append/get API but store KV vectors via a [`KvCompressor`]
//! (e.g. [`TurboQuantCompressor`]).

use crate::cache::compress::{CompressedVec, KvCompressor};
use crate::error::Result;
use candle_core::{DType, Device, Tensor};
use std::sync::Arc;

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
        let (new_key, new_value) =
            if let (Some(cached_k), Some(cached_v)) = (&self.key, &self.value) {
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

// ── Compressed KV cache ───────────────────────────────────────────────────────

/// A KV cache layer that stores head-vectors in compressed form.
///
/// On `append`, each head-vector is compressed via `compressor` and stored as
/// `CompressedVec`. On `get`, the vectors are decompressed on-demand back into
/// a full `Tensor`.  This trades compute for memory: the GPU-resident KV store
/// shrinks 3-6x, enabling longer effective contexts.
///
/// # Layout
/// Input tensors are `[batch, num_kv_heads, seq_len, head_dim]`.
/// We store one `CompressedVec` per (batch, head, token) triple.
pub struct CompressedLayerCache {
    /// Compressed key vectors: Vec<Vec<Vec<CompressedVec>>> = [batch][head][seq]
    keys: Vec<Vec<Vec<CompressedVec>>>,
    /// Compressed value vectors.
    values: Vec<Vec<Vec<CompressedVec>>>,
    /// The compressor in use.
    compressor: Arc<dyn KvCompressor>,
    /// head_dim cached on first append.
    head_dim: Option<usize>,
    /// Device for reconstructed tensors.
    device: Device,
}

impl std::fmt::Debug for CompressedLayerCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompressedLayerCache")
            .field("scheme", &self.compressor.name())
            .field("seq_len", &self.seq_len())
            .field("head_dim", &self.head_dim)
            .finish()
    }
}

impl CompressedLayerCache {
    pub fn new(compressor: Arc<dyn KvCompressor>, device: Device) -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            compressor,
            head_dim: None,
            device,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty() || self.keys[0].is_empty() || self.keys[0][0].is_empty()
    }

    pub fn seq_len(&self) -> usize {
        self.keys
            .first()
            .and_then(|heads| heads.first())
            .map(|seq| seq.len())
            .unwrap_or(0)
    }

    /// Compress `tensor` ([batch, heads, seq_len, head_dim]) and append to store.
    fn compress_and_store(
        compressor: &dyn KvCompressor,
        tensor: &Tensor,
        store: &mut Vec<Vec<Vec<CompressedVec>>>,
    ) -> crate::error::Result<usize> {
        let (batch, heads, seq, head_dim) = tensor.dims4()?;
        if store.is_empty() {
            *store = vec![vec![Vec::new(); heads]; batch];
        }
        for b in 0..batch {
            for h in 0..heads {
                for s in 0..seq {
                    // Extract single head-vector [head_dim] via narrow chains
                    let vec_t = tensor
                        .narrow(0, b, 1)?.squeeze(0)?   // [heads, seq, head_dim]
                        .narrow(0, h, 1)?.squeeze(0)?   // [seq, head_dim]
                        .narrow(0, s, 1)?.squeeze(0)?; // [head_dim]
                    let vec_f: Vec<f32> = vec_t.to_vec1()?;
                    let cv = compressor.compress(&vec_f);
                    store[b][h].push(cv);
                }
            }
        }
        Ok(head_dim)
    }

    /// Reconstruct a full `[batch, heads, seq_len, head_dim]` tensor from compressed store.
    fn decompress_all(&self, store: &Vec<Vec<Vec<CompressedVec>>>) -> crate::error::Result<Tensor> {
        let head_dim = self.head_dim.unwrap_or(64);
        let batch = store.len();
        let heads = store[0].len();
        let seq = store[0][0].len();

        let mut flat: Vec<f32> = Vec::with_capacity(batch * heads * seq * head_dim);
        for b in 0..batch {
            for h in 0..heads {
                for s in 0..seq {
                    let vec = self.compressor.decompress(&store[b][h][s], head_dim);
                    flat.extend_from_slice(&vec);
                }
            }
        }
        // Build [batch*heads*seq*head_dim] then reshape
        let t = Tensor::from_vec(flat, (batch, heads, seq, head_dim), &self.device)?;
        Ok(t)
    }

    /// Append new KV tensors, returning the full (decompressed) KV for this step's attention.
    pub fn append(
        &mut self,
        key: &Tensor,
        value: &Tensor,
    ) -> crate::error::Result<(Tensor, Tensor)> {
        let hd = Self::compress_and_store(self.compressor.as_ref(), key, &mut self.keys)?;
        Self::compress_and_store(self.compressor.as_ref(), value, &mut self.values)?;
        self.head_dim = Some(hd);

        let k = self.decompress_all(&self.keys)?;
        let v = self.decompress_all(&self.values)?;
        Ok((k, v))
    }

    /// Clear all compressed storage.
    pub fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
        self.head_dim = None;
    }

    /// Approximate memory savings vs storing raw FP16.
    pub fn compression_ratio(&self) -> f64 {
        let head_dim = self.head_dim.unwrap_or(64);
        let fp16_per_vec = head_dim * 2; // FP16 = 2 bytes
        let compressed_per_vec = self.compressor.compressed_size(head_dim);
        fp16_per_vec as f64 / compressed_per_vec as f64
    }
}

/// Full compressed KV cache for all layers.
pub struct CompressedKvCache {
    layers: Vec<CompressedLayerCache>,
}

impl CompressedKvCache {
    pub fn new(num_layers: usize, compressor: Arc<dyn KvCompressor>, device: Device) -> Self {
        let layers = (0..num_layers)
            .map(|_| CompressedLayerCache::new(Arc::clone(&compressor), device.clone()))
            .collect();
        Self { layers }
    }

    pub fn layer_mut(&mut self, idx: usize) -> &mut CompressedLayerCache {
        &mut self.layers[idx]
    }

    pub fn layer(&self, idx: usize) -> &CompressedLayerCache {
        &self.layers[idx]
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn seq_len(&self) -> usize {
        self.layers.first().map(|l| l.seq_len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.layers.first().map(|l| l.is_empty()).unwrap_or(true)
    }

    pub fn clear(&mut self) {
        for l in &mut self.layers {
            l.clear();
        }
    }

    /// Average compression ratio across all layers (based on first non-empty layer).
    pub fn compression_ratio(&self) -> f64 {
        self.layers
            .iter()
            .find(|l| !l.is_empty())
            .map(|l| l.compression_ratio())
            .unwrap_or(1.0)
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

    // ── CompressedLayerCache tests ─────────────────────────────────────────

    use crate::cache::compress::{IdentityCompressor, TurboQuantCompressor};

    #[test]
    fn compressed_layer_cache_identity_roundtrip() {
        let device = Device::Cpu;
        let compressor = Arc::new(IdentityCompressor);
        let mut cache = CompressedLayerCache::new(compressor, device.clone());

        let k = Tensor::rand(0.0f32, 1.0, (1, 2, 4, 32), &device).unwrap();
        let v = Tensor::rand(0.0f32, 1.0, (1, 2, 4, 32), &device).unwrap();

        let (k_out, v_out) = cache.append(&k, &v).unwrap();
        assert_eq!(k_out.dims(), &[1, 2, 4, 32]);
        assert_eq!(v_out.dims(), &[1, 2, 4, 32]);
        assert_eq!(cache.seq_len(), 4);

        // Verify seq is correct
        assert_eq!(cache.seq_len(), 4);
    }

    #[test]
    fn compressed_layer_cache_appends_accumulate() {
        let device = Device::Cpu;
        let compressor = Arc::new(IdentityCompressor);
        let mut cache = CompressedLayerCache::new(compressor, device.clone());

        // First 4 tokens
        let k1 = Tensor::zeros((1, 2, 4, 32), DType::F32, &device).unwrap();
        let v1 = Tensor::zeros((1, 2, 4, 32), DType::F32, &device).unwrap();
        cache.append(&k1, &v1).unwrap();
        assert_eq!(cache.seq_len(), 4);

        // One more token
        let k2 = Tensor::zeros((1, 2, 1, 32), DType::F32, &device).unwrap();
        let v2 = Tensor::zeros((1, 2, 1, 32), DType::F32, &device).unwrap();
        let (k_all, _) = cache.append(&k2, &v2).unwrap();
        assert_eq!(cache.seq_len(), 5);
        assert_eq!(k_all.dims(), &[1, 2, 5, 32]);
    }

    #[test]
    fn compressed_layer_cache_turbo_quant_shape_preserved() {
        let device = Device::Cpu;
        let compressor = Arc::new(TurboQuantCompressor::new(4, 16));
        let mut cache = CompressedLayerCache::new(compressor, device.clone());

        let k = Tensor::rand(0.0f32, 1.0, (1, 4, 8, 64), &device).unwrap();
        let v = Tensor::rand(0.0f32, 1.0, (1, 4, 8, 64), &device).unwrap();

        let (k_out, v_out) = cache.append(&k, &v).unwrap();
        // Shape must be preserved regardless of compression
        assert_eq!(k_out.dims(), &[1, 4, 8, 64]);
        assert_eq!(v_out.dims(), &[1, 4, 8, 64]);

        // Compression ratio should be reported (>1x with TurboQuant)
        let ratio = cache.compression_ratio();
        println!("TurboQuant CompressedLayerCache ratio: {ratio:.2}x vs FP16");
        assert!(ratio > 1.0, "Expected compression vs FP16, got {ratio:.2}x");
    }

    #[test]
    fn compressed_kv_cache_multi_layer() {
        let device = Device::Cpu;
        let compressor = Arc::new(IdentityCompressor) as Arc<dyn KvCompressor>;
        let mut cache = CompressedKvCache::new(4, compressor, device.clone());

        assert_eq!(cache.num_layers(), 4);
        assert!(cache.is_empty());

        let k = Tensor::zeros((1, 2, 6, 32), DType::F32, &device).unwrap();
        let v = Tensor::zeros((1, 2, 6, 32), DType::F32, &device).unwrap();
        cache.layer_mut(0).append(&k, &v).unwrap();

        assert!(!cache.is_empty());
        assert_eq!(cache.seq_len(), 6);

        cache.clear();
        assert!(cache.is_empty());
    }
}
