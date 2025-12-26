//! Transformer model implementation.
//!
//! Provides the main Transformer struct for LLM inference, combining:
//! - Token embeddings
//! - Stacked transformer layers (attention + MLP)
//! - Final layer norm
//! - Language model head

use super::{
    create_causal_mask, ModelConfig, RmsNorm, RotaryEmbedding, TransformerLayer, WeightLoader,
};
use crate::attention::{AttentionBackend, AttentionConfig};
use crate::cache::BlockTable;
use crate::error::{DendriteError, Result};
use candle_core::{Device, Tensor};
use std::path::Path;
use std::sync::Arc;

/// A transformer model for inference.
pub struct Transformer {
    /// Model configuration.
    config: ModelConfig,
    /// Token embeddings: [vocab_size, hidden_size]
    embed_tokens: Option<Tensor>,
    /// Transformer layers.
    layers: Vec<TransformerLayer>,
    /// Rotary position embeddings.
    rope: RotaryEmbedding,
    /// Final layer normalization.
    norm: RmsNorm,
    /// Language model head: [vocab_size, hidden_size]
    lm_head: Option<Tensor>,
    /// Attention backend.
    #[allow(dead_code)]
    attention: Arc<dyn AttentionBackend>,
    /// Device for computation.
    device: Device,
    /// Whether weights are loaded.
    weights_loaded: bool,
}

impl Transformer {
    /// Create a new transformer (weights not loaded).
    pub fn new(
        config: ModelConfig,
        attention: Arc<dyn AttentionBackend>,
        device: Device,
    ) -> Result<Self> {
        // Create RoPE
        let rope = RotaryEmbedding::new(
            config.head_dim(),
            config.max_position_embeddings,
            config.rope_theta,
            &device,
        )?;

        // Create final norm with ones (placeholder)
        let norm = RmsNorm::ones(config.hidden_size, config.rms_norm_eps, &device)?;

        // Create placeholder layers (will be replaced when weights load)
        let layers = Vec::new();

        Ok(Self {
            config,
            embed_tokens: None,
            layers,
            rope,
            norm,
            lm_head: None,
            attention,
            device,
            weights_loaded: false,
        })
    }

    /// Create transformer with random weights (for testing).
    pub fn random(
        config: ModelConfig,
        attention: Arc<dyn AttentionBackend>,
        device: Device,
    ) -> Result<Self> {
        let rope = RotaryEmbedding::new(
            config.head_dim(),
            config.max_position_embeddings,
            config.rope_theta,
            &device,
        )?;

        let norm = RmsNorm::ones(config.hidden_size, config.rms_norm_eps, &device)?;

        // Create random embedding
        let embed_tokens = Some(Tensor::randn(
            0.0f32,
            0.02,
            &[config.vocab_size, config.hidden_size],
            &device,
        )?);

        // Create random layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer = TransformerLayer::random(
                config.hidden_size,
                config.intermediate_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim(),
                config.rms_norm_eps,
                i,
                &device,
            )?;
            layers.push(layer);
        }

        // Create random lm_head
        let lm_head = Some(Tensor::randn(
            0.0f32,
            0.02,
            &[config.vocab_size, config.hidden_size],
            &device,
        )?);

        Ok(Self {
            config,
            embed_tokens,
            layers,
            rope,
            norm,
            lm_head,
            attention,
            device,
            weights_loaded: true,
        })
    }

    /// Load weights from a SafeTensors directory.
    pub fn load_weights(&mut self, model_dir: &Path) -> Result<()> {
        let loader = WeightLoader::from_dir(model_dir, &self.device)?;

        // Load embeddings
        self.embed_tokens = Some(loader.get_tensor("model.embed_tokens.weight")?);

        // Load lm_head (may be tied to embeddings)
        self.lm_head = if loader.contains("lm_head.weight") {
            Some(loader.get_tensor("lm_head.weight")?)
        } else {
            // Tied embeddings
            self.embed_tokens.clone()
        };

        // Load final norm
        let norm_weight = loader.get_tensor("model.norm.weight")?;
        self.norm = RmsNorm::new(norm_weight, self.config.rms_norm_eps)?;

        // TODO: Load layer weights
        // For now, keep placeholder layers

        self.weights_loaded = true;
        Ok(())
    }

    /// Get model configuration.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Get attention configuration for this model.
    pub fn attention_config(&self) -> AttentionConfig {
        AttentionConfig::new(
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            self.config.head_dim(),
        )
    }

    /// Get device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Check if weights are loaded.
    pub fn weights_loaded(&self) -> bool {
        self.weights_loaded
    }

    /// Get number of layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Forward pass for prefill (process input tokens).
    ///
    /// This processes a sequence of tokens and returns logits.
    pub async fn prefill(&self, input_ids: &Tensor, _block_table: &BlockTable) -> Result<Tensor> {
        if !self.weights_loaded {
            return Err(DendriteError::ModelError("Weights not loaded".into()));
        }

        let embed_tokens = self
            .embed_tokens
            .as_ref()
            .ok_or_else(|| DendriteError::ModelError("No embedding weights".into()))?;
        let lm_head = self
            .lm_head
            .as_ref()
            .ok_or_else(|| DendriteError::ModelError("No lm_head weights".into()))?;

        let seq_len = input_ids.dims()[1];

        // 1. Embed tokens: input_ids -> hidden_states
        let hidden_states = self.embed_lookup(input_ids, embed_tokens)?;

        // 2. Create causal mask
        let mask = create_causal_mask(seq_len, &self.device)?;

        // 3. Process through layers
        let mut hidden_states = hidden_states;
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, &self.rope, 0, Some(&mask))?;
        }

        // 4. Final norm
        let hidden_states = self.norm.forward(&hidden_states)?;

        // 5. LM head: hidden_states @ lm_head^T -> logits
        let logits = self.lm_head_forward(&hidden_states, lm_head)?;

        Ok(logits)
    }

    /// Forward pass for decode (single token).
    ///
    /// This processes a single token with position offset.
    pub async fn decode(
        &self,
        input_ids: &Tensor,
        _block_table: &BlockTable,
        position: usize,
    ) -> Result<Tensor> {
        if !self.weights_loaded {
            return Err(DendriteError::ModelError("Weights not loaded".into()));
        }

        let embed_tokens = self
            .embed_tokens
            .as_ref()
            .ok_or_else(|| DendriteError::ModelError("No embedding weights".into()))?;
        let lm_head = self
            .lm_head
            .as_ref()
            .ok_or_else(|| DendriteError::ModelError("No lm_head weights".into()))?;

        // 1. Embed token
        let hidden_states = self.embed_lookup(input_ids, embed_tokens)?;

        // 2. No mask needed for single token decode (causal is implicit)
        let mut hidden_states = hidden_states;
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, &self.rope, position, None)?;
        }

        // 3. Final norm
        let hidden_states = self.norm.forward(&hidden_states)?;

        // 4. LM head
        let logits = self.lm_head_forward(&hidden_states, lm_head)?;

        Ok(logits)
    }

    /// Embedding lookup.
    fn embed_lookup(&self, input_ids: &Tensor, embed_tokens: &Tensor) -> Result<Tensor> {
        // input_ids: [batch, seq] of token IDs
        // embed_tokens: [vocab_size, hidden_size]
        // output: [batch, seq, hidden_size]

        let dims = input_ids.dims();
        let batch = dims[0];
        let seq = dims[1];

        // Flatten input_ids to 1D for indexing
        let flat_ids = input_ids.flatten_all()?;

        // Index select: [batch*seq] indices into [vocab, hidden] -> [batch*seq, hidden]
        let embeddings = embed_tokens.index_select(&flat_ids, 0)?;

        // Reshape back to [batch, seq, hidden]
        let hidden = embed_tokens.dims()[1];
        let embeddings = embeddings.reshape((batch, seq, hidden))?;

        Ok(embeddings)
    }

    /// LM head forward pass.
    fn lm_head_forward(&self, hidden_states: &Tensor, lm_head: &Tensor) -> Result<Tensor> {
        let dims = hidden_states.dims();
        let batch = dims[0];
        let seq = dims[1];
        let hidden = dims[2];

        // Reshape to 2D
        let h = hidden_states.reshape((batch * seq, hidden))?;

        // h @ lm_head^T
        let logits = h.matmul(&lm_head.t()?)?;

        // Reshape back
        let logits = logits.reshape((batch, seq, self.config.vocab_size))?;

        Ok(logits)
    }

    /// Sample from logits.
    ///
    /// Takes logits of shape [batch, seq, vocab] and samples from the last position.
    pub fn sample(&self, logits: &Tensor, temperature: f32) -> Result<u32> {
        let dims = logits.dims();

        // Get last position logits: [batch, seq, vocab] -> [vocab]
        // Take batch 0, last seq position
        let seq_len = dims[1];
        let last_logits = logits.narrow(0, 0, 1)?; // [1, seq, vocab]
        let last_logits = last_logits.narrow(1, seq_len - 1, 1)?; // [1, 1, vocab]
        let last_logits = last_logits.squeeze(0)?.squeeze(0)?; // [vocab]

        if temperature == 0.0 {
            // Greedy - argmax returns the index of max value
            let token = last_logits.argmax(0)?.to_scalar::<u32>()?;
            Ok(token)
        } else {
            // Temperature sampling
            let scaled_logits = (&last_logits / temperature as f64)?;
            let probs = candle_nn::ops::softmax(&scaled_logits, 0)?;
            // TODO: Proper multinomial sampling - for now use argmax
            let token = probs.argmax(0)?.to_scalar::<u32>()?;
            Ok(token)
        }
    }
}

impl std::fmt::Debug for Transformer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Transformer")
            .field("config", &self.config)
            .field("num_layers", &self.layers.len())
            .field("weights_loaded", &self.weights_loaded)
            .field("device", &self.device)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::ReferenceBackend;

    fn create_test_config() -> ModelConfig {
        ModelConfig {
            vocab_size: 1000,
            hidden_size: 256,
            intermediate_size: 512,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_hidden_layers: 2,
            max_position_embeddings: 512,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            head_dim: 64,
            model_type: "test".to_string(),
        }
    }

    #[test]
    fn transformer_creation() {
        let config = create_test_config();
        let attention: Arc<dyn AttentionBackend> = Arc::new(ReferenceBackend::new());
        let transformer = Transformer::new(config.clone(), attention, Device::Cpu).unwrap();

        assert_eq!(transformer.config().vocab_size, 1000);
        assert_eq!(transformer.config().hidden_size, 256);
        assert!(!transformer.weights_loaded());
    }

    #[test]
    fn transformer_random() {
        let config = create_test_config();
        let attention: Arc<dyn AttentionBackend> = Arc::new(ReferenceBackend::new());
        let transformer = Transformer::random(config.clone(), attention, Device::Cpu).unwrap();

        assert!(transformer.weights_loaded());
        assert_eq!(transformer.num_layers(), 2);
    }

    #[test]
    fn transformer_attention_config() {
        let config = create_test_config();
        let attention: Arc<dyn AttentionBackend> = Arc::new(ReferenceBackend::new());
        let transformer = Transformer::new(config, attention, Device::Cpu).unwrap();

        let attn_config = transformer.attention_config();
        assert_eq!(attn_config.num_heads, 4);
        assert_eq!(attn_config.num_kv_heads, 2);
        assert_eq!(attn_config.head_dim, 64);
    }

    #[tokio::test]
    async fn transformer_e2e_prefill() {
        let config = create_test_config();
        let attention: Arc<dyn AttentionBackend> = Arc::new(ReferenceBackend::new());
        let transformer = Transformer::random(config.clone(), attention, Device::Cpu).unwrap();

        // Create input tokens: [batch=1, seq=8]
        let input_ids = Tensor::from_slice(&[1u32, 42, 100, 200, 300, 400, 500, 600], (1, 8), &Device::Cpu).unwrap();

        // Create block table (not used in this test, but required by API)
        let block_table = crate::cache::BlockTable::new(16);

        // Run prefill
        let logits = transformer.prefill(&input_ids, &block_table).await.unwrap();

        // Verify output shape: [batch=1, seq=8, vocab=1000]
        assert_eq!(logits.dims(), &[1, 8, 1000]);

        // Verify logits are not all zeros (random weights should produce non-zero output)
        let sum: f32 = logits.abs().unwrap().sum_all().unwrap().to_scalar().unwrap();
        assert!(sum > 0.0, "Logits should not be all zeros");
    }

    #[tokio::test]
    async fn transformer_e2e_decode() {
        let config = create_test_config();
        let attention: Arc<dyn AttentionBackend> = Arc::new(ReferenceBackend::new());
        let transformer = Transformer::random(config.clone(), attention, Device::Cpu).unwrap();

        // Create single token input: [batch=1, seq=1]
        let input_ids = Tensor::from_slice(&[42u32], (1, 1), &Device::Cpu).unwrap();
        let block_table = crate::cache::BlockTable::new(16);

        // Run decode at position 10 (simulating continued generation)
        let logits = transformer.decode(&input_ids, &block_table, 10).await.unwrap();

        // Verify output shape: [batch=1, seq=1, vocab=1000]
        assert_eq!(logits.dims(), &[1, 1, 1000]);
    }

    #[tokio::test]
    async fn transformer_e2e_sample() {
        let config = create_test_config();
        let attention: Arc<dyn AttentionBackend> = Arc::new(ReferenceBackend::new());
        let transformer = Transformer::random(config.clone(), attention, Device::Cpu).unwrap();

        // Create input and run prefill
        let input_ids = Tensor::from_slice(&[1u32, 2, 3, 4], (1, 4), &Device::Cpu).unwrap();
        let block_table = crate::cache::BlockTable::new(16);
        let logits = transformer.prefill(&input_ids, &block_table).await.unwrap();

        // Sample with greedy (temperature=0)
        let token = transformer.sample(&logits, 0.0).unwrap();
        assert!(token < 1000, "Token should be within vocab size");

        // Sample with temperature
        let token_temp = transformer.sample(&logits, 1.0).unwrap();
        assert!(token_temp < 1000, "Token should be within vocab size");
    }

    #[tokio::test]
    async fn transformer_e2e_generation_loop() {
        let config = create_test_config();
        let attention: Arc<dyn AttentionBackend> = Arc::new(ReferenceBackend::new());
        let transformer = Transformer::random(config.clone(), attention, Device::Cpu).unwrap();

        let block_table = crate::cache::BlockTable::new(16);

        // Initial prompt
        let prompt = vec![1u32, 42, 100];
        let mut generated = prompt.clone();

        // Generate 5 tokens
        for i in 0..5 {
            let input = if i == 0 {
                // Prefill with full prompt
                Tensor::from_slice(&prompt, (1, prompt.len()), &Device::Cpu).unwrap()
            } else {
                // Decode single token
                let last_token = *generated.last().unwrap();
                Tensor::from_slice(&[last_token], (1, 1), &Device::Cpu).unwrap()
            };

            let logits = if i == 0 {
                transformer.prefill(&input, &block_table).await.unwrap()
            } else {
                transformer.decode(&input, &block_table, generated.len() - 1).await.unwrap()
            };

            let next_token = transformer.sample(&logits, 0.0).unwrap();
            generated.push(next_token);
        }

        // Should have prompt + 5 generated tokens
        assert_eq!(generated.len(), 8);

        // All tokens should be valid
        for &token in &generated {
            assert!(token < 1000, "Token {} should be within vocab size", token);
        }
    }

    #[tokio::test]
    async fn transformer_e2e_batch_prefill() {
        let config = create_test_config();
        let attention: Arc<dyn AttentionBackend> = Arc::new(ReferenceBackend::new());
        let transformer = Transformer::random(config.clone(), attention, Device::Cpu).unwrap();

        // Create batched input: [batch=2, seq=4]
        let input_ids = Tensor::from_slice(
            &[1u32, 2, 3, 4, 10, 20, 30, 40],
            (2, 4),
            &Device::Cpu,
        ).unwrap();
        let block_table = crate::cache::BlockTable::new(16);

        let logits = transformer.prefill(&input_ids, &block_table).await.unwrap();

        // Verify output shape: [batch=2, seq=4, vocab=1000]
        assert_eq!(logits.dims(), &[2, 4, 1000]);
    }
}
