//! Transformer model implementation.
//!
//! Provides the main Transformer struct for LLM inference, combining:
//! - Token embeddings
//! - Stacked transformer layers (attention + MLP)
//! - Final layer norm
//! - Language model head

use super::{
    create_causal_mask, KvCache, ModelConfig, RmsNorm, RotaryEmbedding, TransformerLayer,
    WeightLoader,
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

        // Load layer weights
        self.layers.clear();
        for i in 0..self.config.num_hidden_layers {
            let layer = self.load_layer(&loader, i)?;
            self.layers.push(layer);
        }

        self.weights_loaded = true;
        Ok(())
    }

    /// Load a single transformer layer.
    fn load_layer(&self, loader: &WeightLoader, layer_idx: usize) -> Result<TransformerLayer> {
        let prefix = format!("model.layers.{}", layer_idx);

        // Load attention weights
        let q_proj = loader.get_tensor(&format!("{}.self_attn.q_proj.weight", prefix))?;
        let k_proj = loader.get_tensor(&format!("{}.self_attn.k_proj.weight", prefix))?;
        let v_proj = loader.get_tensor(&format!("{}.self_attn.v_proj.weight", prefix))?;
        let o_proj = loader.get_tensor(&format!("{}.self_attn.o_proj.weight", prefix))?;

        let attention = super::Attention::new(
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            self.config.head_dim(),
        )?;

        // Load MLP weights
        let gate_proj = loader.get_tensor(&format!("{}.mlp.gate_proj.weight", prefix))?;
        let up_proj = loader.get_tensor(&format!("{}.mlp.up_proj.weight", prefix))?;
        let down_proj = loader.get_tensor(&format!("{}.mlp.down_proj.weight", prefix))?;

        let mlp = super::SwiGluMlp::new(gate_proj, up_proj, down_proj)?;

        // Load layer norms
        let input_norm_weight =
            loader.get_tensor(&format!("{}.input_layernorm.weight", prefix))?;
        let input_layernorm = RmsNorm::new(input_norm_weight, self.config.rms_norm_eps)?;

        let post_attn_norm_weight =
            loader.get_tensor(&format!("{}.post_attention_layernorm.weight", prefix))?;
        let post_attention_layernorm =
            RmsNorm::new(post_attn_norm_weight, self.config.rms_norm_eps)?;

        Ok(TransformerLayer::new(
            input_layernorm,
            attention,
            post_attention_layernorm,
            mlp,
            layer_idx,
        ))
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

    /// Create a new KV cache for this model.
    pub fn create_cache(&self) -> KvCache {
        KvCache::new(self.config.num_hidden_layers, self.device.clone())
    }

    /// Forward pass with KV cache for autoregressive generation.
    ///
    /// This method properly maintains KV cache across calls:
    /// - First call (cache empty): processes all tokens, stores KV
    /// - Subsequent calls: uses cached KV, processes only new tokens
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Input token IDs [batch, seq_len]
    /// * `cache` - KV cache to use and update
    ///
    /// # Returns
    ///
    /// Logits [batch, seq_len, vocab_size]
    pub async fn forward_with_cache(
        &self,
        input_ids: &Tensor,
        cache: &mut KvCache,
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

        let seq_len = input_ids.dims()[1];
        let cached_len = cache.seq_len();
        let total_len = cached_len + seq_len;

        // 1. Embed tokens
        let hidden_states = self.embed_lookup(input_ids, embed_tokens)?;

        // 2. Create causal mask for the new tokens attending to all (cached + new)
        let mask = if seq_len > 1 {
            // Prefill: create mask for seq_len tokens
            Some(create_causal_mask(total_len, &self.device)?)
        } else {
            // Decode: single token can attend to all cached + itself, no mask needed
            None
        };

        // 3. Process through layers with cache
        let mut hidden_states = hidden_states;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let layer_cache = cache.layer_mut(layer_idx);
            hidden_states =
                layer.forward_with_cache(&hidden_states, &self.rope, layer_cache, mask.as_ref())?;
        }

        // 4. Final norm
        let hidden_states = self.norm.forward(&hidden_states)?;

        // 5. LM head
        let logits = self.lm_head_forward(&hidden_states, lm_head)?;

        Ok(logits)
    }

    /// Generate tokens autoregressively.
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial token IDs
    /// * `max_tokens` - Maximum tokens to generate
    /// * `temperature` - Sampling temperature (0.0 for greedy)
    ///
    /// # Returns
    ///
    /// Generated token sequence including prompt
    pub async fn generate(
        &self,
        prompt: &[u32],
        max_tokens: usize,
        temperature: f32,
    ) -> Result<Vec<u32>> {
        self.generate_with_stop(prompt, max_tokens, temperature, None).await
    }

    /// Generate tokens with stop condition.
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial token IDs
    /// * `max_tokens` - Maximum tokens to generate
    /// * `temperature` - Sampling temperature (0.0 for greedy)
    /// * `stop_token` - Optional token ID that stops generation
    ///
    /// # Returns
    ///
    /// Generated token sequence including prompt
    pub async fn generate_with_stop(
        &self,
        prompt: &[u32],
        max_tokens: usize,
        temperature: f32,
        stop_token: Option<u32>,
    ) -> Result<Vec<u32>> {
        let mut cache = self.create_cache();
        let mut generated = prompt.to_vec();

        // Prefill: process all prompt tokens at once
        let input = Tensor::from_slice(prompt, (1, prompt.len()), &self.device)?;
        let logits = self.forward_with_cache(&input, &mut cache).await?;
        let next_token = self.sample(&logits, temperature)?;

        if Some(next_token) == stop_token {
            return Ok(generated);
        }
        generated.push(next_token);

        // Decode: generate one token at a time
        for _ in 1..max_tokens {
            let last_token = *generated.last().unwrap();
            let input = Tensor::from_slice(&[last_token], (1, 1), &self.device)?;
            let logits = self.forward_with_cache(&input, &mut cache).await?;
            let next_token = self.sample(&logits, temperature)?;

            if Some(next_token) == stop_token {
                break;
            }
            generated.push(next_token);
        }

        Ok(generated)
    }

    /// Generate text from a text prompt.
    ///
    /// # Arguments
    ///
    /// * `tokenizer` - Tokenizer to use for encoding/decoding
    /// * `prompt` - Text prompt
    /// * `max_tokens` - Maximum tokens to generate
    /// * `temperature` - Sampling temperature (0.0 for greedy)
    ///
    /// # Returns
    ///
    /// Generated text (including prompt)
    pub async fn generate_text(
        &self,
        tokenizer: &super::Tokenizer,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> Result<String> {
        // Encode prompt with BOS token
        let prompt_tokens = tokenizer.encode(prompt, true)?;

        // Generate with EOS as stop token
        let generated = self
            .generate_with_stop(&prompt_tokens, max_tokens, temperature, tokenizer.eos_token_id())
            .await?;

        // Decode back to text
        tokenizer.decode(&generated, false)
    }

    /// Generate continuation from a text prompt.
    ///
    /// Returns only the generated text, not the prompt.
    ///
    /// # Arguments
    ///
    /// * `tokenizer` - Tokenizer to use for encoding/decoding
    /// * `prompt` - Text prompt
    /// * `max_tokens` - Maximum tokens to generate
    /// * `temperature` - Sampling temperature (0.0 for greedy)
    ///
    /// # Returns
    ///
    /// Generated text (continuation only)
    pub async fn generate_continuation(
        &self,
        tokenizer: &super::Tokenizer,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> Result<String> {
        // Encode prompt with BOS token
        let prompt_tokens = tokenizer.encode(prompt, true)?;
        let prompt_len = prompt_tokens.len();

        // Generate with EOS as stop token
        let generated = self
            .generate_with_stop(&prompt_tokens, max_tokens, temperature, tokenizer.eos_token_id())
            .await?;

        // Decode only the new tokens
        if generated.len() > prompt_len {
            tokenizer.decode(&generated[prompt_len..], true)
        } else {
            Ok(String::new())
        }
    }

    /// Stream tokens during generation.
    ///
    /// Calls the callback for each generated token.
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial token IDs
    /// * `max_tokens` - Maximum tokens to generate
    /// * `temperature` - Sampling temperature (0.0 for greedy)
    /// * `stop_token` - Optional token ID that stops generation
    /// * `callback` - Called with each new token
    ///
    /// # Returns
    ///
    /// Generated token sequence including prompt
    pub async fn generate_streaming<F>(
        &self,
        prompt: &[u32],
        max_tokens: usize,
        temperature: f32,
        stop_token: Option<u32>,
        mut callback: F,
    ) -> Result<Vec<u32>>
    where
        F: FnMut(u32) -> bool,
    {
        let mut cache = self.create_cache();
        let mut generated = prompt.to_vec();

        // Prefill: process all prompt tokens at once
        let input = Tensor::from_slice(prompt, (1, prompt.len()), &self.device)?;
        let logits = self.forward_with_cache(&input, &mut cache).await?;
        let next_token = self.sample(&logits, temperature)?;

        if Some(next_token) == stop_token {
            return Ok(generated);
        }
        generated.push(next_token);
        if !callback(next_token) {
            return Ok(generated);
        }

        // Decode: generate one token at a time
        for _ in 1..max_tokens {
            let last_token = *generated.last().unwrap();
            let input = Tensor::from_slice(&[last_token], (1, 1), &self.device)?;
            let logits = self.forward_with_cache(&input, &mut cache).await?;
            let next_token = self.sample(&logits, temperature)?;

            if Some(next_token) == stop_token {
                break;
            }
            generated.push(next_token);
            if !callback(next_token) {
                break;
            }
        }

        Ok(generated)
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
