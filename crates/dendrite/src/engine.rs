//! High-level inference engine.

use anyhow::Result;
use dendrite_core::{
    attention::ReferenceBackend,
    cache::{KvCache, KvCacheConfig},
    model::{ModelConfig, Transformer},
    scheduler::{BatchConfig, Request, Scheduler},
    tree::TreeState,
};
use parking_lot::RwLock;
use std::path::PathBuf;
use std::sync::Arc;

/// Configuration for the inference engine.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Path to model weights.
    pub model_path: PathBuf,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Maximum batch size.
    pub max_batch_size: usize,
    /// GPU memory fraction to use for KV cache.
    pub gpu_memory_fraction: f32,
    /// Enable tensor parallelism.
    pub tensor_parallel: usize,
    /// Tokens per KV cache block.
    pub tokens_per_block: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            max_seq_len: 8192,
            max_batch_size: 256,
            gpu_memory_fraction: 0.9,
            tensor_parallel: 1,
            tokens_per_block: 16,
        }
    }
}

/// Builder for creating an Engine.
pub struct EngineBuilder {
    config: EngineConfig,
    model_config: Option<ModelConfig>,
}

impl EngineBuilder {
    /// Create a new engine builder.
    pub fn new() -> Self {
        Self {
            config: EngineConfig::default(),
            model_config: None,
        }
    }

    /// Set model path.
    pub fn model_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.model_path = path.into();
        self
    }

    /// Set maximum sequence length.
    pub fn max_seq_len(mut self, len: usize) -> Self {
        self.config.max_seq_len = len;
        self
    }

    /// Set maximum batch size.
    pub fn max_batch_size(mut self, size: usize) -> Self {
        self.config.max_batch_size = size;
        self
    }

    /// Set GPU memory fraction.
    pub fn gpu_memory_fraction(mut self, fraction: f32) -> Self {
        self.config.gpu_memory_fraction = fraction;
        self
    }

    /// Set tensor parallel size.
    pub fn tensor_parallel(mut self, size: usize) -> Self {
        self.config.tensor_parallel = size;
        self
    }

    /// Set model configuration directly.
    pub fn model_config(mut self, config: ModelConfig) -> Self {
        self.model_config = Some(config);
        self
    }

    /// Build the engine.
    pub async fn build(self) -> Result<Engine> {
        let model_config = self.model_config.unwrap_or_default();

        // Calculate KV cache size
        let max_blocks = self.config.max_seq_len / self.config.tokens_per_block
            * self.config.max_batch_size;

        let kv_config = KvCacheConfig {
            num_layers: model_config.num_hidden_layers,
            num_kv_heads: model_config.num_key_value_heads,
            head_dim: model_config.head_dim(),
            max_blocks,
            tokens_per_block: self.config.tokens_per_block,
        };

        let kv_cache = Arc::new(RwLock::new(KvCache::new(kv_config)?));
        let tree_state = Arc::new(TreeState::new(
            kv_cache.clone(),
            self.config.tokens_per_block,
        ));

        let batch_config = BatchConfig {
            max_batch_size: self.config.max_batch_size,
            max_total_tokens: self.config.max_seq_len * self.config.max_batch_size,
            ..Default::default()
        };

        let scheduler = Scheduler::new(batch_config, tree_state.clone());

        // Create model with reference attention backend
        let attention = Arc::new(ReferenceBackend);
        let device = candle_core::Device::Cpu; // TODO: GPU support
        let model = Transformer::new(model_config, attention, device);

        Ok(Engine {
            config: self.config,
            model,
            kv_cache,
            tree_state,
            scheduler,
        })
    }
}

impl Default for EngineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// High-level inference engine.
pub struct Engine {
    config: EngineConfig,
    model: Transformer,
    kv_cache: Arc<RwLock<KvCache>>,
    tree_state: Arc<TreeState>,
    scheduler: Scheduler,
}

impl Engine {
    /// Create a new engine builder.
    pub fn builder() -> EngineBuilder {
        EngineBuilder::new()
    }

    /// Get engine configuration.
    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    /// Get model reference.
    pub fn model(&self) -> &Transformer {
        &self.model
    }

    /// Get tree state.
    pub fn tree_state(&self) -> &Arc<TreeState> {
        &self.tree_state
    }

    /// Get scheduler.
    pub fn scheduler(&self) -> &Scheduler {
        &self.scheduler
    }

    /// Generate text from a prompt.
    pub fn generate(&self, prompt: &str) -> GenerateRequest<'_> {
        GenerateRequest {
            engine: self,
            prompt: prompt.to_string(),
            max_tokens: 256,
            temperature: 1.0,
            tree_search: None,
        }
    }

    /// Fork from current state for tree search.
    pub fn fork(&self, from_node: dendrite_core::tree::NodeId) -> Result<dendrite_core::tree::ForkHandle> {
        Ok(self.tree_state.fork(from_node)?)
    }

    /// Get KV cache statistics.
    pub fn cache_stats(&self) -> CacheStats {
        let cache = self.kv_cache.read();
        CacheStats {
            total_blocks: cache.total_blocks(),
            free_blocks: cache.free_blocks(),
            used_blocks: cache.total_blocks() - cache.free_blocks(),
        }
    }
}

/// A generation request.
pub struct GenerateRequest<'a> {
    engine: &'a Engine,
    prompt: String,
    max_tokens: usize,
    temperature: f32,
    tree_search: Option<TreeSearchConfig>,
}

impl<'a> GenerateRequest<'a> {
    /// Set maximum tokens to generate.
    pub fn max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = n;
        self
    }

    /// Set temperature.
    pub fn temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }

    /// Enable tree search.
    pub fn with_tree_search(mut self, config: TreeSearchConfig) -> Self {
        self.tree_search = Some(config);
        self
    }

    /// Execute the generation.
    pub async fn execute(self) -> Result<GenerateResult> {
        // TODO: Implement actual generation
        // 1. Tokenize prompt
        // 2. Run prefill
        // 3. Run decode loop (with optional tree search)
        // 4. Detokenize output

        Ok(GenerateResult {
            text: String::new(),
            tokens: Vec::new(),
            num_prompt_tokens: 0,
            num_generated_tokens: 0,
        })
    }
}

/// Configuration for tree search.
#[derive(Debug, Clone, Default)]
pub struct TreeSearchConfig {
    /// Number of branches per node.
    pub num_branches: usize,
    /// Maximum depth.
    pub max_depth: usize,
    /// Scoring function.
    pub scoring: ScoringMethod,
}

/// Scoring method for tree search.
#[derive(Debug, Clone, Default)]
pub enum ScoringMethod {
    /// Use log probabilities.
    #[default]
    LogProb,
    /// Use a reward model.
    RewardModel,
    /// Custom scoring function.
    Custom,
}

/// Result of text generation.
#[derive(Debug)]
pub struct GenerateResult {
    /// Generated text.
    pub text: String,
    /// Generated token IDs.
    pub tokens: Vec<u32>,
    /// Number of prompt tokens.
    pub num_prompt_tokens: usize,
    /// Number of generated tokens.
    pub num_generated_tokens: usize,
}

/// KV cache statistics.
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Total blocks in cache.
    pub total_blocks: usize,
    /// Free blocks available.
    pub free_blocks: usize,
    /// Blocks currently in use.
    pub used_blocks: usize,
}

impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine")
            .field("config", &self.config)
            .field("model", &self.model)
            .finish()
    }
}
