//! Batch prefill attention kernel.

use crate::error::{FfiError, Result};

/// Configuration for batch prefill.
#[derive(Debug, Clone)]
pub struct BatchPrefillConfig {
    /// Total number of tokens across all sequences.
    pub total_tokens: usize,
    /// Number of sequences.
    pub num_seqs: usize,
    /// Number of query heads.
    pub num_qo_heads: usize,
    /// Number of KV heads.
    pub num_kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Page size (tokens per page).
    pub page_size: usize,
    /// Enable causal masking.
    pub causal: bool,
    /// Data type (0=f16, 1=bf16).
    pub dtype: u32,
}

/// Batch prefill attention kernel wrapper.
pub struct BatchPrefillKernel {
    config: BatchPrefillConfig,
    workspace_size: usize,
}

impl BatchPrefillKernel {
    /// Create a new batch prefill kernel.
    pub fn new(config: BatchPrefillConfig) -> Result<Self> {
        // Validate configuration
        if !super::SUPPORTED_HEAD_DIMS.contains(&config.head_dim) {
            return Err(FfiError::InvalidArgument(format!(
                "unsupported head_dim: {}, supported: {:?}",
                config.head_dim,
                super::SUPPORTED_HEAD_DIMS
            )));
        }

        if config.num_qo_heads % config.num_kv_heads != 0 {
            return Err(FfiError::InvalidArgument(
                "num_qo_heads must be divisible by num_kv_heads".into(),
            ));
        }

        let workspace_size = Self::calculate_workspace_size(&config);

        Ok(Self {
            config,
            workspace_size,
        })
    }

    /// Get workspace size in bytes.
    pub fn workspace_size(&self) -> usize {
        self.workspace_size
    }

    /// Get configuration.
    pub fn config(&self) -> &BatchPrefillConfig {
        &self.config
    }

    /// Calculate required workspace size.
    fn calculate_workspace_size(config: &BatchPrefillConfig) -> usize {
        // FlashInfer prefill workspace requirements
        let gqa_ratio = config.num_qo_heads / config.num_kv_heads;
        let base_size = config.total_tokens * config.num_qo_heads * config.head_dim * 4;
        base_size * (1 + gqa_ratio)
    }

    /// Run batch prefill attention.
    ///
    /// # Arguments
    /// * `q` - Query tensor [total_tokens, num_qo_heads, head_dim]
    /// * `k` - Key tensor [total_tokens, num_kv_heads, head_dim]
    /// * `v` - Value tensor [total_tokens, num_kv_heads, head_dim]
    /// * `qo_indptr` - Query indptr [num_seqs + 1]
    /// * `kv_indptr` - KV indptr [num_seqs + 1]
    /// * `output` - Output tensor [total_tokens, num_qo_heads, head_dim]
    /// * `workspace` - Workspace buffer
    #[cfg(feature = "cuda")]
    pub fn run(
        &self,
        _q: *const std::ffi::c_void,
        _k: *const std::ffi::c_void,
        _v: *const std::ffi::c_void,
        _qo_indptr: *const i32,
        _kv_indptr: *const i32,
        _output: *mut std::ffi::c_void,
        _workspace: *mut std::ffi::c_void,
        _stream: *mut std::ffi::c_void,
    ) -> Result<()> {
        // TODO: Call FlashInfer kernel via FFI
        Err(FfiError::NotAvailable(
            "FlashInfer FFI not yet implemented".into(),
        ))
    }

    /// Run batch prefill attention (no-op without CUDA).
    #[cfg(not(feature = "cuda"))]
    pub fn run(&self) -> Result<()> {
        Err(FfiError::NotAvailable("CUDA not available".into()))
    }
}
