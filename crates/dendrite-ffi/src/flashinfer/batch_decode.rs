//! Batch decode attention kernel.

use crate::error::{FfiError, Result};

/// Configuration for batch decode.
#[derive(Debug, Clone)]
pub struct BatchDecodeConfig {
    /// Number of sequences in batch.
    pub batch_size: usize,
    /// Number of query heads.
    pub num_qo_heads: usize,
    /// Number of KV heads.
    pub num_kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Page size (tokens per page).
    pub page_size: usize,
    /// Data type (0=f16, 1=bf16).
    pub dtype: u32,
}

/// Batch decode attention kernel wrapper.
pub struct BatchDecodeKernel {
    config: BatchDecodeConfig,
    // Workspace buffer would be here
    workspace_size: usize,
}

impl BatchDecodeKernel {
    /// Create a new batch decode kernel.
    pub fn new(config: BatchDecodeConfig) -> Result<Self> {
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

        // Calculate workspace size
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
    pub fn config(&self) -> &BatchDecodeConfig {
        &self.config
    }

    /// Calculate required workspace size.
    fn calculate_workspace_size(config: &BatchDecodeConfig) -> usize {
        // FlashInfer workspace requirements
        // This is a simplified calculation
        let gqa_ratio = config.num_qo_heads / config.num_kv_heads;
        let base_size = config.batch_size * config.num_qo_heads * config.head_dim * 4;
        base_size * (1 + gqa_ratio)
    }

    /// Run batch decode attention.
    ///
    /// # Arguments
    /// * `q` - Query tensor [batch_size, num_qo_heads, head_dim]
    /// * `paged_kv_cache` - Paged KV cache tensor
    /// * `paged_kv_indices` - Page indices [total_pages]
    /// * `paged_kv_indptr` - Page indptr [batch_size + 1]
    /// * `paged_kv_last_page_len` - Last page lengths [batch_size]
    /// * `output` - Output tensor [batch_size, num_qo_heads, head_dim]
    /// * `workspace` - Workspace buffer
    #[cfg(feature = "cuda")]
    pub fn run(
        &self,
        _q: *const std::ffi::c_void,
        _paged_kv_cache: *const std::ffi::c_void,
        _paged_kv_indices: *const i32,
        _paged_kv_indptr: *const i32,
        _paged_kv_last_page_len: *const i32,
        _output: *mut std::ffi::c_void,
        _workspace: *mut std::ffi::c_void,
        _stream: *mut std::ffi::c_void,
    ) -> Result<()> {
        // TODO: Call FlashInfer kernel via FFI
        Err(FfiError::NotAvailable(
            "FlashInfer FFI not yet implemented".into(),
        ))
    }

    /// Run batch decode attention (no-op without CUDA).
    #[cfg(not(feature = "cuda"))]
    pub fn run(&self) -> Result<()> {
        Err(FfiError::NotAvailable("CUDA not available".into()))
    }
}
