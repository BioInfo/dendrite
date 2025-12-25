//! Cascade attention for shared prefix.

use crate::error::{FfiError, Result};

/// Configuration for cascade attention.
#[derive(Debug, Clone)]
pub struct CascadeConfig {
    /// Number of query heads.
    pub num_qo_heads: usize,
    /// Number of KV heads.
    pub num_kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Shared prefix length.
    pub shared_prefix_len: usize,
    /// Number of sequences sharing the prefix.
    pub num_seqs: usize,
    /// Data type (0=f16, 1=bf16).
    pub dtype: u32,
}

/// Cascade attention kernel for tree-structured KV.
///
/// Cascade attention efficiently handles shared prefixes
/// by computing attention over the shared portion once
/// and combining with per-sequence unique portions.
pub struct CascadeAttention {
    config: CascadeConfig,
    workspace_size: usize,
}

impl CascadeAttention {
    /// Create a new cascade attention kernel.
    pub fn new(config: CascadeConfig) -> Result<Self> {
        if !super::SUPPORTED_HEAD_DIMS.contains(&config.head_dim) {
            return Err(FfiError::InvalidArgument(format!(
                "unsupported head_dim: {}, supported: {:?}",
                config.head_dim,
                super::SUPPORTED_HEAD_DIMS
            )));
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
    pub fn config(&self) -> &CascadeConfig {
        &self.config
    }

    /// Calculate required workspace size.
    fn calculate_workspace_size(config: &CascadeConfig) -> usize {
        // Workspace for shared prefix attention + merge
        let shared_size = config.shared_prefix_len * config.num_qo_heads * config.head_dim * 4;
        let per_seq_size = config.num_seqs * config.num_qo_heads * config.head_dim * 4;
        shared_size + per_seq_size
    }

    /// Run cascade attention.
    ///
    /// This computes attention as:
    /// 1. Attention over shared prefix KV
    /// 2. Attention over per-sequence unique KV
    /// 3. Merge using log-sum-exp combination
    ///
    /// # Arguments
    /// * `q` - Query tensor [num_seqs, 1, num_qo_heads, head_dim]
    /// * `shared_k` - Shared key [shared_prefix_len, num_kv_heads, head_dim]
    /// * `shared_v` - Shared value [shared_prefix_len, num_kv_heads, head_dim]
    /// * `unique_k` - Per-sequence key pages
    /// * `unique_v` - Per-sequence value pages
    /// * `output` - Output tensor [num_seqs, 1, num_qo_heads, head_dim]
    #[cfg(feature = "cuda")]
    pub fn run(
        &self,
        _q: *const std::ffi::c_void,
        _shared_k: *const std::ffi::c_void,
        _shared_v: *const std::ffi::c_void,
        _unique_k: *const std::ffi::c_void,
        _unique_v: *const std::ffi::c_void,
        _unique_kv_indices: *const i32,
        _unique_kv_indptr: *const i32,
        _output: *mut std::ffi::c_void,
        _workspace: *mut std::ffi::c_void,
        _stream: *mut std::ffi::c_void,
    ) -> Result<()> {
        // TODO: Call FlashInfer cascade kernel via FFI
        Err(FfiError::NotAvailable("FlashInfer cascade FFI not yet implemented".into()))
    }

    #[cfg(not(feature = "cuda"))]
    pub fn run(&self) -> Result<()> {
        Err(FfiError::NotAvailable("CUDA not available".into()))
    }
}

/// Merge two attention outputs using log-sum-exp.
///
/// Given outputs and log-sum-exp values from two attention computations,
/// combine them correctly for cascade attention.
#[inline]
pub fn merge_attention_outputs(
    o1: f32,
    lse1: f32,
    o2: f32,
    lse2: f32,
) -> (f32, f32) {
    let max_lse = lse1.max(lse2);
    let exp1 = (lse1 - max_lse).exp();
    let exp2 = (lse2 - max_lse).exp();
    let sum_exp = exp1 + exp2;
    let merged_o = (o1 * exp1 + o2 * exp2) / sum_exp;
    let merged_lse = max_lse + sum_exp.ln();
    (merged_o, merged_lse)
}
