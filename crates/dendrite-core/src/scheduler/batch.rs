//! Batch types for scheduled execution.

use super::Request;

/// Configuration for batch scheduling.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum requests in a batch.
    pub max_batch_size: usize,
    /// Maximum tokens in a prefill batch.
    pub max_prefill_tokens: usize,
    /// Maximum total tokens across all sequences.
    pub max_total_tokens: usize,
    /// Enable chunked prefill.
    pub chunked_prefill: bool,
    /// Chunk size for chunked prefill.
    pub chunk_size: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 256,
            max_prefill_tokens: 4096,
            max_total_tokens: 32768,
            chunked_prefill: true,
            chunk_size: 512,
        }
    }
}

/// Type of batch execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchType {
    /// Prefill phase (process input tokens).
    Prefill,
    /// Decode phase (generate tokens).
    Decode,
    /// Mixed prefill and decode.
    Mixed,
}

/// A batch of requests for execution.
#[derive(Debug, Clone)]
pub struct Batch {
    /// Type of batch.
    pub batch_type: BatchType,
    /// Requests in this batch.
    pub requests: Vec<Request>,
    /// Total input tokens (for prefill).
    pub num_input_tokens: usize,
    /// Total output tokens (for decode).
    pub num_output_tokens: usize,
}

impl Batch {
    /// Create a prefill batch.
    pub fn prefill(requests: Vec<Request>) -> Self {
        let num_input_tokens: usize = requests.iter().map(|r| r.input_tokens.len()).sum();
        Self {
            batch_type: BatchType::Prefill,
            requests,
            num_input_tokens,
            num_output_tokens: 0,
        }
    }

    /// Create a decode batch.
    pub fn decode(requests: Vec<Request>) -> Self {
        let num_output_tokens = requests.len(); // One token per request
        Self {
            batch_type: BatchType::Decode,
            requests,
            num_input_tokens: 0,
            num_output_tokens,
        }
    }

    /// Create a mixed batch.
    pub fn mixed(prefill_requests: Vec<Request>, decode_requests: Vec<Request>) -> Self {
        let num_input_tokens: usize = prefill_requests.iter().map(|r| r.input_tokens.len()).sum();
        let num_output_tokens = decode_requests.len();

        let mut requests = prefill_requests;
        requests.extend(decode_requests);

        Self {
            batch_type: BatchType::Mixed,
            requests,
            num_input_tokens,
            num_output_tokens,
        }
    }

    /// Get batch size.
    pub fn size(&self) -> usize {
        self.requests.len()
    }

    /// Check if batch is empty.
    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    /// Total tokens in batch.
    pub fn total_tokens(&self) -> usize {
        self.num_input_tokens + self.num_output_tokens
    }
}
