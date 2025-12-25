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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::request::Request;

    fn create_test_request(input_len: usize) -> Request {
        let tokens: Vec<u32> = (0..input_len as u32).collect();
        Request::new(tokens, 10)
    }

    #[test]
    fn batch_config_default() {
        let config = BatchConfig::default();

        assert_eq!(config.max_batch_size, 256);
        assert_eq!(config.max_prefill_tokens, 4096);
        assert_eq!(config.max_total_tokens, 32768);
        assert!(config.chunked_prefill);
        assert_eq!(config.chunk_size, 512);
    }

    #[test]
    fn prefill_batch_calculates_input_tokens() {
        let r1 = create_test_request(10);
        let r2 = create_test_request(20);
        let r3 = create_test_request(15);

        let batch = Batch::prefill(vec![r1, r2, r3]);

        assert_eq!(batch.batch_type, BatchType::Prefill);
        assert_eq!(batch.num_input_tokens, 45);
        assert_eq!(batch.num_output_tokens, 0);
        assert_eq!(batch.size(), 3);
    }

    #[test]
    fn decode_batch_calculates_output_tokens() {
        let r1 = create_test_request(10);
        let r2 = create_test_request(20);

        let batch = Batch::decode(vec![r1, r2]);

        assert_eq!(batch.batch_type, BatchType::Decode);
        assert_eq!(batch.num_input_tokens, 0);
        assert_eq!(batch.num_output_tokens, 2); // One token per request
        assert_eq!(batch.size(), 2);
    }

    #[test]
    fn mixed_batch_combines_both() {
        let prefill = vec![create_test_request(10), create_test_request(15)];
        let decode = vec![create_test_request(5), create_test_request(5), create_test_request(5)];

        let batch = Batch::mixed(prefill, decode);

        assert_eq!(batch.batch_type, BatchType::Mixed);
        assert_eq!(batch.num_input_tokens, 25); // 10 + 15
        assert_eq!(batch.num_output_tokens, 3); // 3 decode requests
        assert_eq!(batch.size(), 5); // All combined
    }

    #[test]
    fn batch_is_empty() {
        let batch = Batch::prefill(vec![]);
        assert!(batch.is_empty());
        assert_eq!(batch.size(), 0);
    }

    #[test]
    fn batch_total_tokens() {
        let prefill = vec![create_test_request(100)];
        let decode = vec![create_test_request(5)];

        let batch = Batch::mixed(prefill, decode);

        assert_eq!(batch.total_tokens(), 101); // 100 input + 1 output
    }

    #[test]
    fn empty_prefill_batch() {
        let batch = Batch::prefill(vec![]);

        assert_eq!(batch.num_input_tokens, 0);
        assert_eq!(batch.num_output_tokens, 0);
        assert_eq!(batch.total_tokens(), 0);
    }

    #[test]
    fn empty_decode_batch() {
        let batch = Batch::decode(vec![]);

        assert_eq!(batch.num_input_tokens, 0);
        assert_eq!(batch.num_output_tokens, 0);
        assert_eq!(batch.total_tokens(), 0);
    }

    #[test]
    fn batch_type_equality() {
        assert_eq!(BatchType::Prefill, BatchType::Prefill);
        assert_eq!(BatchType::Decode, BatchType::Decode);
        assert_eq!(BatchType::Mixed, BatchType::Mixed);
        assert_ne!(BatchType::Prefill, BatchType::Decode);
    }
}
