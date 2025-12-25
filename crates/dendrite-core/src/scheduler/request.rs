//! Request types for the scheduler.

use crate::tree::NodeId;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Unique identifier for a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RequestId(pub u64);

static NEXT_REQUEST_ID: AtomicU64 = AtomicU64::new(1);

impl RequestId {
    /// Generate a new unique request ID.
    pub fn new() -> Self {
        RequestId(NEXT_REQUEST_ID.fetch_add(1, Ordering::Relaxed))
    }
}

impl Default for RequestId {
    fn default() -> Self {
        Self::new()
    }
}

/// State of a request in the scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestState {
    /// Waiting in queue.
    Waiting,
    /// Currently in prefill phase.
    Prefilling,
    /// Currently in decode phase.
    Decoding,
    /// Temporarily preempted.
    Preempted,
    /// Completed generation.
    Completed,
    /// Failed with error.
    Failed,
}

/// An inference request.
#[derive(Debug, Clone)]
pub struct Request {
    /// Unique request ID.
    pub id: RequestId,
    /// Input token IDs.
    pub input_tokens: Vec<u32>,
    /// Output tokens generated so far.
    pub output_tokens: Vec<u32>,
    /// Maximum tokens to generate.
    pub max_tokens: usize,
    /// Current state.
    pub state: RequestState,
    /// Tree node for this request.
    pub node_id: Option<NodeId>,
    /// Priority (higher = more urgent).
    pub priority: i32,
    /// Arrival time.
    pub arrival_time: Instant,
    /// Sampling parameters.
    pub sampling: SamplingParams,
}

impl Request {
    /// Create a new request.
    pub fn new(input_tokens: Vec<u32>, max_tokens: usize) -> Self {
        Self {
            id: RequestId::new(),
            input_tokens,
            output_tokens: Vec::new(),
            max_tokens,
            state: RequestState::Waiting,
            node_id: None,
            priority: 0,
            arrival_time: Instant::now(),
            sampling: SamplingParams::default(),
        }
    }

    /// Create a request with priority.
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Create a request with sampling parameters.
    pub fn with_sampling(mut self, sampling: SamplingParams) -> Self {
        self.sampling = sampling;
        self
    }

    /// Set the tree node for this request.
    pub fn with_node(mut self, node_id: NodeId) -> Self {
        self.node_id = Some(node_id);
        self
    }

    /// Get total sequence length.
    pub fn seq_len(&self) -> usize {
        self.input_tokens.len() + self.output_tokens.len()
    }

    /// Get remaining tokens to generate.
    pub fn remaining_tokens(&self) -> usize {
        self.max_tokens.saturating_sub(self.output_tokens.len())
    }

    /// Check if generation is complete.
    pub fn is_finished(&self) -> bool {
        self.output_tokens.len() >= self.max_tokens
            || self.state == RequestState::Completed
            || self.state == RequestState::Failed
    }

    /// Add an output token.
    pub fn add_output_token(&mut self, token: u32) {
        self.output_tokens.push(token);
    }

    /// Time waiting in queue.
    pub fn wait_time(&self) -> std::time::Duration {
        self.arrival_time.elapsed()
    }
}

/// Sampling parameters for generation.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// Temperature for sampling.
    pub temperature: f32,
    /// Top-p (nucleus) sampling.
    pub top_p: f32,
    /// Top-k sampling.
    pub top_k: usize,
    /// Repetition penalty.
    pub repetition_penalty: f32,
    /// Stop sequences.
    pub stop_sequences: Vec<Vec<u32>>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            stop_sequences: Vec::new(),
        }
    }
}

impl SamplingParams {
    /// Greedy sampling (temperature=0).
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            ..Default::default()
        }
    }

    /// Sampling with temperature.
    pub fn with_temperature(temperature: f32) -> Self {
        Self {
            temperature,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_id_generates_unique_ids() {
        let id1 = RequestId::new();
        let id2 = RequestId::new();
        let id3 = RequestId::new();

        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
        assert_ne!(id1, id3);
    }

    #[test]
    fn new_request_has_waiting_state() {
        let request = Request::new(vec![1, 2, 3], 10);

        assert_eq!(request.state, RequestState::Waiting);
        assert_eq!(request.input_tokens, vec![1, 2, 3]);
        assert_eq!(request.max_tokens, 10);
        assert!(request.output_tokens.is_empty());
    }

    #[test]
    fn request_with_priority() {
        let request = Request::new(vec![1], 10).with_priority(5);
        assert_eq!(request.priority, 5);
    }

    #[test]
    fn request_with_sampling() {
        let sampling = SamplingParams::greedy();
        let request = Request::new(vec![1], 10).with_sampling(sampling);

        assert_eq!(request.sampling.temperature, 0.0);
    }

    #[test]
    fn request_with_node() {
        let request = Request::new(vec![1], 10).with_node(NodeId(42));
        assert_eq!(request.node_id, Some(NodeId(42)));
    }

    #[test]
    fn seq_len_includes_input_and_output() {
        let mut request = Request::new(vec![1, 2, 3, 4, 5], 10);

        assert_eq!(request.seq_len(), 5);

        request.add_output_token(100);
        request.add_output_token(101);

        assert_eq!(request.seq_len(), 7);
    }

    #[test]
    fn remaining_tokens_calculates_correctly() {
        let mut request = Request::new(vec![1], 10);

        assert_eq!(request.remaining_tokens(), 10);

        request.add_output_token(100);
        request.add_output_token(101);
        request.add_output_token(102);

        assert_eq!(request.remaining_tokens(), 7);
    }

    #[test]
    fn is_finished_when_max_tokens_reached() {
        let mut request = Request::new(vec![1], 3);

        assert!(!request.is_finished());

        request.add_output_token(100);
        request.add_output_token(101);
        assert!(!request.is_finished());

        request.add_output_token(102);
        assert!(request.is_finished());
    }

    #[test]
    fn is_finished_when_completed() {
        let mut request = Request::new(vec![1], 100);
        request.state = RequestState::Completed;

        assert!(request.is_finished());
    }

    #[test]
    fn is_finished_when_failed() {
        let mut request = Request::new(vec![1], 100);
        request.state = RequestState::Failed;

        assert!(request.is_finished());
    }

    #[test]
    fn add_output_token_appends() {
        let mut request = Request::new(vec![1], 10);

        request.add_output_token(100);
        request.add_output_token(101);
        request.add_output_token(102);

        assert_eq!(request.output_tokens, vec![100, 101, 102]);
    }

    #[test]
    fn sampling_params_defaults() {
        let params = SamplingParams::default();

        assert_eq!(params.temperature, 1.0);
        assert_eq!(params.top_p, 1.0);
        assert_eq!(params.top_k, 0);
        assert_eq!(params.repetition_penalty, 1.0);
        assert!(params.stop_sequences.is_empty());
    }

    #[test]
    fn sampling_params_greedy() {
        let params = SamplingParams::greedy();
        assert_eq!(params.temperature, 0.0);
    }

    #[test]
    fn sampling_params_with_temperature() {
        let params = SamplingParams::with_temperature(0.7);
        assert_eq!(params.temperature, 0.7);
    }
}
