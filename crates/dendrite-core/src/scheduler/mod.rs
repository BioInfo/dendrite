//! Request scheduling for batched inference.
//!
//! Implements continuous batching with:
//! - Prefill/decode phase separation
//! - Priority-based scheduling
//! - Preemption and swapping

mod batch;
mod policy;
mod request;

pub use batch::{Batch, BatchConfig};
pub use policy::SchedulingPolicy;
pub use request::{Request, RequestId, RequestState};

use crate::error::Result;
use crate::tree::TreeState;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::Arc;

/// Main scheduler for inference requests.
#[derive(Debug)]
pub struct Scheduler {
    /// Pending requests (waiting for prefill).
    waiting: Mutex<VecDeque<Request>>,
    /// Running requests (in decode phase).
    running: Mutex<Vec<Request>>,
    /// Configuration.
    config: BatchConfig,
    /// Tree state for KV cache management.
    #[allow(dead_code)]
    tree_state: Arc<TreeState>,
    /// Scheduling policy.
    #[allow(dead_code)]
    policy: SchedulingPolicy,
}

impl Scheduler {
    /// Create a new scheduler.
    pub fn new(config: BatchConfig, tree_state: Arc<TreeState>) -> Self {
        Self {
            waiting: Mutex::new(VecDeque::new()),
            running: Mutex::new(Vec::new()),
            config,
            tree_state,
            policy: SchedulingPolicy::default(),
        }
    }

    /// Add a new request to the scheduler.
    pub fn add_request(&self, request: Request) -> Result<RequestId> {
        let id = request.id;
        self.waiting.lock().push_back(request);
        Ok(id)
    }

    /// Schedule the next batch for execution.
    pub fn schedule(&self) -> Result<Option<Batch>> {
        let mut waiting = self.waiting.lock();
        let mut running = self.running.lock();

        // First, try to schedule decode for running requests
        if !running.is_empty() {
            let decode_requests: Vec<_> = running
                .iter()
                .filter(|r| r.state == RequestState::Decoding)
                .take(self.config.max_batch_size)
                .cloned()
                .collect();

            if !decode_requests.is_empty() {
                return Ok(Some(Batch::decode(decode_requests)));
            }
        }

        // Then, schedule prefill for waiting requests
        if let Some(request) = waiting.pop_front() {
            let mut request = request;
            request.state = RequestState::Prefilling;
            running.push(request.clone());
            return Ok(Some(Batch::prefill(vec![request])));
        }

        Ok(None)
    }

    /// Mark a request as completed.
    pub fn complete_request(&self, request_id: RequestId) -> Result<()> {
        let mut running = self.running.lock();
        running.retain(|r| r.id != request_id);
        Ok(())
    }

    /// Transition request from prefill to decode.
    pub fn prefill_complete(&self, request_id: RequestId) -> Result<()> {
        let mut running = self.running.lock();
        if let Some(request) = running.iter_mut().find(|r| r.id == request_id) {
            request.state = RequestState::Decoding;
        }
        Ok(())
    }

    /// Get number of waiting requests.
    pub fn num_waiting(&self) -> usize {
        self.waiting.lock().len()
    }

    /// Get number of running requests.
    pub fn num_running(&self) -> usize {
        self.running.lock().len()
    }

    /// Check if scheduler is idle.
    pub fn is_idle(&self) -> bool {
        self.waiting.lock().is_empty() && self.running.lock().is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::{KvCache, KvCacheConfig};
    use batch::BatchType;

    fn create_test_scheduler() -> Scheduler {
        let cache_config = KvCacheConfig {
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 64,
            max_blocks: 100,
            tokens_per_block: 16,
        };
        let kv_cache = Arc::new(parking_lot::RwLock::new(
            KvCache::new(cache_config).unwrap(),
        ));
        let tree_state = Arc::new(TreeState::new(kv_cache, 16));
        let batch_config = BatchConfig::default();
        Scheduler::new(batch_config, tree_state)
    }

    #[test]
    fn new_scheduler_is_idle() {
        let scheduler = create_test_scheduler();
        assert!(scheduler.is_idle());
        assert_eq!(scheduler.num_waiting(), 0);
        assert_eq!(scheduler.num_running(), 0);
    }

    #[test]
    fn add_request_increments_waiting() {
        let scheduler = create_test_scheduler();

        let request = Request::new(vec![1, 2, 3], 10);
        scheduler.add_request(request).unwrap();

        assert_eq!(scheduler.num_waiting(), 1);
        assert_eq!(scheduler.num_running(), 0);
        assert!(!scheduler.is_idle());
    }

    #[test]
    fn add_multiple_requests() {
        let scheduler = create_test_scheduler();

        for i in 0..5 {
            let request = Request::new(vec![i], 10);
            scheduler.add_request(request).unwrap();
        }

        assert_eq!(scheduler.num_waiting(), 5);
    }

    #[test]
    fn schedule_returns_none_when_idle() {
        let scheduler = create_test_scheduler();
        let batch = scheduler.schedule().unwrap();
        assert!(batch.is_none());
    }

    #[test]
    fn schedule_returns_prefill_batch() {
        let scheduler = create_test_scheduler();

        let request = Request::new(vec![1, 2, 3, 4, 5], 10);
        scheduler.add_request(request).unwrap();

        let batch = scheduler.schedule().unwrap().unwrap();

        assert_eq!(batch.batch_type, BatchType::Prefill);
        assert_eq!(batch.size(), 1);
        assert_eq!(batch.num_input_tokens, 5);
    }

    #[test]
    fn schedule_moves_request_to_running() {
        let scheduler = create_test_scheduler();

        let request = Request::new(vec![1, 2, 3], 10);
        scheduler.add_request(request).unwrap();

        assert_eq!(scheduler.num_waiting(), 1);
        assert_eq!(scheduler.num_running(), 0);

        let _batch = scheduler.schedule().unwrap();

        assert_eq!(scheduler.num_waiting(), 0);
        assert_eq!(scheduler.num_running(), 1);
    }

    #[test]
    fn prefill_complete_transitions_to_decode() {
        let scheduler = create_test_scheduler();

        let request = Request::new(vec![1, 2, 3], 10);
        let request_id = request.id;
        scheduler.add_request(request).unwrap();

        // Schedule prefill
        let _batch = scheduler.schedule().unwrap();

        // Complete prefill
        scheduler.prefill_complete(request_id).unwrap();

        // Next schedule should return decode batch
        let batch = scheduler.schedule().unwrap().unwrap();
        assert_eq!(batch.batch_type, BatchType::Decode);
    }

    #[test]
    fn complete_request_removes_from_running() {
        let scheduler = create_test_scheduler();

        let request = Request::new(vec![1, 2, 3], 10);
        let request_id = request.id;
        scheduler.add_request(request).unwrap();

        // Schedule and move to running
        let _batch = scheduler.schedule().unwrap();
        assert_eq!(scheduler.num_running(), 1);

        // Complete request
        scheduler.complete_request(request_id).unwrap();
        assert_eq!(scheduler.num_running(), 0);
        assert!(scheduler.is_idle());
    }

    #[test]
    fn scheduler_processes_requests_fifo() {
        let scheduler = create_test_scheduler();

        // Add requests in order
        let r1 = Request::new(vec![1], 10);
        let r2 = Request::new(vec![2], 10);
        let r3 = Request::new(vec![3], 10);

        let id1 = r1.id;
        let id2 = r2.id;
        let id3 = r3.id;

        scheduler.add_request(r1).unwrap();
        scheduler.add_request(r2).unwrap();
        scheduler.add_request(r3).unwrap();

        // Schedule in FIFO order
        let batch1 = scheduler.schedule().unwrap().unwrap();
        assert_eq!(batch1.requests[0].id, id1);

        scheduler.complete_request(id1).unwrap();

        let batch2 = scheduler.schedule().unwrap().unwrap();
        assert_eq!(batch2.requests[0].id, id2);

        scheduler.complete_request(id2).unwrap();

        let batch3 = scheduler.schedule().unwrap().unwrap();
        assert_eq!(batch3.requests[0].id, id3);
    }

    #[test]
    fn decode_batch_has_one_token_per_request() {
        let scheduler = create_test_scheduler();

        let request = Request::new(vec![1, 2, 3, 4, 5], 10);
        let request_id = request.id;
        scheduler.add_request(request).unwrap();

        // Prefill
        let _prefill = scheduler.schedule().unwrap();
        scheduler.prefill_complete(request_id).unwrap();

        // Decode
        let decode = scheduler.schedule().unwrap().unwrap();
        assert_eq!(decode.num_output_tokens, 1);
    }

    #[test]
    fn multiple_decode_requests_batched() {
        // This test verifies that when multiple requests are in Decoding state,
        // they get batched together in a single decode batch.
        // Note: The scheduler prioritizes decode over prefill, so we need to
        // manually set up the state rather than using the schedule() method.

        let cache_config = KvCacheConfig {
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 64,
            max_blocks: 100,
            tokens_per_block: 16,
        };
        let kv_cache = Arc::new(parking_lot::RwLock::new(
            KvCache::new(cache_config).unwrap(),
        ));
        let tree_state = Arc::new(TreeState::new(kv_cache, 16));
        let batch_config = BatchConfig::default();
        let scheduler = Scheduler::new(batch_config, tree_state);

        // Create 3 requests and add them directly to running in Decoding state
        for i in 0..3 {
            let mut request = Request::new(vec![i], 10);
            request.state = RequestState::Decoding;
            scheduler.running.lock().push(request);
        }

        // Schedule should batch all 3 decode requests
        let batch = scheduler.schedule().unwrap().unwrap();
        assert_eq!(batch.batch_type, BatchType::Decode);
        assert_eq!(batch.size(), 3);
    }

    #[test]
    fn decode_prioritized_over_prefill() {
        // This test verifies that decode requests are scheduled before prefill
        let scheduler = create_test_scheduler();

        // Add a waiting request
        let waiting_request = Request::new(vec![1, 2, 3], 10);
        scheduler.add_request(waiting_request).unwrap();

        // Add a decoding request directly to running
        let mut decoding_request = Request::new(vec![4, 5, 6], 10);
        decoding_request.state = RequestState::Decoding;
        scheduler.running.lock().push(decoding_request);

        // Schedule should return decode batch, not prefill
        let batch = scheduler.schedule().unwrap().unwrap();
        assert_eq!(batch.batch_type, BatchType::Decode);

        // The waiting request should still be waiting
        assert_eq!(scheduler.num_waiting(), 1);
    }
}
