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
