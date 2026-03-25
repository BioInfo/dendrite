//! Continuous batching scheduler.
//!
//! True continuous batching: every step mixes ALL currently-decoding requests
//! with as much chunked-prefill as fits in the token budget.
//!
//! # Algorithm (per step)
//!
//! 1. **Collect decode requests** — all requests in `Decoding` state, up to
//!    `max_batch_size`. Each consumes 1 output token of budget.
//! 2. **Fill remaining budget with chunked prefill** — take waiting requests
//!    in FIFO order. For each, take up to `chunk_size` tokens while the
//!    remaining token budget allows. A request that spans multiple steps uses
//!    `tokens_processed` to track progress.
//! 3. **Return a `MixedStep`** — the caller executes one forward pass using
//!    the mixed tensor of decode + prefill sequences.
//!
//! # Why This Matters
//!
//! The original `Scheduler::schedule()` returned *either* a decode batch *or*
//! a prefill batch. That means every step either generates tokens *or* processes
//! prompt — never both. On a real LLM server, that wastes GPU throughput
//! whenever the active decode batch is small (e.g., cold start, short prompts).
//!
//! With continuous batching, decode sequences run every step while new prompts
//! are chunked into the spare token budget. This keeps GPU utilization high
//! and first-token latency low for new arrivals.
//!
//! # References
//!
//! - Orca (OSDI 2022) — original continuous batching paper
//! - vLLM scheduler — production reference implementation
//! - SGLang RadixAttention — prefix caching extension

use crate::error::Result;
use crate::scheduler::batch::BatchConfig;
use crate::scheduler::request::{Request, RequestId, RequestState};
use parking_lot::Mutex;
use std::collections::VecDeque;

/// A single continuous-batching step result.
///
/// Contains both the decoding requests and any prefill chunk being processed
/// this step. The executor runs one forward pass for all of them together.
#[derive(Debug, Clone)]
pub struct MixedStep {
    /// Requests in active decode (generating next token each step).
    pub decode_requests: Vec<Request>,
    /// Requests in active prefill chunk (processing prompt tokens this step).
    pub prefill_chunk: Vec<PrefillChunk>,
    /// Total input tokens this step (prefill portion).
    pub prefill_tokens: usize,
    /// Total output tokens this step (decode portion = decode_requests.len()).
    pub decode_tokens: usize,
}

impl MixedStep {
    /// Total token count for this step.
    pub fn total_tokens(&self) -> usize {
        self.prefill_tokens + self.decode_tokens
    }

    /// True if this step has nothing to do.
    pub fn is_empty(&self) -> bool {
        self.decode_requests.is_empty() && self.prefill_chunk.is_empty()
    }

    /// Number of sequences in this step.
    pub fn num_sequences(&self) -> usize {
        self.decode_requests.len() + self.prefill_chunk.len()
    }
}

/// A chunk of a prefill request being processed this step.
#[derive(Debug, Clone)]
pub struct PrefillChunk {
    /// The underlying request.
    pub request: Request,
    /// Token offset into the full prompt (start of this chunk).
    pub chunk_start: usize,
    /// Number of tokens in this chunk.
    pub chunk_len: usize,
    /// Whether this chunk completes the full prompt.
    pub is_last_chunk: bool,
}

impl PrefillChunk {
    /// Get the token slice for this chunk.
    pub fn tokens(&self) -> &[u32] {
        &self.request.input_tokens[self.chunk_start..self.chunk_start + self.chunk_len]
    }
}

/// Continuous batching scheduler state.
///
/// Designed to replace `Scheduler` as the production-grade implementation.
/// Uses interior mutability for thread-safety.
#[derive(Debug)]
pub struct ContinuousBatcher {
    /// Requests waiting for their first prefill chunk.
    waiting: Mutex<VecDeque<Request>>,
    /// Requests mid-prefill (started but not complete).
    prefilling: Mutex<Vec<PrefillProgress>>,
    /// Requests actively decoding (prompt fully consumed).
    decoding: Mutex<Vec<Request>>,
    /// Configuration.
    config: BatchConfig,
}

/// Progress tracker for a partially-prefilled request.
#[derive(Debug, Clone)]
struct PrefillProgress {
    request: Request,
    /// How many prompt tokens have been processed so far.
    tokens_processed: usize,
}

impl ContinuousBatcher {
    /// Create a new continuous batcher.
    pub fn new(config: BatchConfig) -> Self {
        Self {
            waiting: Mutex::new(VecDeque::new()),
            prefilling: Mutex::new(Vec::new()),
            decoding: Mutex::new(Vec::new()),
            config,
        }
    }

    /// Add a new request to the waiting queue.
    pub fn add_request(&self, request: Request) -> RequestId {
        let id = request.id;
        self.waiting.lock().push_back(request);
        id
    }

    /// Compute the next step's mixed batch.
    ///
    /// This is the core continuous batching logic:
    /// 1. All decoding requests go in every step.
    /// 2. Remaining token budget fills with prefill chunks (chunked prefill).
    pub fn next_step(&self) -> Result<MixedStep> {
        let mut waiting = self.waiting.lock();
        let mut prefilling = self.prefilling.lock();
        let mut decoding = self.decoding.lock();

        // ── Step 1: collect all decoding requests ────────────────────────────
        let decode_requests: Vec<Request> = decoding
            .iter()
            .take(self.config.max_batch_size)
            .cloned()
            .collect();
        let decode_tokens = decode_requests.len();

        // ── Step 2: compute remaining token budget for prefill ────────────────
        let prefill_budget = self
            .config
            .max_total_tokens
            .saturating_sub(decode_tokens);
        let chunk_size = if self.config.chunked_prefill {
            self.config.chunk_size.min(prefill_budget)
        } else {
            prefill_budget
        };

        // ── Step 3: fill prefill chunks ───────────────────────────────────────
        let mut prefill_chunk = Vec::new();
        let mut remaining_budget = chunk_size;

        // First drain any in-progress prefills
        for prog in prefilling.iter_mut() {
            if remaining_budget == 0 {
                break;
            }
            let remaining_tokens = prog.request.input_tokens.len() - prog.tokens_processed;
            let take = remaining_tokens.min(remaining_budget);
            let is_last = prog.tokens_processed + take >= prog.request.input_tokens.len();

            prefill_chunk.push(PrefillChunk {
                request: prog.request.clone(),
                chunk_start: prog.tokens_processed,
                chunk_len: take,
                is_last_chunk: is_last,
            });

            prog.tokens_processed += take;
            remaining_budget = remaining_budget.saturating_sub(take);
        }

        // Move completed prefills to decoding (for next step)
        let completed: Vec<_> = prefilling
            .iter()
            .filter(|p| p.tokens_processed >= p.request.input_tokens.len())
            .map(|p| {
                let mut req = p.request.clone();
                req.state = RequestState::Decoding;
                req
            })
            .collect();
        prefilling.retain(|p| p.tokens_processed < p.request.input_tokens.len());
        for req in completed {
            if !decoding.iter().any(|d| d.id == req.id) {
                decoding.push(req);
            }
        }

        // Then pull from waiting queue
        while remaining_budget > 0 {
            if let Some(request) = waiting.pop_front() {
                let prompt_len = request.input_tokens.len();
                let take = prompt_len.min(remaining_budget);
                let is_last = take >= prompt_len;

                prefill_chunk.push(PrefillChunk {
                    request: request.clone(),
                    chunk_start: 0,
                    chunk_len: take,
                    is_last_chunk: is_last,
                });

                if !is_last {
                    // Partially consumed: put in prefilling
                    prefilling.push(PrefillProgress {
                        request,
                        tokens_processed: take,
                    });
                } else {
                    // Fully consumed in one chunk: goes straight to decoding next step
                    let mut req = request;
                    req.state = RequestState::Decoding;
                    decoding.push(req);
                }

                remaining_budget = remaining_budget.saturating_sub(take);
            } else {
                break;
            }
        }

        let prefill_tokens: usize = prefill_chunk.iter().map(|c| c.chunk_len).sum();

        Ok(MixedStep {
            decode_requests,
            prefill_chunk,
            prefill_tokens,
            decode_tokens,
        })
    }

    /// Notify the batcher that a decode step completed, advancing each
    /// decoding request by one token. Requests that hit `max_tokens`
    /// are removed.
    pub fn advance_decode(&self) {
        let mut decoding = self.decoding.lock();
        for req in decoding.iter_mut() {
            req.add_output_token(0); // placeholder token — real engine fills this in
        }
        decoding.retain(|r| !r.is_finished());
    }

    /// Mark a prefill-chunk as complete (called after the forward pass).
    ///
    /// Requests whose chunk was the last chunk move to decoding state.
    pub fn commit_prefill_chunks(&self, chunks: &[PrefillChunk]) {
        let mut decoding = self.decoding.lock();
        for chunk in chunks {
            if chunk.is_last_chunk {
                let mut req = chunk.request.clone();
                req.state = RequestState::Decoding;
                // Avoid duplicates (already added in next_step for single-chunk case)
                if !decoding.iter().any(|d| d.id == req.id) {
                    decoding.push(req);
                }
            }
        }
    }

    /// Complete a request and remove it from decoding.
    pub fn complete_request(&self, request_id: RequestId) {
        self.decoding.lock().retain(|r| r.id != request_id);
    }

    /// Number of requests waiting for their first chunk.
    pub fn num_waiting(&self) -> usize {
        self.waiting.lock().len()
    }

    /// Number of requests mid-prefill.
    pub fn num_prefilling(&self) -> usize {
        self.prefilling.lock().len()
    }

    /// Number of requests actively decoding.
    pub fn num_decoding(&self) -> usize {
        self.decoding.lock().len()
    }

    /// True when no requests are in flight.
    pub fn is_idle(&self) -> bool {
        self.num_waiting() == 0 && self.num_prefilling() == 0 && self.num_decoding() == 0
    }

    /// Throughput stats for the last step (tokens/step).
    pub fn step_stats(step: &MixedStep) -> StepStats {
        StepStats {
            decode_seqs: step.decode_requests.len(),
            prefill_seqs: step.prefill_chunk.len(),
            prefill_tokens: step.prefill_tokens,
            decode_tokens: step.decode_tokens,
            total_tokens: step.total_tokens(),
        }
    }
}

/// Per-step throughput statistics.
#[derive(Debug, Clone, Copy, Default)]
pub struct StepStats {
    pub decode_seqs: usize,
    pub prefill_seqs: usize,
    pub prefill_tokens: usize,
    pub decode_tokens: usize,
    pub total_tokens: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::request::Request;

    fn make_batcher(max_batch: usize, max_tokens: usize, chunk_size: usize) -> ContinuousBatcher {
        ContinuousBatcher::new(BatchConfig {
            max_batch_size: max_batch,
            max_prefill_tokens: max_tokens,
            max_total_tokens: max_tokens,
            chunked_prefill: true,
            chunk_size,
        })
    }

    fn req(len: usize, max_tokens: usize) -> Request {
        let tokens: Vec<u32> = (0..len as u32).collect();
        Request::new(tokens, max_tokens)
    }

    // ── Basic lifecycle ───────────────────────────────────────────────────────

    #[test]
    fn idle_when_empty() {
        let batcher = make_batcher(8, 512, 128);
        assert!(batcher.is_idle());
        let step = batcher.next_step().unwrap();
        assert!(step.is_empty());
    }

    #[test]
    fn single_short_request_prefill_then_decode() {
        let batcher = make_batcher(8, 512, 512);

        batcher.add_request(req(10, 5));

        // Step 1: prompt fits in one chunk → prefill
        let step = batcher.next_step().unwrap();
        assert_eq!(step.prefill_chunk.len(), 1);
        assert_eq!(step.prefill_chunk[0].chunk_len, 10);
        assert!(step.prefill_chunk[0].is_last_chunk);
        assert_eq!(step.decode_requests.len(), 0);

        // Step 2: request should now be decoding
        let step2 = batcher.next_step().unwrap();
        assert_eq!(step2.decode_requests.len(), 1);
        assert_eq!(step2.prefill_chunk.len(), 0);
    }

    #[test]
    fn chunked_prefill_splits_long_prompt() {
        // chunk_size=8, prompt=20 tokens → 3 chunks (8, 8, 4)
        let batcher = make_batcher(8, 32, 8);

        batcher.add_request(req(20, 5));

        // Step 1: first 8 tokens
        let s1 = batcher.next_step().unwrap();
        assert_eq!(s1.prefill_tokens, 8);
        assert!(!s1.prefill_chunk[0].is_last_chunk);

        // Step 2: next 8 tokens
        let s2 = batcher.next_step().unwrap();
        assert_eq!(s2.prefill_tokens, 8);
        assert!(!s2.prefill_chunk[0].is_last_chunk);

        // Step 3: final 4 tokens
        let s3 = batcher.next_step().unwrap();
        assert_eq!(s3.prefill_tokens, 4);
        assert!(s3.prefill_chunk[0].is_last_chunk);

        // Step 4: should be decoding
        let s4 = batcher.next_step().unwrap();
        assert_eq!(s4.decode_requests.len(), 1);
        assert_eq!(s4.prefill_chunk.len(), 0);
    }

    // ── Continuous batching (the key property) ────────────────────────────────

    #[test]
    fn decode_and_prefill_mixed_in_same_step() {
        // This is the fundamental continuous batching invariant:
        // a decoding request and a new prefill run in the SAME step.
        let batcher = make_batcher(8, 512, 512);

        // Add first request, prefill it
        batcher.add_request(req(5, 10));
        let _s1 = batcher.next_step().unwrap(); // prefills request 1

        // Add second request while first is now decoding
        batcher.add_request(req(8, 5));

        // Next step: decode req1 AND prefill req2 simultaneously
        let s2 = batcher.next_step().unwrap();
        assert_eq!(s2.decode_requests.len(), 1, "req1 should be decoding");
        assert_eq!(s2.prefill_chunk.len(), 1, "req2 should be prefilling");
        assert!(s2.total_tokens() > 1, "should have tokens from both");
    }

    #[test]
    fn token_budget_limits_prefill() {
        // max_total_tokens=16, decode takes 4, leaving 12 for prefill
        let batcher = make_batcher(8, 16, 16);

        // Put 4 requests in decoding state by hand
        for _ in 0..4 {
            let mut r = req(5, 10);
            r.state = RequestState::Decoding;
            batcher.decoding.lock().push(r);
        }

        // Add a 20-token prompt — should only get 12 tokens this step
        batcher.add_request(req(20, 5));
        let step = batcher.next_step().unwrap();

        assert_eq!(step.decode_tokens, 4);
        // prefill_tokens ≤ 12 (budget after decode)
        assert!(
            step.prefill_tokens <= 12,
            "prefill_tokens={} exceeded budget",
            step.prefill_tokens
        );
    }

    #[test]
    fn batch_size_limits_decode() {
        // max_batch_size=3; 5 decoding requests → only 3 scheduled
        let batcher = make_batcher(3, 256, 128);

        for _ in 0..5 {
            let mut r = req(5, 10);
            r.state = RequestState::Decoding;
            batcher.decoding.lock().push(r);
        }

        let step = batcher.next_step().unwrap();
        assert_eq!(step.decode_requests.len(), 3);
    }

    // ── advance_decode ────────────────────────────────────────────────────────

    #[test]
    fn advance_decode_removes_finished_requests() {
        let batcher = make_batcher(8, 256, 128);

        // Put a request in decoding with max_new_tokens=1
        let mut r = req(5, 1);
        r.state = RequestState::Decoding;
        batcher.decoding.lock().push(r);

        assert_eq!(batcher.num_decoding(), 1);

        batcher.advance_decode();

        // After generating 1 token (= max_new_tokens), request should be gone
        assert_eq!(batcher.num_decoding(), 0);
    }

    #[test]
    fn advance_decode_keeps_requests_with_budget() {
        let batcher = make_batcher(8, 256, 128);

        let mut r = req(5, 5);
        r.state = RequestState::Decoding;
        batcher.decoding.lock().push(r);

        batcher.advance_decode(); // tokens_generated = 1
        assert_eq!(batcher.num_decoding(), 1); // still running (needs 4 more)

        for _ in 0..4 {
            batcher.advance_decode();
        }
        assert_eq!(batcher.num_decoding(), 0); // done after 5 total
    }

    // ── Queue counters ─────────────────────────────────────────────────────────

    #[test]
    fn counters_track_lifecycle() {
        let batcher = make_batcher(8, 512, 512);

        assert_eq!(batcher.num_waiting(), 0);
        assert_eq!(batcher.num_decoding(), 0);

        batcher.add_request(req(4, 3));
        assert_eq!(batcher.num_waiting(), 1);

        // Prefill
        let _s = batcher.next_step().unwrap();
        assert_eq!(batcher.num_waiting(), 0);
        assert_eq!(batcher.num_decoding(), 1);

        // Decode 3 times
        for _ in 0..3 {
            batcher.advance_decode();
        }
        assert_eq!(batcher.num_decoding(), 0);
        assert!(batcher.is_idle());
    }

    // ── Multiple requests overlap ─────────────────────────────────────────────

    #[test]
    fn two_requests_overlap_at_all_times() {
        // Classic continuous batching scenario:
        // - req1 arrives, gets prefilled
        // - req2 arrives during req1 decode
        // - Both overlap for some steps
        let batcher = make_batcher(8, 512, 512);

        // req1: 6 token prompt, generate 10 tokens
        batcher.add_request(req(6, 10));

        // Step 1: prefill req1 (no decoding yet)
        let s1 = batcher.next_step().unwrap();
        assert_eq!(s1.prefill_chunk.len(), 1, "req1 should be prefilling");
        assert_eq!(s1.decode_requests.len(), 0, "nothing decoding yet");

        // Add req2 while req1 is about to start decoding
        batcher.add_request(req(8, 10));

        // Step 2: req1 decodes + req2 prefills simultaneously
        let s2 = batcher.next_step().unwrap();
        assert_eq!(s2.decode_requests.len(), 1, "req1 should decode");
        assert_eq!(s2.prefill_chunk.len(), 1, "req2 should prefill");
        assert!(s2.total_tokens() > 1, "should have tokens from both");

        // Step 3: both should be decoding now
        let s3 = batcher.next_step().unwrap();
        assert_eq!(s3.decode_requests.len(), 2, "both should be decoding");
        assert_eq!(s3.prefill_chunk.len(), 0, "no more prefill");
    }

    // ── StepStats ─────────────────────────────────────────────────────────────

    #[test]
    fn step_stats_correct() {
        let batcher = make_batcher(8, 512, 512);
        batcher.add_request(req(10, 5));

        let step = batcher.next_step().unwrap();
        let stats = ContinuousBatcher::step_stats(&step);

        assert_eq!(stats.prefill_tokens, 10);
        assert_eq!(stats.decode_tokens, 0);
        assert_eq!(stats.total_tokens, 10);
        assert_eq!(stats.prefill_seqs, 1);
        assert_eq!(stats.decode_seqs, 0);
    }
}
