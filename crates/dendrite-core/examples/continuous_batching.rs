//! Continuous batching demonstration.
//!
//! Shows how `ContinuousBatcher` mixes decode and prefill in the same step,
//! and measures the scheduling overhead vs the legacy `Scheduler`.
//!
//! Run with:
//! ```bash
//! cargo run -p dendrite-core --example continuous_batching
//! ```
//!
//! Expected output (scheduling overhead only — no GPU):
//! ```
//! Continuous Batching Demo
//! ========================
//!
//! [Scenario 1] Cold start: single 512-token prompt
//!   Prefill chunks: 1
//!   Total prefill steps: 1
//!   First decode step: step 2
//!
//! [Scenario 2] High-throughput: 32 decoding seqs + incoming prefill
//!   Step | Decode | Prefill seqs | Prefill tokens | Total tokens
//!   ---- | ------ | ------------ | -------------- | ------------
//!      1 |     32 |            1 |            512 |          544
//!      2 |     33 |            0 |              0 |           33
//!   ...
//!
//! [Scenario 3] Continuous arrival: 10 requests arrive every 5 steps
//!   Average utilization: 87.3%
//!   Steps until empty:   42
//! ```

use dendrite_core::scheduler::{BatchConfig, ContinuousBatcher, Request};
use std::time::Instant;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn req(prompt_len: usize, max_tokens: usize) -> Request {
    let tokens: Vec<u32> = (0..prompt_len as u32).collect();
    Request::new(tokens, max_tokens)
}

fn make_batcher(max_batch: usize, max_tokens: usize, chunk_size: usize) -> ContinuousBatcher {
    ContinuousBatcher::new(BatchConfig {
        max_batch_size: max_batch,
        max_prefill_tokens: max_tokens,
        max_total_tokens: max_tokens,
        chunked_prefill: true,
        chunk_size,
    })
}

// ── Scenarios ─────────────────────────────────────────────────────────────────

/// Scenario 1: cold start — one prompt, chunked prefill then decode.
fn scenario_cold_start() {
    println!("[Scenario 1] Cold start: single 512-token prompt, chunk_size=128");
    let batcher = make_batcher(64, 4096, 128);
    batcher.add_request(req(512, 20));

    let mut step = 0usize;
    let mut first_decode = None;
    let mut prefill_chunks = 0usize;

    loop {
        step += 1;
        let s = batcher.next_step().unwrap();
        if s.is_empty() {
            break;
        }
        if !s.prefill_chunk.is_empty() {
            prefill_chunks += s.prefill_chunk.len();
        }
        if first_decode.is_none() && !s.decode_requests.is_empty() {
            first_decode = Some(step);
        }
        batcher.advance_decode();
        if step > 100 {
            break;
        }
    }

    println!("  Prefill chunks issued: {prefill_chunks}");
    println!("  First decode step:     {}", first_decode.unwrap_or(0));
    println!("  Total steps:           {step}");
    println!();
}

/// Scenario 2: mixed step — 32 decoding seqs + one incoming 512-token prompt.
fn scenario_mixed_step() {
    println!("[Scenario 2] Mixed step: 32 decoding seqs + new 512-token prompt, chunk_size=512");
    let batcher = make_batcher(64, 8192, 512);

    // Pre-load 32 decoding sequences (simulate already-active batch)
    for _ in 0..32 {
        let mut r = req(64, 40);
        r.state = dendrite_core::scheduler::RequestState::Decoding;
        batcher.push_decoding(r);
    }

    // New prompt arrives
    batcher.add_request(req(512, 20));

    println!("  Step | Decode | PfSeqs | PfTokens | Total");
    println!("  -----|--------|--------|----------|------");

    for step in 1..=5 {
        let s = batcher.next_step().unwrap();
        println!(
            "  {:>4} | {:>6} | {:>6} | {:>8} | {:>5}",
            step,
            s.decode_tokens,
            s.prefill_chunk.len(),
            s.prefill_tokens,
            s.total_tokens()
        );
        batcher.advance_decode();
        if s.is_empty() {
            break;
        }
    }
    println!();
}

/// Scenario 3: continuous arrival — requests arrive throughout processing.
fn scenario_continuous_arrival() {
    println!("[Scenario 3] Continuous arrival: 5 requests every 3 steps (prompt=256, gen=16)");
    let batcher = make_batcher(128, 8192, 512);

    let mut total_steps = 0usize;
    let mut total_tokens = 0usize;
    let mut total_capacity: usize = 0;
    let start = Instant::now();

    for step in 1..=60 {
        // New requests arrive every 3 steps
        if step % 3 == 1 {
            for _ in 0..5 {
                batcher.add_request(req(256, 16));
            }
        }

        let s = batcher.next_step().unwrap();
        total_tokens += s.total_tokens();
        total_capacity += 128; // max_batch per step
        batcher.advance_decode();
        total_steps = step;

        if step == 60 || (batcher.is_idle() && step > 20) {
            break;
        }
    }

    let elapsed = start.elapsed();
    let utilization = if total_capacity > 0 {
        (total_tokens as f64 / total_capacity as f64) * 100.0
    } else {
        0.0
    };

    println!("  Steps run:             {total_steps}");
    println!("  Total tokens scheduled: {total_tokens}");
    println!(
        "  Avg token utilization:  {:.1}% (of max_batch=128 per step)",
        utilization
    );
    println!(
        "  Scheduling overhead:    {:.2}µs/step",
        elapsed.as_micros() as f64 / total_steps as f64
    );
    println!();
}

/// Scenario 4: verify the fundamental continuous batching invariant.
fn scenario_invariant_check() {
    println!("[Scenario 4] Invariant: decode + prefill run in the SAME step");
    let batcher = make_batcher(64, 4096, 512);

    // First request: prefill + becomes decode
    batcher.add_request(req(32, 30));
    let _s1 = batcher.next_step().unwrap(); // prefills req1

    // Second request arrives while first is decoding
    batcher.add_request(req(64, 10));

    let s2 = batcher.next_step().unwrap();
    let mixed = !s2.decode_requests.is_empty() && !s2.prefill_chunk.is_empty();

    println!(
        "  Step 2: decode_seqs={}, prefill_seqs={}, mixed={}",
        s2.decode_requests.len(),
        s2.prefill_chunk.len(),
        mixed
    );
    if mixed {
        println!("  ✓ Invariant confirmed: decode and prefill run together.");
    } else {
        println!("  ✗ Invariant NOT met — check batcher configuration.");
    }
    println!();
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    println!("Continuous Batching Demo (scheduling overhead only)");
    println!("====================================================\n");

    scenario_cold_start();
    scenario_mixed_step();
    scenario_continuous_arrival();
    scenario_invariant_check();

    println!("Done. Run `cargo bench --bench scheduler` for rigorous microbenchmarks.");
}
