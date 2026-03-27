//! Scheduler benchmarks: decode throughput, prefill latency, mixed step.
//!
//! These benchmarks validate the performance characteristics of the
//! ContinuousBatcher relative to the original Scheduler, and profile
//! the throughput of the Orca-style mixed step.
//!
//! # Metrics
//!
//! - **decode_throughput**: step scheduling time for N decoding sequences
//! - **prefill_latency**: scheduling steps from first chunk to decode start
//! - **mixed_step_overhead**: overhead of decode+prefill scheduling logic
//! - **add_request**: request enqueue cost under varying queue depths

use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput,
};
use dendrite_core::cache::{KvCache, KvCacheConfig};
use dendrite_core::scheduler::{BatchConfig, ContinuousBatcher, Request, RequestState, Scheduler};
use dendrite_core::tree::TreeState;
use parking_lot::RwLock;
use std::sync::Arc;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn make_cache() -> Arc<RwLock<KvCache>> {
    let config = KvCacheConfig {
        num_layers: 1,
        num_kv_heads: 1,
        head_dim: 64,
        max_blocks: 4096,
        tokens_per_block: 16,
    };
    Arc::new(RwLock::new(KvCache::new(config).unwrap()))
}

fn make_scheduler(max_batch: usize) -> Scheduler {
    let kv = make_cache();
    let tree = Arc::new(TreeState::new(kv, 16));
    let cfg = BatchConfig {
        max_batch_size: max_batch,
        max_prefill_tokens: 4096,
        max_total_tokens: 8192,
        chunked_prefill: true,
        chunk_size: 512,
    };
    Scheduler::new(cfg, tree)
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

fn req(prompt_len: usize, max_tokens: usize) -> Request {
    let tokens: Vec<u32> = (0..prompt_len as u32).collect();
    Request::new(tokens, max_tokens)
}

fn decoding_req(prompt_len: usize, max_tokens: usize) -> Request {
    let mut r = req(prompt_len, max_tokens);
    r.state = RequestState::Decoding;
    r
}

// ── Benchmarks ────────────────────────────────────────────────────────────────

/// How fast can we schedule a decode step for N simultaneously-decoding sequences?
/// This is the hot path on a loaded server — runs every forward pass step.
fn bench_decode_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode_throughput");

    for batch_size in [1usize, 4, 8, 16, 32, 64, 128] {
        group.throughput(Throughput::Elements(batch_size as u64));

        // ContinuousBatcher
        group.bench_with_input(
            BenchmarkId::new("continuous_batcher", batch_size),
            &batch_size,
            |b, &n| {
                b.iter_batched(
                    || {
                        let batcher = make_batcher(n * 2, 8192, 512);
                        for _ in 0..n {
                            batcher.push_decoding(decoding_req(32, 128));
                        }
                        batcher
                    },
                    |batcher| {
                        let step = batcher.next_step().unwrap();
                        black_box(step.decode_tokens)
                    },
                    BatchSize::SmallInput,
                )
            },
        );

        // Legacy Scheduler — decode path
        group.bench_with_input(
            BenchmarkId::new("legacy_scheduler", batch_size),
            &batch_size,
            |b, &n| {
                b.iter_batched(
                    || {
                        let sched = make_scheduler(n * 2);
                        for _ in 0..n {
                            sched.push_running(decoding_req(32, 128));
                        }
                        sched
                    },
                    |sched| {
                        let batch = sched.schedule().unwrap();
                        black_box(batch.map(|b| b.num_output_tokens).unwrap_or(0))
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

/// Time from request arrival to first decode step (prefill latency).
/// Measures scheduling overhead only — not the actual forward pass.
fn bench_prefill_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefill_latency");

    for prompt_len in [64usize, 256, 512, 1024, 2048, 4096] {
        group.throughput(Throughput::Elements(prompt_len as u64));

        group.bench_with_input(
            BenchmarkId::new("chunks_to_decode", prompt_len),
            &prompt_len,
            |b, &len| {
                b.iter_batched(
                    || make_batcher(64, 8192, 512),
                    |batcher| {
                        batcher.add_request(req(len, 1));
                        // Drive scheduling steps until the request enters decode
                        let mut steps = 0usize;
                        loop {
                            let step = batcher.next_step().unwrap();
                            steps += 1;
                            if !step.decode_requests.is_empty() || steps > 200 {
                                break;
                            }
                        }
                        black_box(steps)
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

/// Overhead of the mixed-step scheduling call itself.
/// Should be O(batch_size) — this catches accidental O(n²) regressions.
fn bench_mixed_step_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_step_overhead");

    for decode_n in [1usize, 8, 32, 64, 128, 256] {
        group.throughput(Throughput::Elements((decode_n + 1) as u64));

        group.bench_with_input(
            BenchmarkId::new("decode_n_plus_1_prefill", decode_n),
            &decode_n,
            |b, &n| {
                b.iter_batched(
                    || {
                        let batcher = make_batcher(n * 2, 32768, 512);
                        for _ in 0..n {
                            batcher.push_decoding(decoding_req(64, 128));
                        }
                        batcher.add_request(req(256, 32));
                        batcher
                    },
                    |batcher| {
                        let step = batcher.next_step().unwrap();
                        black_box(step.total_tokens())
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

/// add_request cost — must be O(1) regardless of queue depth.
fn bench_add_request(c: &mut Criterion) {
    let mut group = c.benchmark_group("add_request");

    for queue_depth in [0usize, 100, 1000, 10_000] {
        group.bench_with_input(
            BenchmarkId::new("queue_depth", queue_depth),
            &queue_depth,
            |b, &depth| {
                b.iter_batched(
                    || {
                        let batcher = make_batcher(256, 32768, 512);
                        for _ in 0..depth {
                            batcher.add_request(req(32, 16));
                        }
                        batcher
                    },
                    |batcher| {
                        let id = batcher.add_request(black_box(req(32, 16)));
                        black_box(id)
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_decode_throughput,
    bench_prefill_latency,
    bench_mixed_step_overhead,
    bench_add_request,
);
criterion_main!(benches);
