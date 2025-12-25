//! Benchmarks to prove O(1) fork latency.
//!
//! These benchmarks demonstrate that fork operations are constant time
//! regardless of:
//! - Tree depth
//! - Number of tokens in the sequence
//! - Number of existing forks
//!
//! This is achieved through copy-on-write semantics where forking only
//! copies block table pointers, not actual KV cache data.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use dendrite_core::cache::{KvCache, KvCacheConfig};
use dendrite_core::tree::TreeState;
use parking_lot::RwLock;
use std::sync::Arc;

/// Create a test KvCache with given configuration.
fn create_cache(max_blocks: usize) -> Arc<RwLock<KvCache>> {
    let config = KvCacheConfig {
        num_layers: 32,
        num_kv_heads: 8,
        head_dim: 128,
        max_blocks,
        tokens_per_block: 16,
    };
    Arc::new(RwLock::new(KvCache::new(config).unwrap()))
}

/// Create a TreeState and build a chain of forks to a given depth.
fn setup_tree_with_depth(depth: usize) -> (TreeState, Vec<dendrite_core::tree::NodeId>) {
    let cache = create_cache(65536);
    let state = TreeState::new(cache, 16);
    let mut node_ids = vec![dendrite_core::tree::NodeId::ROOT];

    let mut current = dendrite_core::tree::NodeId::ROOT;
    for _ in 0..depth {
        let handle = state.fork(current).unwrap();
        node_ids.push(handle.node_id);
        current = handle.node_id;
    }

    (state, node_ids)
}

/// Create a TreeState and add tokens to build up blocks.
fn setup_tree_with_tokens(num_tokens: usize) -> (TreeState, dendrite_core::tree::NodeId) {
    let cache = create_cache(65536);
    let state = TreeState::new(cache, 16);

    // Fork from root to get a working node
    let handle = state.fork(dendrite_core::tree::NodeId::ROOT).unwrap();
    let node_id = handle.node_id;

    // Add tokens
    for i in 0..num_tokens {
        state.append_token(node_id, i as u32).unwrap();
    }

    (state, node_id)
}

/// Benchmark: Fork latency should be O(1) regardless of tree depth.
///
/// This test forks from nodes at various depths and verifies that
/// the fork time is constant (not proportional to depth).
fn bench_fork_vs_depth(c: &mut Criterion) {
    let mut group = c.benchmark_group("fork_vs_depth");
    group.throughput(Throughput::Elements(1));

    for depth in [1, 10, 50, 100, 500].iter() {
        let (state, node_ids) = setup_tree_with_depth(*depth);
        let deepest = node_ids.last().copied().unwrap();

        group.bench_with_input(BenchmarkId::new("depth", depth), depth, |b, _| {
            b.iter(|| {
                let handle = state.fork(black_box(deepest)).unwrap();
                black_box(handle)
            })
        });
    }

    group.finish();
}

/// Benchmark: Fork latency should be O(1) regardless of token count.
///
/// More tokens means more blocks, but fork should still be constant
/// time because we only copy block table pointers.
fn bench_fork_vs_tokens(c: &mut Criterion) {
    let mut group = c.benchmark_group("fork_vs_tokens");
    group.throughput(Throughput::Elements(1));

    for num_tokens in [16, 256, 1024, 4096, 16384].iter() {
        let (state, node_id) = setup_tree_with_tokens(*num_tokens);
        let blocks = *num_tokens / 16;

        group.bench_with_input(
            BenchmarkId::new("tokens", format!("{}_({}_blocks)", num_tokens, blocks)),
            num_tokens,
            |b, _| {
                b.iter(|| {
                    let handle = state.fork(black_box(node_id)).unwrap();
                    black_box(handle)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark: Fork latency with multiple existing forks.
///
/// Having many forks from the same parent shouldn't slow down new forks.
fn bench_fork_with_siblings(c: &mut Criterion) {
    let mut group = c.benchmark_group("fork_with_siblings");
    group.throughput(Throughput::Elements(1));

    for num_siblings in [0, 10, 100, 500].iter() {
        let cache = create_cache(65536);
        let state = TreeState::new(cache, 16);

        // Create siblings
        for _ in 0..*num_siblings {
            state.fork(dendrite_core::tree::NodeId::ROOT).unwrap();
        }

        group.bench_with_input(
            BenchmarkId::new("siblings", num_siblings),
            num_siblings,
            |b, _| {
                b.iter(|| {
                    let handle = state
                        .fork(black_box(dendrite_core::tree::NodeId::ROOT))
                        .unwrap();
                    black_box(handle)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark: Bulk fork throughput.
///
/// Measures how many forks per second we can achieve.
fn bench_fork_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("fork_throughput");

    let cache = create_cache(65536);
    let state = TreeState::new(cache, 16);

    // Create a node with some tokens
    let handle = state.fork(dendrite_core::tree::NodeId::ROOT).unwrap();
    for i in 0..256 {
        state.append_token(handle.node_id, i).unwrap();
    }

    group.throughput(Throughput::Elements(100));
    group.bench_function("100_forks", |b| {
        b.iter(|| {
            for _ in 0..100 {
                let h = state.fork(black_box(handle.node_id)).unwrap();
                black_box(h);
            }
        })
    });

    group.finish();
}

/// Benchmark: Compare fork vs deep copy (if we had one).
///
/// This demonstrates the O(1) vs O(n) difference conceptually.
fn bench_fork_vs_block_count(c: &mut Criterion) {
    let mut group = c.benchmark_group("fork_scaling");
    group.throughput(Throughput::Elements(1));

    // Vary block count significantly
    for num_blocks in [1, 16, 64, 256, 1024].iter() {
        let num_tokens = num_blocks * 16;
        let (state, node_id) = setup_tree_with_tokens(num_tokens);

        group.bench_with_input(
            BenchmarkId::new("blocks", num_blocks),
            num_blocks,
            |b, _| {
                b.iter(|| {
                    let handle = state.fork(black_box(node_id)).unwrap();
                    black_box(handle)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_fork_vs_depth,
    bench_fork_vs_tokens,
    bench_fork_with_siblings,
    bench_fork_throughput,
    bench_fork_vs_block_count,
);

criterion_main!(benches);
