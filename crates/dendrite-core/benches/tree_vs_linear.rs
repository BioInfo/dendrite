//! Benchmarks comparing tree-structured vs linear inference.
//!
//! These benchmarks demonstrate the memory and performance benefits of
//! tree-structured KV cache with copy-on-write vs traditional linear inference
//! that doesn't share cache between branches.
//!
//! # Key Comparisons
//!
//! 1. **Memory Usage**: Tree structure shares KV cache blocks via CoW
//! 2. **Branch Creation**: O(1) fork vs O(n) full copy
//! 3. **Multi-Branch Scenarios**: Beam search, MCTS, speculative decoding
//!
//! # Results Interpretation
//!
//! - Lower is better for time benchmarks
//! - Block usage shows memory efficiency of tree structure
//! - Speedup = linear_time / tree_time

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use dendrite_core::cache::{KvCache, KvCacheConfig};
use dendrite_core::tree::{NodeId, TreeState};
use parking_lot::RwLock;
use std::sync::Arc;

/// Configuration for benchmark scenarios.
#[derive(Clone)]
struct BenchConfig {
    /// Number of initial tokens (prefix length).
    prefix_tokens: usize,
    /// Number of branches to create.
    num_branches: usize,
    /// Tokens to add per branch.
    tokens_per_branch: usize,
    /// Maximum blocks in cache.
    max_blocks: usize,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            prefix_tokens: 1024,
            num_branches: 4,
            tokens_per_branch: 128,
            max_blocks: 65536,
        }
    }
}

/// Create a test KvCache.
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

/// Simulate tree-structured branching (uses CoW sharing).
fn tree_branch_simulation(config: &BenchConfig) -> (usize, usize) {
    let cache = create_cache(config.max_blocks);
    let state = TreeState::new(cache.clone(), 16);

    // Create prefix sequence
    let prefix = state.fork(NodeId::ROOT).unwrap();
    for i in 0..config.prefix_tokens {
        state.append_token(prefix.node_id, i as u32).unwrap();
    }

    let blocks_after_prefix = cache.read().used_blocks();

    // Create branches (O(1) fork with CoW)
    let mut branches = Vec::new();
    for _ in 0..config.num_branches {
        let branch = state.fork(prefix.node_id).unwrap();
        branches.push(branch.node_id);
    }

    let blocks_after_fork = cache.read().used_blocks();

    // Add tokens to each branch
    for (b, &node_id) in branches.iter().enumerate() {
        for i in 0..config.tokens_per_branch {
            let token = (b * 1000 + i) as u32;
            state.append_token(node_id, token).unwrap();
        }
    }

    let blocks_final = cache.read().used_blocks();

    // Return: blocks after fork (showing sharing), final blocks
    (blocks_after_fork - blocks_after_prefix, blocks_final)
}

/// Simulate linear branching (no sharing - each branch copies prefix).
fn linear_branch_simulation(config: &BenchConfig) -> (usize, usize) {
    let cache = create_cache(config.max_blocks);

    // Each branch is independent with its own copy of prefix
    for b in 0..config.num_branches {
        // Each branch starts fresh and adds prefix + branch tokens
        let state = TreeState::new(cache.clone(), 16);
        let branch = state.fork(NodeId::ROOT).unwrap();

        // Add prefix tokens (simulating full copy)
        for i in 0..config.prefix_tokens {
            state.append_token(branch.node_id, i as u32).unwrap();
        }

        // Add branch-specific tokens
        for i in 0..config.tokens_per_branch {
            let token = (b * 1000 + i) as u32;
            state.append_token(branch.node_id, token).unwrap();
        }
    }

    // In linear mode, prefix is duplicated for each branch
    let prefix_blocks = (config.prefix_tokens + 15) / 16;
    let branch_blocks = (config.tokens_per_branch + 15) / 16;
    let duplicated_blocks = prefix_blocks * config.num_branches;
    let total_branch_blocks = branch_blocks * config.num_branches;

    (duplicated_blocks, duplicated_blocks + total_branch_blocks)
}

/// Benchmark: Compare branch creation time.
fn bench_branch_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("branch_creation");

    for prefix_tokens in [256, 1024, 4096].iter() {
        let config = BenchConfig {
            prefix_tokens: *prefix_tokens,
            num_branches: 4,
            tokens_per_branch: 0, // Just measuring fork
            ..Default::default()
        };

        // Tree-structured branching
        group.bench_with_input(
            BenchmarkId::new("tree", prefix_tokens),
            prefix_tokens,
            |b, _| {
                let cache = create_cache(config.max_blocks);
                let state = TreeState::new(cache, 16);

                // Setup prefix
                let prefix = state.fork(NodeId::ROOT).unwrap();
                for i in 0..config.prefix_tokens {
                    state.append_token(prefix.node_id, i as u32).unwrap();
                }

                b.iter(|| {
                    // Create branches (O(1) each)
                    for _ in 0..config.num_branches {
                        let handle = state.fork(black_box(prefix.node_id)).unwrap();
                        black_box(handle);
                    }
                })
            },
        );

        // Linear branching (simulate full copy overhead)
        group.bench_with_input(
            BenchmarkId::new("linear_simulated", prefix_tokens),
            prefix_tokens,
            |b, _| {
                let cache = create_cache(config.max_blocks);

                b.iter(|| {
                    // Each branch would need to copy prefix
                    for _ in 0..config.num_branches {
                        let state = TreeState::new(cache.clone(), 16);
                        let branch = state.fork(NodeId::ROOT).unwrap();

                        // Simulating O(n) copy of prefix
                        for i in 0..config.prefix_tokens {
                            state.append_token(branch.node_id, black_box(i as u32)).unwrap();
                        }
                    }
                })
            },
        );
    }

    group.finish();
}

/// Benchmark: Memory efficiency in beam search scenario.
fn bench_beam_search_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("beam_memory");

    for beam_width in [4, 8, 16].iter() {
        let config = BenchConfig {
            prefix_tokens: 1024,
            num_branches: *beam_width,
            tokens_per_branch: 64,
            ..Default::default()
        };

        group.throughput(Throughput::Elements(*beam_width as u64));

        // Tree-structured beam
        group.bench_with_input(
            BenchmarkId::new("tree", beam_width),
            beam_width,
            |b, _| {
                b.iter(|| {
                    let (fork_blocks, total_blocks) = tree_branch_simulation(&config);
                    black_box((fork_blocks, total_blocks))
                })
            },
        );
    }

    group.finish();
}

/// Benchmark: Multi-step branching (like MCTS).
fn bench_multi_step_branching(c: &mut Criterion) {
    let mut group = c.benchmark_group("mcts_branching");

    for branching_factor in [2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::new("branching_factor", branching_factor),
            branching_factor,
            |b, &bf| {
                let cache = create_cache(65536);
                let state = TreeState::new(cache, 16);

                b.iter(|| {
                    // Create tree structure
                    let root = state.fork(NodeId::ROOT).unwrap();

                    // Add some initial tokens
                    for i in 0..256 {
                        state.append_token(root.node_id, i as u32).unwrap();
                    }

                    // Level 1 branches
                    let mut level1 = Vec::new();
                    for _ in 0..bf {
                        level1.push(state.fork(root.node_id).unwrap());
                    }

                    // Level 2 branches (from each level 1)
                    for l1 in &level1 {
                        for _ in 0..bf {
                            let l2 = state.fork(l1.node_id).unwrap();
                            black_box(l2);
                        }
                    }
                })
            },
        );
    }

    group.finish();
}

/// Benchmark: Speculative decoding scenario.
///
/// In speculative decoding, we fork to speculatively generate tokens,
/// then accept or reject based on verification.
fn bench_speculative_decoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("speculative_decoding");

    for spec_length in [4, 8, 16].iter() {
        group.bench_with_input(
            BenchmarkId::new("spec_length", spec_length),
            spec_length,
            |b, &spec_len| {
                let cache = create_cache(65536);
                let state = TreeState::new(cache.clone(), 16);

                // Setup base sequence
                let base = state.fork(NodeId::ROOT).unwrap();
                for i in 0..512 {
                    state.append_token(base.node_id, i as u32).unwrap();
                }

                b.iter(|| {
                    // Fork for speculative generation
                    let spec = state.fork(black_box(base.node_id)).unwrap();

                    // Generate speculative tokens
                    for i in 0..spec_len {
                        state.append_token(spec.node_id, (1000 + i) as u32).unwrap();
                    }

                    // Simulate 50% acceptance (release half the time)
                    if spec_len % 2 == 0 {
                        state.release(spec.node_id).unwrap();
                    }

                    black_box(spec)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark: Block usage comparison.
fn bench_block_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_efficiency");

    let config = BenchConfig {
        prefix_tokens: 2048,
        num_branches: 8,
        tokens_per_branch: 256,
        ..Default::default()
    };

    group.bench_function("tree_block_usage", |b| {
        b.iter(|| {
            let (fork_blocks, total_blocks) = tree_branch_simulation(&config);
            black_box((fork_blocks, total_blocks))
        })
    });

    group.bench_function("linear_block_usage", |b| {
        b.iter(|| {
            let (dup_blocks, total_blocks) = linear_branch_simulation(&config);
            black_box((dup_blocks, total_blocks))
        })
    });

    group.finish();
}

/// Benchmark: Full scenario - tree of thought pattern.
fn bench_tree_of_thought(c: &mut Criterion) {
    let mut group = c.benchmark_group("tree_of_thought");

    // Simulate: problem (1K tokens) -> 3 thoughts -> 3 evaluations each
    let config = BenchConfig {
        prefix_tokens: 1024,  // Problem statement
        num_branches: 3,       // Thoughts
        tokens_per_branch: 512, // Reasoning per thought
        ..Default::default()
    };

    group.bench_function("tot_simulation", |b| {
        b.iter(|| {
            let cache = create_cache(config.max_blocks);
            let state = TreeState::new(cache.clone(), 16);

            // Problem statement
            let problem = state.fork(NodeId::ROOT).unwrap();
            for i in 0..config.prefix_tokens {
                state.append_token(problem.node_id, i as u32).unwrap();
            }

            // Generate 3 thoughts
            let mut thoughts = Vec::new();
            for t in 0..3 {
                let thought = state.fork(problem.node_id).unwrap();
                for i in 0..config.tokens_per_branch {
                    state.append_token(thought.node_id, (t * 1000 + i) as u32).unwrap();
                }
                thoughts.push(thought);
            }

            // Evaluate each thought with 3 branches
            for (t, thought) in thoughts.iter().enumerate() {
                for e in 0..3 {
                    let eval = state.fork(thought.node_id).unwrap();
                    for i in 0..128 {
                        state
                            .append_token(eval.node_id, ((t + 1) * 10000 + e * 1000 + i) as u32)
                            .unwrap();
                    }
                }
            }

            let used = cache.read().used_blocks();
            used
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_branch_creation,
    bench_beam_search_memory,
    bench_multi_step_branching,
    bench_speculative_decoding,
    bench_block_efficiency,
    bench_tree_of_thought,
);

criterion_main!(benches);
