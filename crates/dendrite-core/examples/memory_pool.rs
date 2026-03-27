//! Memory pool demonstration.
//!
//! Shows `BlockPool`'s batch allocation API and the PoolStats watermark
//! system. All operations are pure CPU — no GPU or model weights required.
//!
//! Run with:
//! ```bash
//! cargo run -p dendrite-core --example memory_pool
//! ```

use dendrite_core::cache::{BlockPool, KvCache, KvCacheConfig};
use std::time::Instant;

fn main() {
    println!("Memory Pool Demo");
    println!("================\n");

    demo_batch_vs_sequential();
    demo_watermarks();
    demo_cow_headroom();
    demo_kvcache_stats();
}

/// Compare batch allocation vs sequential for a 128-sequence decode step.
fn demo_batch_vs_sequential() {
    println!("[1] Batch vs sequential allocation (128-sequence decode step)\n");

    const N: usize = 128;
    const POOL: usize = 4096;

    // Sequential: 128 individual lock acquisitions
    let pool_seq = BlockPool::with_headroom(POOL, 16, 0).unwrap();
    let t0 = Instant::now();
    let mut ids_seq = Vec::with_capacity(N);
    for _ in 0..N {
        ids_seq.push(pool_seq.allocate().unwrap());
    }
    let seq_us = t0.elapsed().as_nanos();

    // Batch: 1 lock acquisition
    let pool_batch = BlockPool::with_headroom(POOL, 16, 0).unwrap();
    let t1 = Instant::now();
    let ids_batch = pool_batch.allocate_batch(N).unwrap();
    let batch_us = t1.elapsed().as_nanos();

    println!("  Sequential ({N} allocate() calls): {}ns", seq_us);
    println!("  Batch     (1 allocate_batch({N})): {}ns", batch_us);
    let speedup = seq_us as f64 / batch_us.max(1) as f64;
    println!("  Speedup: {speedup:.1}x\n");

    // Cleanup
    pool_seq.free_batch(&ids_seq).unwrap();
    pool_batch.free_batch(&ids_batch).unwrap();
}

/// Show high-water mark and low-water event tracking.
fn demo_watermarks() {
    println!("[2] PoolStats watermark tracking\n");

    let pool = BlockPool::with_headroom(1000, 16, 0).unwrap();

    // Initial state
    let s0 = pool.stats();
    println!("  Initial: free={}, used={}, peak={}", s0.free_blocks, s0.used_blocks, s0.peak_used);

    // First wave: allocate 600 blocks
    let wave1 = pool.allocate_batch(600).unwrap();
    let s1 = pool.stats();
    println!("  After alloc 600: free={}, used={}, peak={}", s1.free_blocks, s1.used_blocks, s1.peak_used);

    // Second wave: allocate 300 more (total 900 — peak)
    let wave2 = pool.allocate_batch(300).unwrap();
    let s2 = pool.stats();
    println!("  After alloc 300: free={}, used={}, peak={}", s2.free_blocks, s2.used_blocks, s2.peak_used);

    // Free wave2 — peak stays at 900
    pool.free_batch(&wave2).unwrap();
    let s3 = pool.stats();
    println!("  After free 300:  free={}, used={}, peak={} (peak preserved)", s3.free_blocks, s3.used_blocks, s3.peak_used);

    // Utilization
    println!("  Utilization: {:.1}%", pool.utilization() * 100.0);
    println!("  Low memory:  {}", pool.is_low_memory());

    // Drive into low-memory territory (> 90% used)
    let wave3 = pool.allocate_batch(350).unwrap();
    println!("  After alloc 350: low_memory={}", pool.is_low_memory());
    let s4 = pool.stats();
    println!("  Low-water events: {}\n", s4.low_water_events);

    pool.free_batch(&wave1).unwrap();
    pool.free_batch(&wave3).unwrap();
}

/// Show CoW headroom reservation preventing deadlock.
fn demo_cow_headroom() {
    println!("[3] CoW headroom reservation\n");

    // Pool with explicit headroom
    let pool = BlockPool::with_headroom(100, 16, 10).unwrap();
    let s = pool.stats();

    println!("  Total blocks:    {}", s.total_blocks);
    println!("  Free (available): {}", s.free_blocks);
    println!("  Reserved (CoW):  {}", s.reserved_blocks);
    println!(
        "  → {} blocks available for normal allocation, {} held as CoW reserve",
        s.free_blocks, s.reserved_blocks
    );

    // Exhaust the available (non-reserved) blocks
    let all = pool.allocate_batch(s.free_blocks).unwrap();
    println!("  After exhausting available: free={}", pool.free_count());
    println!("  Pool returns OutOfMemory for new alloc (reserved blocks protected)");
    let result = pool.allocate();
    println!("  pool.allocate() → {}", if result.is_err() { "Err(OutOfMemory) ✓" } else { "Ok (unexpected)" });
    println!();

    pool.free_batch(&all).unwrap();
}

/// Show KvCache integration with pool stats.
fn demo_kvcache_stats() {
    println!("[4] KvCache pool stats\n");

    let config = KvCacheConfig {
        num_layers: 4,
        num_kv_heads: 8,
        head_dim: 64,
        max_blocks: 256,
        tokens_per_block: 16,
    };
    let mut cache = KvCache::new(config).unwrap();

    println!("  Created KvCache: {} total blocks", cache.total_blocks());
    println!("  Free: {}, Used: {}", cache.free_blocks(), cache.used_blocks());

    // Simulate allocating blocks for a batch
    let mut allocated = Vec::new();
    for _ in 0..32 {
        if let Ok(id) = cache.allocate_block() {
            allocated.push(id);
        }
    }
    println!("  After allocating 32 blocks:");
    println!("  Free: {}, Used: {}", cache.free_blocks(), cache.used_blocks());
    println!("  Utilization: {:.1}%", cache.utilization() * 100.0);

    // Free them
    for id in &allocated {
        cache.free_block(*id).unwrap();
    }
    println!("  After freeing: Free={}, Used={}", cache.free_blocks(), cache.used_blocks());
    println!();

    println!("Done. See `cargo bench --bench scheduler` for perf numbers.");
}
