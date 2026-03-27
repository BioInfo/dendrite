#![allow(missing_docs)]
//! Prefix cache for KV block reuse across sequences.
//!
//! [`PrefixCache`] bridges the [`RadixTree`] (token-sequence lookup) and
//! [`BlockPool`] (physical block management) to enable **cascade attention**:
//! sequences that share a common prefix (e.g. a system prompt) can reuse
//! already-computed KV blocks instead of recomputing them from scratch.
//!
//! # How it works
//!
//! 1. When a sequence finishes or is evicted, its token→block mapping is
//!    **committed** to the prefix cache.
//! 2. When a new request arrives with tokens `[t0, t1, ..., tN]`, the cache
//!    does a **prefix lookup** to find the longest already-stored prefix.
//! 3. The matched blocks are **pinned** (refcount bumped) and returned to the
//!    scheduler, which can skip prefilling those tokens and start generation
//!    from the prefix boundary.
//! 4. Blocks are **unpinned** when the scheduler finishes with them.
//!
//! # Eviction
//!
//! When the pool is low on free blocks, [`PrefixCache::evict_lru`] removes
//! the least-recently-used entries, freeing their blocks back to the pool.
//!
//! # Block granularity
//!
//! Prefix matching is at block granularity (`tokens_per_block` tokens per
//! block). A prefix of 17 tokens with a 16-token block size yields one full
//! cached block (the partial tail block is not cached).
//!
//! # Example
//!
//! ```rust
//! use dendrite_core::cache::{BlockPool, PrefixCache};
//! use std::sync::Arc;
//!
//! let pool = Arc::new(BlockPool::new(128, 16).unwrap());
//! let mut cache = PrefixCache::new(Arc::clone(&pool), 16);
//!
//! // Simulate committing blocks from a completed request
//! let tokens: Vec<u32> = (0..32).collect();
//! let blocks = pool.allocate_batch(2).unwrap();
//! cache.commit(&tokens, &blocks).unwrap();
//!
//! // New request with the same prefix
//! let new_tokens: Vec<u32> = (0..48).collect();
//! let hit = cache.lookup(&new_tokens);
//! assert_eq!(hit.matched_tokens, 32); // 2 full blocks reused
//! assert_eq!(hit.blocks.len(), 2);
//!
//! // Unpin when done
//! cache.unpin(&hit.blocks);
//! ```

use super::{BlockId, BlockPool, RadixTree};
use crate::error::Result;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

// ── Types ─────────────────────────────────────────────────────────────────────

/// A single entry in the prefix cache: a contiguous run of full blocks that
/// cover `tokens` starting at position 0 in that sequence.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct CacheEntry {
    /// Block IDs covering this prefix (in order).
    blocks: Vec<BlockId>,
    /// Number of tokens covered (must equal `blocks.len() * tokens_per_block`).
    token_count: usize,
    /// Last time this entry was accessed (for LRU eviction).
    last_used: Instant,
    /// Number of active users (pins). Only evict when pinned == 0.
    pinned: usize,
}

/// Result of a [`PrefixCache::lookup`].
#[derive(Debug, Default)]
pub struct PrefixHit {
    /// Blocks covering the matched prefix (in order), pinned.
    pub blocks: Vec<BlockId>,
    /// Number of tokens covered by the hit.
    pub matched_tokens: usize,
}

impl PrefixHit {
    /// True if at least one block was matched.
    pub fn is_hit(&self) -> bool {
        !self.blocks.is_empty()
    }

    /// Fraction of `total_tokens` that was a cache hit.
    pub fn hit_rate(&self, total_tokens: usize) -> f64 {
        if total_tokens == 0 {
            return 0.0;
        }
        self.matched_tokens as f64 / total_tokens as f64
    }
}

// ── PrefixCache ───────────────────────────────────────────────────────────────

/// Prefix cache: maps token-sequence prefixes to cached KV blocks.
pub struct PrefixCache {
    /// Token-sequence radix index. Values are entry IDs.
    tree: RadixTree,
    /// Entry storage keyed by the entry ID stored in the radix tree.
    entries: HashMap<usize, CacheEntry>,
    /// Monotonic ID counter for entries.
    next_id: usize,
    /// Block pool reference for pinning/freeing.
    pool: Arc<BlockPool>,
    /// Tokens per block — prefix matching is at this granularity.
    tokens_per_block: usize,
    /// Aggregate hit/miss statistics.
    stats: PrefixCacheStats,
}

impl std::fmt::Debug for PrefixCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrefixCache")
            .field("entries", &self.entries.len())
            .field("tokens_per_block", &self.tokens_per_block)
            .field("stats", &self.stats)
            .finish()
    }
}

/// Cumulative stats for a [`PrefixCache`].
#[derive(Debug, Default, Clone)]
pub struct PrefixCacheStats {
    /// Total lookup calls.
    pub lookups: u64,
    /// Lookups that found at least one block.
    pub hits: u64,
    /// Total tokens saved (not prefilled) due to hits.
    pub tokens_saved: u64,
    /// Total entries committed.
    pub commits: u64,
    /// Total entries evicted.
    pub evictions: u64,
}

impl PrefixCacheStats {
    pub fn hit_rate(&self) -> f64 {
        if self.lookups == 0 {
            return 0.0;
        }
        self.hits as f64 / self.lookups as f64
    }
}

impl PrefixCache {
    /// Create a new prefix cache backed by `pool`.
    pub fn new(pool: Arc<BlockPool>, tokens_per_block: usize) -> Self {
        Self {
            tree: RadixTree::new(),
            entries: HashMap::new(),
            next_id: 0,
            pool,
            tokens_per_block,
            stats: PrefixCacheStats::default(),
        }
    }

    /// Commit a completed sequence's blocks to the prefix cache.
    ///
    /// Only **full** blocks are cached — the partial tail block is ignored.
    /// Blocks are pinned in the pool until evicted.
    ///
    /// `tokens` is the full token sequence; `blocks` are the physical block
    /// IDs in order (blocks[0] covers tokens[0..tokens_per_block], etc.).
    pub fn commit(&mut self, tokens: &[u32], blocks: &[BlockId]) -> Result<()> {
        let full_blocks = tokens.len() / self.tokens_per_block;
        if full_blocks == 0 || blocks.len() < full_blocks {
            return Ok(()); // nothing to cache
        }

        let cached_blocks = blocks[..full_blocks].to_vec();
        let token_count = full_blocks * self.tokens_per_block;
        let key = tokens[..token_count].to_vec();

        // Check if this exact prefix is already present
        if let Some(eid) = self.tree.find_exact(&key) {
            if let Some(e) = self.entries.get_mut(&eid) {
                e.last_used = Instant::now();
            }
            return Ok(()); // already cached
        }

        // Pin all blocks (inc refcount so pool won't reclaim them)
        for &bid in &cached_blocks {
            self.pool.share(bid)?;
        }

        let id = self.next_id;
        self.next_id += 1;

        self.entries.insert(
            id,
            CacheEntry {
                blocks: cached_blocks,
                token_count,
                last_used: Instant::now(),
                pinned: 0,
            },
        );
        self.tree.insert(&key, id);
        self.stats.commits += 1;
        Ok(())
    }

    /// Look up the longest prefix match for `tokens`.
    ///
    /// Matched blocks are **pinned** — call [`unpin`](Self::unpin) when done.
    pub fn lookup(&mut self, tokens: &[u32]) -> PrefixHit {
        self.stats.lookups += 1;

        let (matched_len, entry_id) = self.tree.find_prefix(tokens);
        if matched_len == 0 || entry_id.is_none() {
            return PrefixHit::default();
        }

        let eid = entry_id.unwrap();
        let entry = match self.entries.get_mut(&eid) {
            Some(e) => e,
            None => return PrefixHit::default(),
        };

        // Align to block granularity
        let full_blocks = matched_len / self.tokens_per_block;
        if full_blocks == 0 {
            return PrefixHit::default();
        }

        let blocks = entry.blocks[..full_blocks].to_vec();
        let matched_tokens = full_blocks * self.tokens_per_block;
        entry.pinned += 1;
        entry.last_used = Instant::now();

        self.stats.hits += 1;
        self.stats.tokens_saved += matched_tokens as u64;

        PrefixHit {
            blocks,
            matched_tokens,
        }
    }

    /// Unpin blocks returned from a previous [`lookup`](Self::lookup).
    ///
    /// This decrements the pin count so the entry becomes eligible for eviction.
    pub fn unpin(&mut self, blocks: &[BlockId]) {
        // Find entries whose block set matches and decrement pin
        for entry in self.entries.values_mut() {
            if !entry.blocks.is_empty() && entry.blocks.first() == blocks.first() {
                if entry.pinned > 0 {
                    entry.pinned -= 1;
                }
                break;
            }
        }
    }

    /// Evict the `n` least-recently-used unpinned entries.
    ///
    /// Returns the number actually evicted (may be < `n` if insufficient
    /// unpinned entries exist).
    pub fn evict_lru(&mut self, n: usize) -> usize {
        // Collect unpinned entries sorted by last_used ascending
        let mut candidates: Vec<(usize, Instant)> = self
            .entries
            .iter()
            .filter(|(_, e)| e.pinned == 0)
            .map(|(&id, e)| (id, e.last_used))
            .collect();
        candidates.sort_by_key(|(_, t)| *t);

        let to_evict: Vec<usize> = candidates.into_iter().take(n).map(|(id, _)| id).collect();
        let evicted = to_evict.len();

        for eid in to_evict {
            if let Some(entry) = self.entries.remove(&eid) {
                // Unpin from pool (dec refcount — free returns to pool when count→0)
                for &bid in &entry.blocks {
                    let _ = self.pool.free(bid);
                }
                // Remove from radix tree by value scan
                self.tree.remove_by_value(eid);
                self.stats.evictions += 1;
            }
        }
        evicted
    }

    /// Evict all entries older than `max_age` that are not pinned.
    pub fn evict_older_than(&mut self, max_age: Duration) -> usize {
        let cutoff = Instant::now() - max_age;
        let stale: Vec<usize> = self
            .entries
            .iter()
            .filter(|(_, e)| e.pinned == 0 && e.last_used < cutoff)
            .map(|(&id, _)| id)
            .collect();
        let n = stale.len();
        for eid in stale {
            if let Some(entry) = self.entries.remove(&eid) {
                for &bid in &entry.blocks {
                    let _ = self.pool.free(bid);
                }
                self.tree.remove_by_value(eid);
                self.stats.evictions += 1;
            }
        }
        n
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Cumulative statistics.
    pub fn stats(&self) -> &PrefixCacheStats {
        &self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pool(blocks: usize) -> Arc<BlockPool> {
        Arc::new(BlockPool::new(blocks, 16).unwrap())
    }

    #[test]
    fn prefix_cache_empty_lookup() {
        let pool = make_pool(64);
        let mut cache = PrefixCache::new(pool, 16);
        let hit = cache.lookup(&[1, 2, 3]);
        assert!(!hit.is_hit());
        assert_eq!(hit.matched_tokens, 0);
    }

    #[test]
    fn prefix_cache_exact_hit() {
        let pool = make_pool(64);
        let mut cache = PrefixCache::new(Arc::clone(&pool), 16);

        // 32 tokens = 2 full blocks
        let tokens: Vec<u32> = (0..32).collect();
        let blocks = pool.allocate_batch(2).unwrap();
        cache.commit(&tokens, &blocks).unwrap();

        let hit = cache.lookup(&tokens);
        assert!(hit.is_hit());
        assert_eq!(hit.matched_tokens, 32);
        assert_eq!(hit.blocks.len(), 2);
        assert_eq!(hit.hit_rate(32), 1.0);
    }

    #[test]
    fn prefix_cache_partial_hit() {
        let pool = make_pool(64);
        let mut cache = PrefixCache::new(Arc::clone(&pool), 16);

        // Commit 32-token prefix
        let tokens: Vec<u32> = (0..32).collect();
        let blocks = pool.allocate_batch(2).unwrap();
        cache.commit(&tokens, &blocks).unwrap();

        // Lookup 48 tokens — should match first 32
        let long_tokens: Vec<u32> = (0..48).collect();
        let hit = cache.lookup(&long_tokens);
        assert!(hit.is_hit());
        assert_eq!(hit.matched_tokens, 32);
        assert_eq!(hit.hit_rate(48), 32.0 / 48.0);
    }

    #[test]
    fn prefix_cache_no_partial_blocks() {
        let pool = make_pool(64);
        let mut cache = PrefixCache::new(Arc::clone(&pool), 16);

        // 20 tokens: 1 full + 4 partial — only 1 block should be committed
        let tokens: Vec<u32> = (0..20).collect();
        let blocks = pool.allocate_batch(2).unwrap();
        cache.commit(&tokens, &blocks).unwrap();

        // Lookup exactly 20 tokens
        let hit = cache.lookup(&tokens);
        assert!(hit.is_hit());
        assert_eq!(hit.matched_tokens, 16); // aligned down to block boundary
    }

    #[test]
    fn prefix_cache_miss_on_divergence() {
        let pool = make_pool(64);
        let mut cache = PrefixCache::new(Arc::clone(&pool), 16);

        let tokens: Vec<u32> = (0..32).collect();
        let blocks = pool.allocate_batch(2).unwrap();
        cache.commit(&tokens, &blocks).unwrap();

        // Completely different tokens
        let other: Vec<u32> = (100..132).collect();
        let hit = cache.lookup(&other);
        assert!(!hit.is_hit());
    }

    #[test]
    fn prefix_cache_lru_eviction() {
        let pool = make_pool(128);
        let mut cache = PrefixCache::new(Arc::clone(&pool), 16);

        // Commit two entries
        let t1: Vec<u32> = (0..32).collect();
        let b1 = pool.allocate_batch(2).unwrap();
        cache.commit(&t1, &b1).unwrap();

        let t2: Vec<u32> = (100..132).collect();
        let b2 = pool.allocate_batch(2).unwrap();
        cache.commit(&t2, &b2).unwrap();

        assert_eq!(cache.len(), 2);

        let evicted = cache.evict_lru(1);
        assert_eq!(evicted, 1);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn prefix_cache_stats_accumulate() {
        let pool = make_pool(64);
        let mut cache = PrefixCache::new(Arc::clone(&pool), 16);

        let tokens: Vec<u32> = (0..32).collect();
        let blocks = pool.allocate_batch(2).unwrap();
        cache.commit(&tokens, &blocks).unwrap();

        cache.lookup(&tokens);
        cache.lookup(&tokens);
        let miss_tokens: Vec<u32> = (200..232).collect();
        cache.lookup(&miss_tokens);

        let s = cache.stats();
        assert_eq!(s.lookups, 3);
        assert_eq!(s.hits, 2);
        assert_eq!(s.tokens_saved, 64); // 2 hits × 32 tokens
        assert!((s.hit_rate() - 2.0 / 3.0).abs() < 1e-9);
    }
}
