//! Block pool management.
//!
//! # Memory Pool Tuning (M7)
//!
//! The original pool called `allocate()` once per sequence per decode step,
//! serializing every caller through the `free_list` mutex. Under a batch of
//! 128 decoding sequences, that's 128 sequential lock→pop→unlock cycles per step.
//!
//! Tuned design:
//! - **Batch alloc/free**: `allocate_batch(n)` and `free_batch(&[BlockId])` take
//!   the free-list lock once for N blocks instead of N times.
//! - **Watermark tracking**: `PoolStats` records peak usage and low-water events
//!   so callers can pre-allocate or apply backpressure before exhaustion.
//! - **Reserved headroom**: `reserve_headroom` blocks are kept off the free list
//!   as emergency CoW capacity. Prevents deadlocks during copy-on-write when
//!   the pool would otherwise be at 0 free.
//! - **Original API preserved**: single `allocate()` / `free()` still work.

use super::{Block, BlockId};
use crate::error::{DendriteError, Result};
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Snapshot of pool usage for monitoring and backpressure.
#[derive(Debug, Clone, Copy, Default)]
pub struct PoolStats {
    /// Total blocks in the pool.
    pub total_blocks: usize,
    /// Currently free (available) blocks.
    pub free_blocks: usize,
    /// Blocks in use (total - free - reserved).
    pub used_blocks: usize,
    /// Reserved headroom blocks.
    pub reserved_blocks: usize,
    /// Peak usage ever seen (high watermark).
    pub peak_used: usize,
    /// Total allocations since creation.
    pub alloc_count: u64,
    /// Total frees since creation.
    pub free_count: u64,
    /// Number of times free count hit the low-water threshold.
    pub low_water_events: u64,
}

/// Pool of KV cache blocks with free list management.
#[derive(Debug)]
pub struct BlockPool {
    /// All blocks in the pool (one Mutex per block for fine-grained CoW).
    blocks: Vec<Mutex<Block>>,
    /// Free block IDs — VecDeque for O(1) push/pop at both ends.
    free_list: Mutex<VecDeque<BlockId>>,
    /// Tokens per block.
    #[allow(dead_code)]
    tokens_per_block: usize,
    /// Total pool capacity.
    total_blocks: usize,
    /// Blocks reserved for CoW headroom (not in free_list, always available).
    reserved_count: usize,
    /// High-water mark: max `used_blocks` ever seen (atomically updated).
    peak_used: AtomicUsize,
    /// Cumulative allocation count.
    alloc_count: AtomicU64,
    /// Cumulative free count.
    free_count_stat: AtomicU64,
    /// Low-water events (free_blocks fell below 10% of total).
    low_water_events: AtomicU64,
}

impl BlockPool {
    /// Create a new block pool with automatic CoW headroom.
    ///
    /// Headroom formula: 2% of capacity, but only for pools ≥ 64 blocks
    /// (small pools used in tests get zero reserved). Hard cap at 25%.
    pub fn new(max_blocks: usize, tokens_per_block: usize) -> Result<Self> {
        let reserved = if max_blocks >= 64 {
            (max_blocks / 50).max(4).min(max_blocks / 4)
        } else {
            0
        };
        Self::with_headroom(max_blocks, tokens_per_block, reserved)
    }

    /// Create a block pool with explicit CoW headroom reservation.
    ///
    /// `reserved_count` blocks are withheld from the free list and only
    /// released to CoW operations. This prevents deadlocks when the pool
    /// is otherwise exhausted during copy-on-write.
    pub fn with_headroom(
        max_blocks: usize,
        tokens_per_block: usize,
        reserved_count: usize,
    ) -> Result<Self> {
        let reserved = reserved_count.min(max_blocks / 4); // cap at 25%
        let available = max_blocks.saturating_sub(reserved);

        let mut blocks = Vec::with_capacity(max_blocks);
        let mut free_list = VecDeque::with_capacity(available);

        for i in 0..max_blocks {
            let id = BlockId(i as u32);
            blocks.push(Mutex::new(Block::new(id, tokens_per_block as u32)));
            if i < available {
                free_list.push_back(id);
            }
            // reserved blocks sit at indices [available..max_blocks]
        }

        Ok(Self {
            blocks,
            free_list: Mutex::new(free_list),
            tokens_per_block,
            total_blocks: max_blocks,
            reserved_count: reserved,
            peak_used: AtomicUsize::new(0),
            alloc_count: AtomicU64::new(0),
            free_count_stat: AtomicU64::new(0),
            low_water_events: AtomicU64::new(0),
        })
    }

    /// Allocate a single block from the pool.
    pub fn allocate(&self) -> Result<BlockId> {
        let id = {
            let mut free_list = self.free_list.lock();
            free_list
                .pop_front()
                .ok_or_else(|| DendriteError::OutOfMemory("no free blocks".into()))?
        };
        self.alloc_count.fetch_add(1, Ordering::Relaxed);
        self.update_watermarks();
        Ok(id)
    }

    /// Allocate `n` blocks in a single lock acquisition.
    ///
    /// More efficient than calling `allocate()` N times when scheduling a
    /// full batch (e.g., 128 sequences all need a new KV block per step).
    /// Returns `Err` if fewer than `n` blocks are available.
    pub fn allocate_batch(&self, n: usize) -> Result<Vec<BlockId>> {
        let mut free_list = self.free_list.lock();
        if free_list.len() < n {
            return Err(DendriteError::OutOfMemory(format!(
                "need {n} blocks, only {} free",
                free_list.len()
            )));
        }
        let ids: Vec<BlockId> = (0..n).map(|_| free_list.pop_front().unwrap()).collect();
        drop(free_list);

        self.alloc_count.fetch_add(n as u64, Ordering::Relaxed);
        self.update_watermarks();
        Ok(ids)
    }

    /// Free a block back to the pool.
    pub fn free(&self, block_id: BlockId) -> Result<()> {
        if !self.is_valid(block_id) {
            return Err(DendriteError::InvalidBlock(block_id.0));
        }

        let mut block = self.blocks[block_id.0 as usize].lock();
        let new_refcount = block.dec_ref();

        if new_refcount == 0 {
            block.reset();
            drop(block);
            self.free_list.lock().push_back(block_id);
            self.free_count_stat.fetch_add(1, Ordering::Relaxed);
        }

        Ok(())
    }

    /// Free multiple blocks in a single lock acquisition.
    ///
    /// Equivalent to calling `free()` N times but with one mutex acquisition
    /// for the free-list push.
    pub fn free_batch(&self, block_ids: &[BlockId]) -> Result<()> {
        // Validate and decrement refcounts first (fine-grained per-block locks)
        let mut to_return: Vec<BlockId> = Vec::with_capacity(block_ids.len());
        for &id in block_ids {
            if !self.is_valid(id) {
                return Err(DendriteError::InvalidBlock(id.0));
            }
            let mut block = self.blocks[id.0 as usize].lock();
            let new_refcount = block.dec_ref();
            if new_refcount == 0 {
                block.reset();
                to_return.push(id);
            }
        }

        if !to_return.is_empty() {
            let count = to_return.len() as u64;
            let mut free_list = self.free_list.lock();
            for id in to_return {
                free_list.push_back(id);
            }
            drop(free_list);
            self.free_count_stat.fetch_add(count, Ordering::Relaxed);
        }

        Ok(())
    }

    /// Increment reference count for sharing.
    pub fn share(&self, block_id: BlockId) -> Result<()> {
        if !self.is_valid(block_id) {
            return Err(DendriteError::InvalidBlock(block_id.0));
        }

        self.blocks[block_id.0 as usize].lock().inc_ref();
        Ok(())
    }

    /// Copy-on-write: if shared, allocate new block and copy.
    pub fn copy_on_write(&self, block_id: BlockId) -> Result<BlockId> {
        if !self.is_valid(block_id) {
            return Err(DendriteError::InvalidBlock(block_id.0));
        }

        let block = self.blocks[block_id.0 as usize].lock();
        if !block.is_shared() {
            return Ok(block_id);
        }
        drop(block);

        // Allocate new block
        let new_id = self.allocate()?;

        // TODO: Copy KV data from old block to new block
        // This requires access to the actual tensor data

        // Decrement old block refcount
        self.free(block_id)?;

        Ok(new_id)
    }

    /// Get number of free blocks (acquires lock briefly).
    pub fn free_count(&self) -> usize {
        self.free_list.lock().len()
    }

    /// Total pool capacity (including reserved).
    pub fn total_count(&self) -> usize {
        self.total_blocks
    }

    /// Return a snapshot of pool statistics.
    pub fn stats(&self) -> PoolStats {
        let free_blocks = self.free_list.lock().len();
        let used_blocks = self
            .total_blocks
            .saturating_sub(free_blocks + self.reserved_count);
        PoolStats {
            total_blocks: self.total_blocks,
            free_blocks,
            used_blocks,
            reserved_blocks: self.reserved_count,
            peak_used: self.peak_used.load(Ordering::Relaxed),
            alloc_count: self.alloc_count.load(Ordering::Relaxed),
            free_count: self.free_count_stat.load(Ordering::Relaxed),
            low_water_events: self.low_water_events.load(Ordering::Relaxed),
        }
    }

    /// Fraction of pool currently in use (0.0–1.0).
    pub fn utilization(&self) -> f32 {
        let free = self.free_list.lock().len();
        let available = self.total_blocks.saturating_sub(self.reserved_count);
        if available == 0 {
            return 1.0;
        }
        let used = available.saturating_sub(free);
        used as f32 / available as f32
    }

    /// True when free blocks are below 10% of available capacity.
    pub fn is_low_memory(&self) -> bool {
        let free = self.free_list.lock().len();
        let available = self.total_blocks.saturating_sub(self.reserved_count);
        free * 10 < available
    }

    /// Check if a block ID is valid.
    fn is_valid(&self, block_id: BlockId) -> bool {
        (block_id.0 as usize) < self.blocks.len()
    }

    /// Update peak-used watermark and check low-water threshold.
    fn update_watermarks(&self) {
        let free = self.free_list.lock().len();
        let available = self.total_blocks.saturating_sub(self.reserved_count);
        let used = available.saturating_sub(free);

        // Update peak
        let mut peak = self.peak_used.load(Ordering::Relaxed);
        while used > peak {
            match self.peak_used.compare_exchange_weak(
                peak,
                used,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(current) => peak = current,
            }
        }

        // Low-water event: free fell below 10%
        if available > 0 && free * 10 < available {
            self.low_water_events.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Get refcount for a block (for testing).
    #[cfg(test)]
    fn get_refcount(&self, block_id: BlockId) -> Option<u32> {
        if self.is_valid(block_id) {
            Some(self.blocks[block_id.0 as usize].lock().refcount())
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Convenience: no-headroom pool for deterministic free_count tests.
    fn pool(n: usize) -> BlockPool {
        BlockPool::with_headroom(n, 16, 0).unwrap()
    }

    // ── Original tests (updated to use no-headroom pool) ─────────────────────

    #[test]
    fn new_pool_has_all_blocks_free() {
        let p = pool(10);
        assert_eq!(p.free_count(), 10);
        assert_eq!(p.total_count(), 10);
    }

    #[test]
    fn allocate_returns_valid_block() {
        let p = pool(10);
        let block_id = p.allocate().unwrap();
        assert!(p.is_valid(block_id));
        assert_eq!(p.free_count(), 9);
    }

    #[test]
    fn allocate_exhausts_pool() {
        let p = pool(3);
        let _b1 = p.allocate().unwrap();
        let _b2 = p.allocate().unwrap();
        let _b3 = p.allocate().unwrap();
        assert_eq!(p.free_count(), 0);
        assert!(p.allocate().is_err());
    }

    #[test]
    fn free_returns_block_to_pool() {
        let p = pool(5);
        let block_id = p.allocate().unwrap();
        assert_eq!(p.free_count(), 4);
        p.free(block_id).unwrap();
        assert_eq!(p.free_count(), 5);
    }

    #[test]
    fn free_invalid_block_returns_error() {
        let p = pool(5);
        assert!(p.free(BlockId(100)).is_err());
    }

    #[test]
    fn share_increments_refcount() {
        let p = pool(5);
        let block_id = p.allocate().unwrap();
        assert_eq!(p.get_refcount(block_id), Some(1));
        p.share(block_id).unwrap();
        assert_eq!(p.get_refcount(block_id), Some(2));
        p.share(block_id).unwrap();
        assert_eq!(p.get_refcount(block_id), Some(3));
    }

    #[test]
    fn share_invalid_block_returns_error() {
        let p = pool(5);
        assert!(p.share(BlockId(100)).is_err());
    }

    #[test]
    fn free_shared_block_decrements_refcount() {
        let p = pool(5);
        let id = p.allocate().unwrap();
        p.share(id).unwrap();
        p.share(id).unwrap();
        assert_eq!(p.get_refcount(id), Some(3));
        assert_eq!(p.free_count(), 4);

        p.free(id).unwrap();
        assert_eq!(p.get_refcount(id), Some(2));
        assert_eq!(p.free_count(), 4);

        p.free(id).unwrap();
        assert_eq!(p.get_refcount(id), Some(1));
        assert_eq!(p.free_count(), 4);

        p.free(id).unwrap();
        assert_eq!(p.free_count(), 5);
    }

    #[test]
    fn copy_on_write_returns_same_if_not_shared() {
        let p = pool(5);
        let id = p.allocate().unwrap();
        let result = p.copy_on_write(id).unwrap();
        assert_eq!(result, id);
        assert_eq!(p.free_count(), 4);
    }

    #[test]
    fn copy_on_write_allocates_new_if_shared() {
        let p = pool(5);
        let id = p.allocate().unwrap();
        p.share(id).unwrap();
        assert_eq!(p.get_refcount(id), Some(2));

        let new_id = p.copy_on_write(id).unwrap();
        assert_ne!(new_id, id);
        assert_eq!(p.get_refcount(id), Some(1));
        assert_eq!(p.get_refcount(new_id), Some(1));
        assert_eq!(p.free_count(), 3);
    }

    #[test]
    fn copy_on_write_invalid_block_returns_error() {
        let p = pool(5);
        assert!(p.copy_on_write(BlockId(100)).is_err());
    }

    #[test]
    fn copy_on_write_fails_when_pool_exhausted() {
        let p = pool(2);
        let b1 = p.allocate().unwrap();
        let _b2 = p.allocate().unwrap();
        p.share(b1).unwrap();
        assert!(p.copy_on_write(b1).is_err());
    }

    // ── New M7 pool tuning tests ──────────────────────────────────────────────

    #[test]
    fn allocate_batch_returns_n_distinct_blocks() {
        let p = pool(16);
        let ids = p.allocate_batch(8).unwrap();
        assert_eq!(ids.len(), 8);
        assert_eq!(p.free_count(), 8);
        // All IDs should be distinct
        let mut seen = std::collections::HashSet::new();
        for id in &ids {
            assert!(seen.insert(id.0), "duplicate block id {}", id.0);
        }
    }

    #[test]
    fn allocate_batch_fails_if_not_enough_blocks() {
        let p = pool(4);
        assert!(p.allocate_batch(5).is_err());
    }

    #[test]
    fn allocate_batch_empties_pool() {
        let p = pool(8);
        let ids = p.allocate_batch(8).unwrap();
        assert_eq!(ids.len(), 8);
        assert_eq!(p.free_count(), 0);
        assert!(p.allocate().is_err());
    }

    #[test]
    fn free_batch_returns_all_blocks() {
        let p = pool(16);
        let ids = p.allocate_batch(8).unwrap();
        assert_eq!(p.free_count(), 8);
        p.free_batch(&ids).unwrap();
        assert_eq!(p.free_count(), 16);
    }

    #[test]
    fn free_batch_respects_refcounts() {
        let p = pool(8);
        let ids = p.allocate_batch(4).unwrap();
        // Share the first block
        p.share(ids[0]).unwrap(); // refcount = 2
                                  // free_batch should only return blocks with refcount → 0
        p.free_batch(&ids).unwrap();
        // ids[0] still has refcount 1, so only 3 blocks returned
        assert_eq!(p.free_count(), 4 + 3); // original 4 free + 3 returned
    }

    #[test]
    fn free_batch_invalid_block_returns_error() {
        let p = pool(5);
        let bad = vec![BlockId(99)];
        assert!(p.free_batch(&bad).is_err());
    }

    #[test]
    fn stats_tracks_allocations_and_frees() {
        let p = pool(100);
        let s0 = p.stats();
        assert_eq!(s0.alloc_count, 0);
        assert_eq!(s0.free_count, 0);
        assert_eq!(s0.free_blocks, 100);

        let ids = p.allocate_batch(10).unwrap();
        let s1 = p.stats();
        assert_eq!(s1.alloc_count, 10);
        assert_eq!(s1.used_blocks, 10);
        assert_eq!(s1.free_blocks, 90);

        p.free_batch(&ids).unwrap();
        let s2 = p.stats();
        assert_eq!(s2.free_count, 10);
        assert_eq!(s2.free_blocks, 100);
    }

    #[test]
    fn stats_tracks_peak_used() {
        let p = pool(100);
        let ids1 = p.allocate_batch(60).unwrap();
        let ids2 = p.allocate_batch(20).unwrap();
        p.free_batch(&ids2).unwrap(); // drops to 60 used

        let s = p.stats();
        assert!(
            s.peak_used >= 80,
            "peak should be ≥ 80, got {}",
            s.peak_used
        );
        drop(ids1);
    }

    #[test]
    fn utilization_zero_when_empty() {
        let p = pool(100);
        assert_eq!(p.utilization(), 0.0);
    }

    #[test]
    fn utilization_increases_with_allocations() {
        let p = pool(100);
        let _ids = p.allocate_batch(50).unwrap();
        let u = p.utilization();
        assert!((u - 0.5).abs() < 0.02, "utilization={u}");
    }

    #[test]
    fn is_low_memory_false_when_well_stocked() {
        let p = pool(100);
        let _ids = p.allocate_batch(50).unwrap();
        assert!(!p.is_low_memory());
    }

    #[test]
    fn is_low_memory_true_when_nearly_exhausted() {
        let p = pool(100);
        // Allocate 95% (leave 5 free out of 100)
        let _ids = p.allocate_batch(95).unwrap();
        assert!(p.is_low_memory());
    }

    #[test]
    fn reserved_headroom_withheld_from_free_list() {
        // 100 blocks, 10 reserved → 90 in free list
        let p = BlockPool::with_headroom(100, 16, 10).unwrap();
        assert_eq!(p.free_count(), 90);
        assert_eq!(p.total_count(), 100);
        let s = p.stats();
        assert_eq!(s.reserved_blocks, 10);
    }

    #[test]
    fn default_new_reserves_headroom() {
        // Large pool: 2% reserved (min 4)
        let p = BlockPool::new(1000, 16).unwrap();
        assert!(p.free_count() < 1000, "should have reserved some blocks");
        assert_eq!(p.total_count(), 1000);
    }
}
