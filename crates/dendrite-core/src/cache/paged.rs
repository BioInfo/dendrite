//! Paged KV cache for efficient tree-structured inference.
//!
//! This module provides a paged memory management system for KV cache that enables:
//! - O(1) fork latency via copy-on-write pages
//! - Memory efficient sharing across branches
//! - Compatible with FlashInfer's paged attention kernels
//!
//! # Architecture
//!
//! ```text
//! Page Pool (fixed size pages)
//! ┌──────────────────────────────────┐
//! │ Page 0 │ Page 1 │ Page 2 │ ...  │
//! └──────────────────────────────────┘
//!        ↑         ↑
//!        │         └─────────┐
//!        │                   │
//! ┌──────┴──────┐     ┌──────┴──────┐
//! │  Sequence A │     │  Sequence B │  (forked from A)
//! │  [0, 1]     │     │  [0, 1, 2]  │  (shares pages 0,1)
//! └─────────────┘     └─────────────┘
//! ```
//!
//! # Copy-on-Write
//!
//! When a sequence is forked:
//! 1. New sequence gets a reference to parent's page table
//! 2. Pages are reference-counted, not copied
//! 3. When a shared page is modified, it's copied first (CoW)
//!
//! This enables O(1) fork instead of O(context_length).

use crate::error::{DendriteError, Result};
use candle_core::{DType, Device, Tensor};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::Arc;

/// Default page size in tokens.
pub const DEFAULT_PAGE_SIZE: usize = 16;

/// A unique page identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PageId(pub u32);

impl PageId {
    /// Create a new page ID.
    pub fn new(id: u32) -> Self {
        Self(id)
    }
}

/// Storage format for KV cache pages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageFormat {
    /// Standard FP16/BF16 storage: [2, kv_heads, page_size, head_dim]
    Full,
    /// TurboQuant 4-bit packed: indices as uint8 (2 values/byte) + fp16 norms.
    /// Indices: [2, kv_heads, page_size, head_dim/2] (uint8, packed)
    /// Norms: [2, kv_heads, page_size, 1] (fp16)
    /// Compression: ~3.9x vs FP16 at scale.
    TurboQuant4Bit,
    /// TurboQuant 2-bit packed: indices as uint8 (4 values/byte) + fp16 norms.
    /// Compression: ~7.5x vs FP16 at scale.
    TurboQuant2Bit,
}

impl PageFormat {
    /// Bytes per element for quantized indices.
    pub fn bytes_per_element(&self) -> f64 {
        match self {
            PageFormat::Full => 2.0,           // fp16
            PageFormat::TurboQuant4Bit => 0.5, // 4 bits packed
            PageFormat::TurboQuant2Bit => 0.25, // 2 bits packed
        }
    }

    /// Memory per token per KV head (bytes).
    pub fn bytes_per_token_per_head(&self, head_dim: usize) -> usize {
        match self {
            PageFormat::Full => head_dim * 2 * 2, // K+V, fp16
            PageFormat::TurboQuant4Bit => {
                // Packed indices: head_dim/2 bytes + 2 bytes norm, for K+V
                (head_dim / 2 + 2) * 2
            }
            PageFormat::TurboQuant2Bit => {
                // Packed indices: head_dim/4 bytes + 2 bytes norm, for K+V
                (head_dim / 4 + 2) * 2
            }
        }
    }

    /// Compression ratio vs FP16.
    pub fn compression_ratio(&self, head_dim: usize) -> f64 {
        let fp16 = PageFormat::Full.bytes_per_token_per_head(head_dim) as f64;
        let this = self.bytes_per_token_per_head(head_dim) as f64;
        fp16 / this
    }
}

/// A single page of KV cache data.
///
/// Each page stores `page_size` tokens worth of keys and values.
/// Shape depends on format:
/// - Full: [2, num_kv_heads, page_size, head_dim] (fp16)
/// - TurboQuant4Bit: indices [2, kv_heads, page_size, head_dim/2] (u8) + norms
#[derive(Debug)]
pub struct Page {
    /// Page identifier.
    id: PageId,
    /// Reference count for copy-on-write.
    ref_count: AtomicU32,
    /// Number of valid tokens in this page (0 to page_size).
    len: AtomicUsize,
    /// KV data: [2, num_kv_heads, page_size, head_dim]
    data: Option<Tensor>,
    /// Storage format (defaults to Full for backward compatibility).
    format: PageFormat,
    /// Quantization norms (only for TurboQuant formats).
    /// Shape: [2, num_kv_heads, page_size, 1]
    norms: Option<Tensor>,
}

impl Page {
    /// Create a new empty page.
    pub fn new(id: PageId) -> Self {
        Self {
            id,
            ref_count: AtomicU32::new(1),
            len: AtomicUsize::new(0),
            data: None,
            format: PageFormat::Full,
            norms: None,
        }
    }

    /// Create a page with pre-allocated tensor.
    pub fn with_tensor(
        id: PageId,
        num_kv_heads: usize,
        page_size: usize,
        head_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let data = Tensor::zeros(
            (2, num_kv_heads, page_size, head_dim),
            dtype,
            device,
        )?;

        Ok(Self {
            id,
            ref_count: AtomicU32::new(1),
            len: AtomicUsize::new(0),
            data: Some(data),
            format: PageFormat::Full,
            norms: None,
        })
    }

    /// Create a TurboQuant quantized page (4-bit packed).
    ///
    /// Stores KV as packed uint8 indices (2 values per byte) + fp16 norms.
    /// Uses ~3.9x less memory than FP16 pages at scale.
    pub fn with_turboquant_4bit(
        id: PageId,
        num_kv_heads: usize,
        page_size: usize,
        head_dim: usize,
        device: &Device,
    ) -> Result<Self> {
        // Packed 4-bit: head_dim/2 bytes per element
        let packed_dim = head_dim / 2;
        let data = Tensor::zeros(
            (2, num_kv_heads, page_size, packed_dim),
            DType::U8,
            device,
        )?;
        let norms = Tensor::zeros(
            (2, num_kv_heads, page_size, 1),
            DType::F16,
            device,
        )?;

        Ok(Self {
            id,
            ref_count: AtomicU32::new(1),
            len: AtomicUsize::new(0),
            data: Some(data),
            format: PageFormat::TurboQuant4Bit,
            norms: Some(norms),
        })
    }

    /// Get the storage format.
    pub fn format(&self) -> PageFormat {
        self.format
    }

    /// Get the norms tensor (TurboQuant only).
    pub fn norms(&self) -> Option<&Tensor> {
        self.norms.as_ref()
    }

    /// Get page ID.
    pub fn id(&self) -> PageId {
        self.id
    }

    /// Get reference count.
    pub fn ref_count(&self) -> u32 {
        self.ref_count.load(Ordering::SeqCst)
    }

    /// Increment reference count.
    pub fn inc_ref(&self) -> u32 {
        self.ref_count.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Decrement reference count, returns new count.
    pub fn dec_ref(&self) -> u32 {
        self.ref_count.fetch_sub(1, Ordering::SeqCst) - 1
    }

    /// Get number of valid tokens.
    pub fn len(&self) -> usize {
        self.len.load(Ordering::SeqCst)
    }

    /// Check if page is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Check if page is full.
    pub fn is_full(&self, page_size: usize) -> bool {
        self.len() >= page_size
    }

    /// Get the data tensor.
    pub fn data(&self) -> Option<&Tensor> {
        self.data.as_ref()
    }

    /// Set the data tensor.
    pub fn set_data(&mut self, data: Tensor) {
        self.data = Some(data);
    }

    /// Append a token's KV to this page.
    ///
    /// # Arguments
    /// * `key` - Key tensor [num_kv_heads, 1, head_dim]
    /// * `value` - Value tensor [num_kv_heads, 1, head_dim]
    ///
    /// Returns the token position within the page.
    pub fn append(&mut self, key: &Tensor, value: &Tensor, page_size: usize) -> Result<usize> {
        let pos = self.len.load(Ordering::SeqCst);

        if pos >= page_size {
            return Err(DendriteError::CacheError("Page is full".into()));
        }

        // Write K and V to the appropriate position
        if let Some(ref mut data) = self.data {
            // data shape: [2, num_kv_heads, page_size, head_dim]
            // key shape: [num_kv_heads, 1, head_dim]

            // Squeeze the seq dimension
            let key = key.squeeze(1)?;  // [num_kv_heads, head_dim]
            let value = value.squeeze(1)?;

            // Get slices for K and V
            let k_slice = data.narrow(0, 0, 1)?.squeeze(0)?; // [num_kv_heads, page_size, head_dim]
            let v_slice = data.narrow(0, 1, 1)?.squeeze(0)?;

            // Update at position
            k_slice.slice_set(&key.unsqueeze(1)?, 1, pos)?;
            v_slice.slice_set(&value.unsqueeze(1)?, 1, pos)?;
        }

        self.len.fetch_add(1, Ordering::SeqCst);
        Ok(pos)
    }
}

/// Pool of pages for KV cache.
///
/// Manages allocation and deallocation of pages across sequences.
pub struct PagePool {
    /// All pages in the pool.
    pages: RwLock<Vec<Arc<RwLock<Page>>>>,
    /// Free page indices.
    free_list: RwLock<Vec<usize>>,
    /// Page size in tokens.
    page_size: usize,
    /// Number of KV heads.
    num_kv_heads: usize,
    /// Head dimension.
    head_dim: usize,
    /// Data type.
    dtype: DType,
    /// Device.
    device: Device,
    /// Next page ID.
    next_id: AtomicU32,
}

impl PagePool {
    /// Create a new page pool.
    pub fn new(
        initial_pages: usize,
        page_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype: DType,
        device: Device,
    ) -> Result<Self> {
        let mut pages = Vec::with_capacity(initial_pages);
        let mut free_list = Vec::with_capacity(initial_pages);

        for i in 0..initial_pages {
            let page = Page::with_tensor(
                PageId::new(i as u32),
                num_kv_heads,
                page_size,
                head_dim,
                dtype,
                &device,
            )?;
            pages.push(Arc::new(RwLock::new(page)));
            free_list.push(i);
        }

        Ok(Self {
            pages: RwLock::new(pages),
            free_list: RwLock::new(free_list),
            page_size,
            num_kv_heads,
            head_dim,
            dtype,
            device,
            next_id: AtomicU32::new(initial_pages as u32),
        })
    }

    /// Allocate a page from the pool.
    pub fn allocate(&self) -> Result<Arc<RwLock<Page>>> {
        // Try to get from free list
        let mut free_list = self.free_list.write();

        if let Some(idx) = free_list.pop() {
            let pages = self.pages.read();
            return Ok(pages[idx].clone());
        }

        // No free pages, allocate new one
        drop(free_list);

        let id = PageId::new(self.next_id.fetch_add(1, Ordering::SeqCst));
        let page = Page::with_tensor(
            id,
            self.num_kv_heads,
            self.page_size,
            self.head_dim,
            self.dtype,
            &self.device,
        )?;
        let page = Arc::new(RwLock::new(page));

        let mut pages = self.pages.write();
        let idx = pages.len();
        pages.push(page.clone());

        Ok(page)
    }

    /// Free a page back to the pool.
    pub fn free(&self, page: Arc<RwLock<Page>>) {
        let page_guard = page.read();
        if page_guard.dec_ref() == 0 {
            // Find and add to free list
            let pages = self.pages.read();
            for (idx, p) in pages.iter().enumerate() {
                if Arc::ptr_eq(p, &page) {
                    drop(page_guard);
                    drop(pages);
                    self.free_list.write().push(idx);
                    return;
                }
            }
        }
    }

    /// Get page size.
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Get number of allocated pages.
    pub fn num_pages(&self) -> usize {
        self.pages.read().len()
    }

    /// Get number of free pages.
    pub fn num_free(&self) -> usize {
        self.free_list.read().len()
    }
}

/// Page table for a single sequence.
///
/// Maps logical positions to physical pages.
#[derive(Debug, Clone)]
pub struct PageTable {
    /// Ordered list of pages for this sequence.
    pages: Vec<Arc<RwLock<Page>>>,
    /// Page size.
    page_size: usize,
    /// Total number of tokens.
    num_tokens: usize,
}

impl PageTable {
    /// Create an empty page table.
    pub fn new(page_size: usize) -> Self {
        Self {
            pages: Vec::new(),
            page_size,
            num_tokens: 0,
        }
    }

    /// Get the number of pages.
    pub fn num_pages(&self) -> usize {
        self.pages.len()
    }

    /// Get total tokens.
    pub fn num_tokens(&self) -> usize {
        self.num_tokens
    }

    /// Get page at index.
    pub fn get_page(&self, idx: usize) -> Option<&Arc<RwLock<Page>>> {
        self.pages.get(idx)
    }

    /// Get all pages.
    pub fn pages(&self) -> &[Arc<RwLock<Page>>] {
        &self.pages
    }

    /// Add a page.
    pub fn push_page(&mut self, page: Arc<RwLock<Page>>) {
        self.pages.push(page);
    }

    /// Fork this page table (O(1) copy-on-write).
    ///
    /// Creates a new page table sharing all pages with the parent.
    /// Pages are reference-counted, not copied.
    pub fn fork(&self) -> Self {
        // Increment ref count on all pages
        for page in &self.pages {
            page.read().inc_ref();
        }

        Self {
            pages: self.pages.clone(),
            page_size: self.page_size,
            num_tokens: self.num_tokens,
        }
    }

    /// Get page indices for FlashInfer.
    pub fn page_indices(&self) -> Vec<i32> {
        self.pages
            .iter()
            .map(|p| p.read().id().0 as i32)
            .collect()
    }

    /// Get last page length for FlashInfer.
    pub fn last_page_len(&self) -> i32 {
        if self.pages.is_empty() {
            0
        } else {
            (self.num_tokens % self.page_size) as i32
        }
    }

    /// Increment token count.
    pub fn inc_tokens(&mut self, count: usize) {
        self.num_tokens += count;
    }
}

/// Paged KV cache for a model.
///
/// Manages pages across all layers for multiple sequences.
pub struct PagedKvCache {
    /// Page pool (shared across layers).
    pool: Arc<PagePool>,
    /// Per-layer, per-sequence page tables.
    /// Outer: layers, Inner: sequence_id -> PageTable
    layer_tables: Vec<RwLock<HashMap<u64, PageTable>>>,
    /// Number of layers.
    num_layers: usize,
    /// Page size.
    page_size: usize,
    /// Next sequence ID.
    next_seq_id: AtomicU64,
}

use std::sync::atomic::AtomicU64;

impl PagedKvCache {
    /// Create a new paged KV cache.
    pub fn new(
        num_layers: usize,
        initial_pages: usize,
        page_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype: DType,
        device: Device,
    ) -> Result<Self> {
        let pool = Arc::new(PagePool::new(
            initial_pages * num_layers,
            page_size,
            num_kv_heads,
            head_dim,
            dtype,
            device,
        )?);

        let layer_tables = (0..num_layers)
            .map(|_| RwLock::new(HashMap::new()))
            .collect();

        Ok(Self {
            pool,
            layer_tables,
            num_layers,
            page_size,
            next_seq_id: AtomicU64::new(0),
        })
    }

    /// Allocate a new sequence.
    ///
    /// Returns a sequence ID for managing this sequence's KV cache.
    pub fn allocate_sequence(&self) -> u64 {
        let seq_id = self.next_seq_id.fetch_add(1, Ordering::SeqCst);

        for layer in &self.layer_tables {
            layer.write().insert(seq_id, PageTable::new(self.page_size));
        }

        seq_id
    }

    /// Fork a sequence (O(1) operation).
    ///
    /// Creates a new sequence that shares pages with the parent.
    pub fn fork_sequence(&self, parent_id: u64) -> Result<u64> {
        let child_id = self.next_seq_id.fetch_add(1, Ordering::SeqCst);

        for layer in &self.layer_tables {
            let mut tables = layer.write();
            let parent_table = tables
                .get(&parent_id)
                .ok_or_else(|| DendriteError::CacheError(format!("Unknown sequence: {}", parent_id)))?;

            let child_table = parent_table.fork();
            tables.insert(child_id, child_table);
        }

        Ok(child_id)
    }

    /// Free a sequence.
    pub fn free_sequence(&self, seq_id: u64) {
        for layer in &self.layer_tables {
            if let Some(table) = layer.write().remove(&seq_id) {
                for page in table.pages() {
                    self.pool.free(page.clone());
                }
            }
        }
    }

    /// Get the page table for a layer and sequence.
    pub fn get_page_table(&self, layer_idx: usize, seq_id: u64) -> Option<PageTable> {
        self.layer_tables
            .get(layer_idx)?
            .read()
            .get(&seq_id)
            .cloned()
    }

    /// Get sequence length.
    pub fn seq_len(&self, seq_id: u64) -> usize {
        self.layer_tables
            .first()
            .and_then(|l| l.read().get(&seq_id).map(|t| t.num_tokens()))
            .unwrap_or(0)
    }

    /// Get page size.
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Get number of layers.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Get pool statistics.
    pub fn pool_stats(&self) -> (usize, usize) {
        (self.pool.num_pages(), self.pool.num_free())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_creation() {
        let page = Page::new(PageId::new(0));
        assert_eq!(page.ref_count(), 1);
        assert!(page.is_empty());
    }

    #[test]
    fn page_ref_counting() {
        let page = Page::new(PageId::new(0));
        assert_eq!(page.ref_count(), 1);

        page.inc_ref();
        assert_eq!(page.ref_count(), 2);

        page.dec_ref();
        assert_eq!(page.ref_count(), 1);
    }

    #[test]
    fn page_pool_creation() {
        let pool = PagePool::new(
            4,
            DEFAULT_PAGE_SIZE,
            4,
            64,
            DType::F32,
            Device::Cpu,
        ).unwrap();

        assert_eq!(pool.num_pages(), 4);
        assert_eq!(pool.num_free(), 4);
    }

    #[test]
    fn page_pool_allocation() {
        let pool = PagePool::new(
            2,
            DEFAULT_PAGE_SIZE,
            4,
            64,
            DType::F32,
            Device::Cpu,
        ).unwrap();

        let p1 = pool.allocate().unwrap();
        assert_eq!(pool.num_free(), 1);

        let p2 = pool.allocate().unwrap();
        assert_eq!(pool.num_free(), 0);

        // Should allocate new page
        let p3 = pool.allocate().unwrap();
        assert_eq!(pool.num_pages(), 3);

        // Free a page
        pool.free(p1);
        assert_eq!(pool.num_free(), 1);
    }

    #[test]
    fn page_table_fork() {
        let pool = PagePool::new(
            4,
            DEFAULT_PAGE_SIZE,
            4,
            64,
            DType::F32,
            Device::Cpu,
        ).unwrap();

        let mut table1 = PageTable::new(DEFAULT_PAGE_SIZE);
        table1.push_page(pool.allocate().unwrap());
        table1.push_page(pool.allocate().unwrap());

        // Fork
        let table2 = table1.fork();

        // Both should have same pages
        assert_eq!(table1.num_pages(), 2);
        assert_eq!(table2.num_pages(), 2);

        // Pages should have ref_count = 2
        assert_eq!(table1.pages()[0].read().ref_count(), 2);
    }

    #[test]
    fn paged_cache_sequence_lifecycle() {
        let cache = PagedKvCache::new(
            2,  // layers
            4,  // initial pages
            DEFAULT_PAGE_SIZE,
            4,  // kv heads
            64, // head dim
            DType::F32,
            Device::Cpu,
        ).unwrap();

        // Allocate sequence
        let seq1 = cache.allocate_sequence();
        assert_eq!(cache.seq_len(seq1), 0);

        // Fork sequence
        let seq2 = cache.fork_sequence(seq1).unwrap();
        assert_eq!(cache.seq_len(seq2), 0);

        // Free sequences
        cache.free_sequence(seq1);
        cache.free_sequence(seq2);
    }

    #[test]
    fn turboquant_page_format_compression() {
        // Verify TurboQuant compression ratios
        let head_dim = 128;

        let full_bytes = PageFormat::Full.bytes_per_token_per_head(head_dim);
        let tq4_bytes = PageFormat::TurboQuant4Bit.bytes_per_token_per_head(head_dim);
        let tq2_bytes = PageFormat::TurboQuant2Bit.bytes_per_token_per_head(head_dim);

        // Full: 128 * 2 bytes * 2 (K+V) = 512 bytes/token/head
        assert_eq!(full_bytes, 512);
        // TQ4: (128/2 + 2) * 2 = 132 bytes/token/head
        assert_eq!(tq4_bytes, 132);
        // TQ2: (128/4 + 2) * 2 = 68 bytes/token/head
        assert_eq!(tq2_bytes, 68);

        let ratio_4bit = PageFormat::TurboQuant4Bit.compression_ratio(head_dim);
        let ratio_2bit = PageFormat::TurboQuant2Bit.compression_ratio(head_dim);

        // 4-bit: ~3.88x compression
        assert!(ratio_4bit > 3.5 && ratio_4bit < 4.5, "4-bit ratio: {ratio_4bit}");
        // 2-bit: ~7.53x compression
        assert!(ratio_2bit > 7.0 && ratio_2bit < 8.0, "2-bit ratio: {ratio_2bit}");
    }

    #[test]
    fn turboquant_page_creation() {
        let page = Page::with_turboquant_4bit(
            PageId::new(0), 8, DEFAULT_PAGE_SIZE, 128, &Device::Cpu,
        ).unwrap();

        assert_eq!(page.format(), PageFormat::TurboQuant4Bit);
        assert!(page.norms().is_some());
        assert!(page.is_empty());

        // Data should be uint8 with halved last dim
        let data = page.data().unwrap();
        let shape = data.dims();
        assert_eq!(shape, &[2, 8, DEFAULT_PAGE_SIZE, 64]); // 128/2 = 64 packed
    }

    #[test]
    fn turboquant_memory_savings() {
        // Compare memory: 1000 pages, 32 layers, 8 heads, dim=128
        let num_pages = 1000;
        let head_dim = 128;
        let page_size = DEFAULT_PAGE_SIZE;
        let num_heads = 8;

        let full_mem = num_pages * page_size * num_heads * PageFormat::Full.bytes_per_token_per_head(head_dim);
        let tq4_mem = num_pages * page_size * num_heads * PageFormat::TurboQuant4Bit.bytes_per_token_per_head(head_dim);

        let savings_gb = (full_mem - tq4_mem) as f64 / 1e9;
        let ratio = full_mem as f64 / tq4_mem as f64;

        // With 1000 pages * 16 tokens = 16K tokens
        // Full: ~64MB, TQ4: ~16MB per layer
        assert!(ratio > 3.5, "Expected >3.5x savings, got {ratio}x");
        assert!(savings_gb > 0.0, "Expected positive savings");
    }
}
