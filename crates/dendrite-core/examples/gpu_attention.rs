//! GPU attention example.
//!
//! Demonstrates FlashAttnBackend for GPU-accelerated attention.
//!
//! Run with:
//! ```bash
//! cargo run -p dendrite-core --features cuda --example gpu_attention
//! ```

use candle_core::{DType, Device, Tensor};
use dendrite_core::attention::{AttentionBackend, AttentionConfig};
use dendrite_core::cache::BlockTable;
use std::time::Instant;

#[cfg(feature = "cuda")]
use dendrite_core::attention::FlashAttnBackend;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    #[cfg(not(feature = "cuda"))]
    {
        println!("This example requires the 'cuda' feature.");
        println!("Run with: cargo run -p dendrite-core --features cuda --example gpu_attention");
        return Ok(());
    }

    #[cfg(feature = "cuda")]
    {
        println!("GPU Attention Example");
        println!("=====================\n");

        // Create GPU backend
        let backend = FlashAttnBackend::new(0)?;
        println!("Created FlashAttnBackend on device 0");

        // Configuration: Llama-3-8B style
        let config = AttentionConfig::new(
            32,  // num_heads (32 query heads)
            8,   // num_kv_heads (8 KV heads for GQA)
            128, // head_dim
        );
        println!("Config: {} query heads, {} KV heads, {} head dim",
            config.num_heads, config.num_kv_heads, config.head_dim);
        println!("GQA ratio: {}x\n", config.num_queries_per_kv());

        // Test prefill
        println!("Testing prefill...");
        let batch_size = 1;
        let seq_len = 512;

        let device = Device::new_cuda(0)?;

        // Flash attention requires f16 or bf16
        let query = Tensor::randn(
            0f32, 1.0,
            &[batch_size, config.num_heads, seq_len, config.head_dim],
            &device,
        )?.to_dtype(DType::F16)?;
        let key = Tensor::randn(
            0f32, 1.0,
            &[batch_size, config.num_kv_heads, seq_len, config.head_dim],
            &device,
        )?.to_dtype(DType::F16)?;
        let value = Tensor::randn(
            0f32, 1.0,
            &[batch_size, config.num_kv_heads, seq_len, config.head_dim],
            &device,
        )?.to_dtype(DType::F16)?;
        let block_table = BlockTable::new(16);

        // Warm-up
        let _ = backend.prefill(&query, &key, &value, &block_table, &config).await?;

        // Benchmark
        let iterations = 10;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = backend.prefill(&query, &key, &value, &block_table, &config).await?;
        }
        let elapsed = start.elapsed();
        let avg_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;

        println!("  Sequence length: {}", seq_len);
        println!("  Average prefill time: {:.3} ms", avg_ms);
        println!("  Throughput: {:.1} tokens/s\n", seq_len as f64 / (avg_ms / 1000.0));

        // Test decode
        println!("Testing decode...");
        let decode_query = Tensor::randn(
            0f32, 1.0,
            &[batch_size, config.num_heads, 1, config.head_dim],
            &device,
        )?.to_dtype(DType::F16)?;

        // Store some KV in cache for decode
        use dendrite_core::cache::BlockId;
        let mut block_table = BlockTable::new(16);
        block_table.push(BlockId(0));
        backend.append_kv(&key, &value, &block_table, 0).await?;

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = backend.decode(&decode_query, &block_table, &config).await?;
        }
        let elapsed = start.elapsed();
        let avg_us = elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64;

        println!("  Context length: {}", seq_len);
        println!("  Average decode time: {:.1} us\n", avg_us);

        println!("GPU attention working!");
        Ok(())
    }
}
