//! GPU inference example with real model weights and proper KV caching.
//!
//! Loads TinyLlama-1.1B and runs inference on GPU with KV cache.
//!
//! Run with:
//! ```bash
//! cargo run -p dendrite-core --features cuda --example gpu_inference -- /path/to/tinyllama
//! ```

use candle_core::Device;
use dendrite_core::attention::ReferenceBackend;
use dendrite_core::model::{ModelConfig, Transformer};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

#[cfg(feature = "cuda")]
use dendrite_core::attention::FlashAttnBackend;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model_dir = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("/home/bioinfo/models/tinyllama-1.1b");
    let model_path = Path::new(model_dir);

    println!("GPU Inference Example with KV Cache");
    println!("====================================\n");
    println!("Model directory: {}", model_path.display());

    // Load config
    let config_path = model_path.join("config.json");
    let config = ModelConfig::from_file(&config_path)?;
    println!("\nModel config:");
    println!("  Vocab size: {}", config.vocab_size);
    println!("  Hidden size: {}", config.hidden_size);
    println!("  Layers: {}", config.num_hidden_layers);
    println!(
        "  Heads: {} query, {} KV (GQA {}x)",
        config.num_attention_heads,
        config.num_key_value_heads,
        config.gqa_ratio()
    );
    println!("  Head dim: {}", config.head_dim());

    // Create device and backend
    #[cfg(feature = "cuda")]
    let (device, backend): (Device, Arc<dyn dendrite_core::attention::AttentionBackend>) = {
        println!("\nUsing CUDA device 0");
        let device = Device::new_cuda(0)?;
        let backend = Arc::new(FlashAttnBackend::new(0)?);
        (device, backend)
    };

    #[cfg(not(feature = "cuda"))]
    let (device, backend): (Device, Arc<dyn dendrite_core::attention::AttentionBackend>) = {
        println!("\nUsing CPU (no CUDA)");
        let device = Device::Cpu;
        let backend = Arc::new(ReferenceBackend::new());
        (device, backend)
    };

    // Create transformer
    println!("\nCreating transformer...");
    let mut transformer = Transformer::new(config.clone(), backend, device.clone())?;

    // Load weights
    println!("Loading weights...");
    let load_start = Instant::now();
    transformer.load_weights(model_path)?;
    let load_time = load_start.elapsed();
    println!(
        "Loaded {} layers in {:.2}s",
        transformer.num_layers(),
        load_time.as_secs_f64()
    );

    // Create prompt tokens (TinyLlama format)
    // <s>Hello, my name is
    let prompt_tokens: Vec<u32> = vec![1, 15043, 29892, 590, 1024, 338];
    println!("\nPrompt tokens: {:?}", prompt_tokens);
    println!("(Approximate: '<s>Hello, my name is')");

    // Generate with proper KV caching
    println!("\nGenerating with KV cache...");
    let gen_start = Instant::now();
    let generated = tokio::runtime::Runtime::new()?.block_on(async {
        transformer.generate(&prompt_tokens, 20, 0.0).await
    })?;
    let gen_time = gen_start.elapsed();

    let num_new_tokens = generated.len() - prompt_tokens.len();
    let tokens_per_sec = num_new_tokens as f64 / gen_time.as_secs_f64();

    println!("\nGeneration complete!");
    println!("  Generated {} new tokens in {:.2}ms", num_new_tokens, gen_time.as_secs_f64() * 1000.0);
    println!("  Throughput: {:.1} tokens/s", tokens_per_sec);
    println!("\nGenerated token IDs: {:?}", generated);

    // Now let's benchmark prefill + decode separately
    println!("\n--- Detailed Timing ---");

    // Fresh cache for benchmarking
    let mut cache = transformer.create_cache();

    // Benchmark prefill
    let prefill_input =
        candle_core::Tensor::from_slice(&prompt_tokens, (1, prompt_tokens.len()), &device)?;
    let prefill_start = Instant::now();
    let logits = tokio::runtime::Runtime::new()?
        .block_on(transformer.forward_with_cache(&prefill_input, &mut cache))?;
    let prefill_time = prefill_start.elapsed();
    println!(
        "Prefill {} tokens: {:.2}ms",
        prompt_tokens.len(),
        prefill_time.as_secs_f64() * 1000.0
    );

    // Get first generated token
    let next_token = transformer.sample(&logits, 0.0)?;
    println!("First generated token: {}", next_token);
    println!("Cache size after prefill: {} tokens", cache.seq_len());

    // Benchmark decode steps
    println!("\nDecode timing (with KV cache):");
    let mut decode_times = Vec::new();
    let mut current_token = next_token;

    for i in 0..5 {
        let input = candle_core::Tensor::from_slice(&[current_token], (1, 1), &device)?;
        let decode_start = Instant::now();
        let logits = tokio::runtime::Runtime::new()?
            .block_on(transformer.forward_with_cache(&input, &mut cache))?;
        let decode_time = decode_start.elapsed();
        decode_times.push(decode_time.as_secs_f64() * 1000.0);

        current_token = transformer.sample(&logits, 0.0)?;
        println!(
            "  Step {}: token {} in {:.2}ms (cache: {} tokens)",
            i + 1,
            current_token,
            decode_time.as_secs_f64() * 1000.0,
            cache.seq_len()
        );
    }

    let avg_decode: f64 = decode_times.iter().sum::<f64>() / decode_times.len() as f64;
    println!("\nAverage decode time: {:.2}ms", avg_decode);
    println!("Final cache size: {} tokens", cache.seq_len());

    println!("\nGPU inference with KV cache complete!");

    Ok(())
}
