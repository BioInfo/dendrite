//! GPU inference example with real model weights.
//!
//! Loads TinyLlama-1.1B and runs inference on GPU.
//!
//! Run with:
//! ```bash
//! cargo run -p dendrite-core --features cuda --example gpu_inference -- /path/to/tinyllama
//! ```

use candle_core::Device;
use dendrite_core::attention::ReferenceBackend;
use dendrite_core::cache::BlockTable;
use dendrite_core::model::{ModelConfig, Transformer};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

#[cfg(feature = "cuda")]
use dendrite_core::attention::FlashAttnBackend;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model_dir = args.get(1).map(|s| s.as_str()).unwrap_or("/home/bioinfo/models/tinyllama-1.1b");
    let model_path = Path::new(model_dir);

    println!("GPU Inference Example");
    println!("=====================\n");
    println!("Model directory: {}", model_path.display());

    // Load config
    let config_path = model_path.join("config.json");
    let config = ModelConfig::from_file(&config_path)?;
    println!("\nModel config:");
    println!("  Vocab size: {}", config.vocab_size);
    println!("  Hidden size: {}", config.hidden_size);
    println!("  Layers: {}", config.num_hidden_layers);
    println!("  Heads: {} query, {} KV (GQA {}x)",
        config.num_attention_heads,
        config.num_key_value_heads,
        config.gqa_ratio());
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
    println!("Loaded {} layers in {:.2}s", transformer.num_layers(), load_time.as_secs_f64());

    // Create input tokens
    let prompt_tokens: Vec<u32> = vec![1, 15043, 29892, 590, 1024, 338]; // "Hello, my name is"
    println!("\nPrompt tokens: {:?}", prompt_tokens);

    // Run prefill
    println!("\nRunning prefill...");
    let input = candle_core::Tensor::from_slice(
        &prompt_tokens,
        (1, prompt_tokens.len()),
        &device,
    )?;

    let block_table = BlockTable::new(16);

    let prefill_start = Instant::now();
    let logits = tokio::runtime::Runtime::new()?.block_on(
        transformer.prefill(&input, &block_table)
    )?;
    let prefill_time = prefill_start.elapsed();

    println!("Prefill time: {:.2}ms", prefill_time.as_secs_f64() * 1000.0);
    println!("Output shape: {:?}", logits.dims());

    // Sample next token
    let next_token = transformer.sample(&logits, 0.0)?;
    println!("Next token (greedy): {}", next_token);

    // Generate a few more tokens
    println!("\nGenerating tokens...");
    let mut generated = prompt_tokens.clone();
    generated.push(next_token);

    for i in 0..10 {
        let last_token = *generated.last().unwrap();
        let input = candle_core::Tensor::from_slice(&[last_token], (1, 1), &device)?;

        let decode_start = Instant::now();
        let logits = tokio::runtime::Runtime::new()?.block_on(
            transformer.decode(&input, &block_table, generated.len() - 1)
        )?;
        let decode_time = decode_start.elapsed();

        let next = transformer.sample(&logits, 0.7)?;
        generated.push(next);

        println!("  Token {}: {} ({:.2}ms)", i + 1, next, decode_time.as_secs_f64() * 1000.0);
    }

    println!("\nGenerated sequence: {:?}", generated);
    println!("\nGPU inference complete!");

    Ok(())
}
