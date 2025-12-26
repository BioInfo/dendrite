//! Text inference example with tokenizer integration.
//!
//! Loads TinyLlama-1.1B and generates text from a text prompt.
//!
//! Run with:
//! ```bash
//! cargo run -p dendrite-core --example text_inference -- /path/to/tinyllama
//! ```

use candle_core::Device;
use dendrite_core::attention::ReferenceBackend;
use dendrite_core::model::{ModelConfig, Tokenizer, Transformer};
use std::io::{self, Write};
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

    println!("Text Inference Example");
    println!("======================\n");

    // Load tokenizer
    println!("Loading tokenizer...");
    let tokenizer = Tokenizer::from_dir(model_path)?;
    println!("  Vocab size: {}", tokenizer.vocab_size());
    println!("  BOS token: {:?}", tokenizer.bos_token_id());
    println!("  EOS token: {:?}", tokenizer.eos_token_id());

    // Load config
    let config_path = model_path.join("config.json");
    let config = ModelConfig::from_file(&config_path)?;
    println!("\nModel: {} layers, {} hidden", config.num_hidden_layers, config.hidden_size);

    // Create device and backend
    #[cfg(feature = "cuda")]
    let (device, backend): (Device, Arc<dyn dendrite_core::attention::AttentionBackend>) = {
        println!("\nUsing CUDA");
        let device = Device::new_cuda(0)?;
        let backend = Arc::new(FlashAttnBackend::new(0)?);
        (device, backend)
    };

    #[cfg(not(feature = "cuda"))]
    let (device, backend): (Device, Arc<dyn dendrite_core::attention::AttentionBackend>) = {
        println!("\nUsing CPU");
        let device = Device::Cpu;
        let backend = Arc::new(ReferenceBackend::new());
        (device, backend)
    };

    // Create and load transformer
    println!("Loading model weights...");
    let load_start = Instant::now();
    let mut transformer = Transformer::new(config.clone(), backend, device)?;
    transformer.load_weights(model_path)?;
    println!("Loaded in {:.2}s", load_start.elapsed().as_secs_f64());

    // Interactive generation
    println!("\n--- Text Generation ---");
    println!("Enter prompts (empty line to quit):\n");

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("> ");
        stdout.flush()?;

        let mut prompt = String::new();
        stdin.read_line(&mut prompt)?;
        let prompt = prompt.trim();

        if prompt.is_empty() {
            break;
        }

        // Show tokenization
        let tokens = tokenizer.encode(prompt, true)?;
        println!("\nTokens: {:?}", tokens);
        println!("({} tokens)", tokens.len());

        // Generate with streaming output
        print!("\nGenerated: ");
        stdout.flush()?;

        let gen_start = Instant::now();
        let mut token_count = 0;

        let runtime = tokio::runtime::Runtime::new()?;
        let result = runtime.block_on(async {
            transformer
                .generate_streaming(
                    &tokens,
                    50,  // max tokens
                    0.0, // greedy
                    tokenizer.eos_token_id(),
                    |token| {
                        token_count += 1;
                        if let Ok(text) = tokenizer.decode(&[token], true) {
                            print!("{}", text);
                            let _ = stdout.flush();
                        }
                        true // continue
                    },
                )
                .await
        })?;

        let gen_time = gen_start.elapsed();
        let new_tokens = result.len() - tokens.len();
        let tokens_per_sec = new_tokens as f64 / gen_time.as_secs_f64();

        println!("\n");
        println!(
            "Generated {} tokens in {:.2}ms ({:.1} tok/s)",
            new_tokens,
            gen_time.as_secs_f64() * 1000.0,
            tokens_per_sec
        );

        // Show full decoded text
        let full_text = tokenizer.decode(&result, true)?;
        println!("\nFull output: {}", full_text);
        println!();
    }

    println!("Goodbye!");
    Ok(())
}
