//! Golden token validation against HuggingFace reference.
//!
//! This example validates Dendrite's token generation against
//! pre-computed golden sequences from HuggingFace Transformers.
//!
//! Prerequisites:
//! 1. Generate golden cases:
//!    python scripts/generate_golden.py --model /path/to/tinyllama --output golden_cases.json
//!
//! 2. Run validation:
//!    cargo run -p dendrite-core --example golden_validation -- /path/to/tinyllama golden_cases.json

use candle_core::Device;
use dendrite_core::attention::ReferenceBackend;
use dendrite_core::model::{
    GoldenCase, GoldenTestHarness, ModelConfig, Tokenizer, Transformer,
};
use serde::Deserialize;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

#[cfg(feature = "cuda")]
use dendrite_core::attention::FlashAttnBackend;

/// Golden case as stored in JSON.
#[derive(Debug, Deserialize)]
struct JsonGoldenCase {
    name: String,
    prompt: String,
    input_tokens: Vec<u32>,
    expected_tokens: Vec<u32>,
    max_tokens: usize,
    #[allow(dead_code)]
    temperature: f32,
}

impl From<JsonGoldenCase> for GoldenCase {
    fn from(json: JsonGoldenCase) -> Self {
        GoldenCase::greedy(json.name, json.input_tokens, json.expected_tokens)
            .with_max_tokens(json.max_tokens)
            .with_description(json.prompt)
    }
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: golden_validation <model_dir> <golden_cases.json>");
        eprintln!("\nGenerate golden cases with:");
        eprintln!("  python scripts/generate_golden.py --model <model_dir> --output golden_cases.json");
        std::process::exit(1);
    }

    let model_dir = &args[1];
    let golden_path = &args[2];
    let model_path = Path::new(model_dir);

    println!("Golden Token Validation");
    println!("=======================\n");

    // Load golden cases from JSON
    println!("Loading golden cases from {}...", golden_path);
    let json_content = fs::read_to_string(golden_path)?;
    let json_cases: Vec<JsonGoldenCase> = serde_json::from_str(&json_content)?;
    println!("Loaded {} test cases", json_cases.len());

    // Convert to GoldenCase
    let mut harness = GoldenTestHarness::new();
    for json_case in json_cases {
        println!(
            "  - {}: {} input tokens, {} expected tokens",
            json_case.name,
            json_case.input_tokens.len(),
            json_case.expected_tokens.len()
        );
        harness.add_case(json_case.into());
    }

    // Load tokenizer (for display only)
    println!("\nLoading tokenizer...");
    let tokenizer = Tokenizer::from_dir(model_path)?;

    // Load model
    println!("Loading model config...");
    let config = ModelConfig::from_file(&model_path.join("config.json"))?;
    println!(
        "  Model: {} layers, {} hidden, {} vocab",
        config.num_hidden_layers, config.hidden_size, config.vocab_size
    );

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
        println!("\nUsing CPU (for deterministic validation)");
        let device = Device::Cpu;
        let backend = Arc::new(ReferenceBackend::new());
        (device, backend)
    };

    // Load weights
    println!("Loading model weights...");
    let load_start = Instant::now();
    let mut transformer = Transformer::new(config, backend, device)?;
    transformer.load_weights(model_path)?;
    println!("Loaded in {:.2}s\n", load_start.elapsed().as_secs_f64());

    // Run golden tests
    println!("Running golden validation...\n");

    let runtime = tokio::runtime::Runtime::new()?;
    let summary = harness.run_with(|input_tokens, max_tokens| {
        runtime
            .block_on(transformer.generate(input_tokens, max_tokens, 0.0))
            .map(|full| {
                // Return only generated tokens (not prompt)
                full[input_tokens.len()..].to_vec()
            })
            .map_err(|e| e.to_string())
    });

    // Display results
    println!("{}", summary);

    // Detailed analysis for failures
    if !summary.all_passed() {
        println!("\n--- Detailed Failure Analysis ---\n");

        for result in summary.results.values() {
            if !result.passed {
                println!("Case: {}", result.name);

                if let Some(ref err) = result.error {
                    println!("  Error: {}", err);
                }

                if let Some(idx) = result.first_divergence {
                    println!("  First divergence at position {}", idx);

                    let expected_token = result.expected.get(idx).copied();
                    let actual_token = result.actual.get(idx).copied();

                    if let Some(exp) = expected_token {
                        let exp_str = tokenizer.decode(&[exp], false).unwrap_or_default();
                        println!("  Expected: {} ({:?})", exp, exp_str);
                    }

                    if let Some(act) = actual_token {
                        let act_str = tokenizer.decode(&[act], false).unwrap_or_default();
                        println!("  Actual:   {} ({:?})", act, act_str);
                    }
                }

                // Show first few tokens of each
                println!(
                    "  Expected tokens: {:?}",
                    &result.expected[..result.expected.len().min(10)]
                );
                println!(
                    "  Actual tokens:   {:?}",
                    &result.actual[..result.actual.len().min(10)]
                );
                println!();
            }
        }
    }

    // Exit with error code if tests failed
    if summary.all_passed() {
        println!("\nAll golden tests passed!");
        Ok(())
    } else {
        eprintln!(
            "\n{} of {} tests failed",
            summary.failed, summary.total
        );
        std::process::exit(1);
    }
}
