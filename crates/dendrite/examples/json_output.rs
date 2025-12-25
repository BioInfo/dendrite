//! JSON output example.
//!
//! Demonstrates using grammar constraints for structured output.

use anyhow::Result;
use dendrite::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Build engine
    let engine = Engine::builder()
        .max_seq_len(2048)
        .max_batch_size(16)
        .build()
        .await?;

    println!("Engine initialized");

    // Define JSON schema for output
    let schema = r#"{
        "type": "object",
        "properties": {
            "name": { "type": "string" },
            "age": { "type": "integer" },
            "hobbies": {
                "type": "array",
                "items": { "type": "string" }
            }
        },
        "required": ["name", "age", "hobbies"]
    }"#;

    // Create grammar constraint
    let grammar = Grammar::json_schema(schema);
    let constraint = GrammarConstraint::new(grammar, 128000)?;

    println!("Grammar constraint created");
    println!("Vocab size: {}", constraint.vocab_size());

    // In a real scenario, you would:
    // 1. At each decode step, call constraint.compute_mask()
    // 2. Apply mask to logits before sampling
    // 3. Call constraint.accept_token() with sampled token
    // 4. Continue until constraint.is_complete()

    let prompt = "Generate a person's profile as JSON:";

    let result = engine
        .generate(prompt)
        .max_tokens(128)
        .temperature(0.0) // Greedy for structured output
        .execute()
        .await?;

    println!("Result: {}", result.text);

    Ok(())
}
