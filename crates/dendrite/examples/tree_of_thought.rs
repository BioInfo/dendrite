//! Tree of Thought example.
//!
//! Demonstrates using Dendrite's O(1) fork for tree-structured reasoning.

use anyhow::Result;
use dendrite::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Build engine
    let engine = Engine::builder()
        .max_seq_len(4096)
        .max_batch_size(32)
        .build()
        .await?;

    println!("Engine initialized");
    println!("Cache stats: {:?}", engine.cache_stats());

    // Example: Simple tree search structure
    // In a real scenario, you would:
    // 1. Generate multiple continuations at each node
    // 2. Score each continuation
    // 3. Prune and expand the best branches
    // 4. Continue until a stopping condition

    let prompt = "Let's solve this step by step:";

    // Generate with tree search
    let result = engine
        .generate(prompt)
        .max_tokens(256)
        .temperature(0.8)
        .with_tree_search(TreeSearchConfig {
            num_branches: 3,
            max_depth: 5,
            scoring: ScoringMethod::LogProb,
        })
        .execute()
        .await?;

    println!("Generated {} tokens", result.num_generated_tokens);
    println!("Result: {}", result.text);

    Ok(())
}
