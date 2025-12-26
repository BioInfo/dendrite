//! Beam Search example.
//!
//! Demonstrates using beam search for sequence generation.
//! This example simulates text generation with mock logits.

use dendrite_core::search::{BeamConfig, BeamSearch};
use dendrite_core::tree::NodeId;

/// Mock vocabulary for demonstration.
const VOCAB: &[&str] = &[
    "<pad>", "<eos>", "the", "a", "cat", "dog", "sat", "ran", "on", "mat", "quick", "lazy",
];

/// Simulates getting next token logits (unnormalized log probabilities).
fn mock_next_token_logits(sequence: &[u32]) -> Vec<f32> {
    // Simple mock language model:
    // - After "the", prefer "cat" or "dog"
    // - After "cat" or "dog", prefer "sat" or "ran"
    // - After "sat" or "ran", prefer "on"
    // - After "on", prefer "the" or "mat" or end
    // - Short sequences get an EOS probability boost

    let last_token = sequence.last().copied().unwrap_or(0);

    // Create logits for all 12 tokens, default to low values
    let mut logits = vec![-10.0f32; 12];

    match last_token {
        2 | 3 => {
            // After "the" or "a" -> prefer nouns
            logits[4] = 2.0; // cat
            logits[5] = 1.5; // dog
            logits[10] = 0.5; // quick
            logits[11] = 0.0; // lazy
            logits[1] = -2.0; // eos
        }
        4 | 5 => {
            // After "cat" or "dog" -> prefer verbs
            logits[6] = 2.5; // sat
            logits[7] = 2.0; // ran
            logits[1] = if sequence.len() > 3 { 0.0 } else { -5.0 }; // eos
            logits[8] = 0.0; // on
        }
        6 | 7 => {
            // After "sat" or "ran" -> prefer "on"
            logits[8] = 3.0; // on
            logits[1] = if sequence.len() > 4 { 0.5 } else { -3.0 }; // eos
            logits[2] = 0.0; // the
        }
        8 => {
            // After "on" -> prefer "the", "mat", or end
            logits[2] = 1.5; // the
            logits[9] = 2.0; // mat
            logits[3] = 1.0; // a
            logits[1] = if sequence.len() > 4 { 1.0 } else { -2.0 }; // eos
        }
        9 => {
            // After "mat" -> likely end
            logits[1] = 3.0; // eos
            logits[2] = 0.5; // the
            logits[8] = -1.0; // on
        }
        10 | 11 => {
            // After adjective -> prefer noun
            logits[4] = 2.0; // cat
            logits[5] = 2.0; // dog
            logits[1] = -2.0; // eos
        }
        _ => {
            // Default: start with "the" or "a"
            logits[2] = 3.0; // the
            logits[3] = 2.0; // a
            logits[10] = 1.0; // quick
            logits[11] = 0.5; // lazy
        }
    }

    logits
}

/// Decode tokens to text.
fn decode(tokens: &[u32]) -> String {
    tokens
        .iter()
        .filter_map(|&t| VOCAB.get(t as usize).copied())
        .collect::<Vec<_>>()
        .join(" ")
}

fn main() {
    println!("=== Beam Search Demo ===\n");

    // Create beam search configuration
    let config = BeamConfig {
        beam_width: 4,
        max_length: 10,
        min_length: 3,
        length_alpha: 0.6, // Slight preference for longer sequences
        eos_token_id: Some(1),
        num_return: 4,
        early_stopping: true,
    };

    println!("Configuration:");
    println!("  Beam width: {}", config.beam_width);
    println!("  Max length: {}", config.max_length);
    println!("  Length penalty (alpha): {}", config.length_alpha);
    println!("  EOS token: {:?}", config.eos_token_id);
    println!();

    let mut beam = BeamSearch::new(NodeId::ROOT, config.clone());

    println!("Step-by-step generation:\n");

    // Run beam search steps
    for step in 0..8 {
        println!("Step {}:", step + 1);

        // Get current candidates
        let candidates = beam.candidates();
        if candidates.is_empty() {
            println!("  All candidates finished");
            break;
        }

        // Show current candidates
        for candidate in candidates {
            let text = decode(&candidate.tokens);
            let norm_score = candidate.normalized_score(config.length_alpha);
            println!(
                "  [{}] \"{}\" (score: {:.3}, norm: {:.3})",
                candidate.id,
                text,
                candidate.score,
                norm_score
            );
        }

        // Check if done
        if beam.is_done() {
            println!("\n  Search complete!");
            break;
        }

        // Expand each candidate
        let candidate_ids: Vec<usize> = candidates.iter().map(|c| c.id).collect();
        let candidate_tokens: Vec<Vec<u32>> = candidates.iter().map(|c| c.tokens.clone()).collect();

        for (cid, tokens) in candidate_ids.iter().zip(candidate_tokens.iter()) {
            // Get next token logits
            let logits = mock_next_token_logits(tokens);

            // Expand with top-k tokens
            beam.expand_with_logits(*cid, &logits, 3);
        }

        // Complete the step (select top candidates)
        beam.step();

        println!();
    }

    // Get final results
    println!("\n=== Final Results ===\n");

    // Best sequence
    if let Some(best) = beam.best_sequence() {
        let text = decode(&best.tokens);
        println!("Best sequence: \"{}\"", text);
        println!("  Raw score: {:.3}", best.score);
        println!(
            "  Normalized score: {:.3}",
            best.normalized_score(config.length_alpha)
        );
        println!("  Length: {} tokens", best.tokens.len());
        let finished = if best.is_finished { "yes" } else { "no" };
        println!("  Finished: {}", finished);
    }

    // All finished sequences
    let finished = beam.finished();
    if !finished.is_empty() {
        println!("\nAll finished sequences ({}):", finished.len());
        for (i, seq) in finished.iter().enumerate() {
            let text = decode(&seq.tokens);
            println!(
                "  {}: \"{}\" (norm score: {:.3})",
                i + 1,
                text,
                seq.normalized_score(config.length_alpha)
            );
        }
    }

    // N-best list (from n_best method)
    println!("\nN-best list (top 4):");
    let nbest = beam.n_best(4);
    for (i, seq) in nbest.iter().enumerate() {
        let text = decode(&seq.tokens);
        let finished = if seq.is_finished { "✓" } else { "..." };
        println!(
            "  {}: \"{}\" {} (score: {:.3})",
            i + 1,
            text,
            finished,
            seq.normalized_score(config.length_alpha)
        );
    }

    println!("\n=== Beam Search Demo Complete ===");
}
