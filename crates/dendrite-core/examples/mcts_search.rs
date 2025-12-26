//! MCTS (Monte Carlo Tree Search) example.
//!
//! Demonstrates using MCTS for tree search with UCT scoring.
//! This example simulates a simple decision-making scenario where
//! each action has a stochastic reward.

use dendrite_core::search::{ExpansionResult, MctsConfig, MctsSearch};
use dendrite_core::tree::NodeId;
use rand::Rng;

/// Simulates a simple environment where actions have stochastic rewards.
struct SimpleEnvironment {
    /// Reward distribution mean for each action.
    action_means: Vec<f64>,
    /// Noise level.
    noise: f64,
}

impl SimpleEnvironment {
    fn new() -> Self {
        Self {
            // Action 0 = medium, Action 1 = low, Action 2 = high variance but best mean
            action_means: vec![0.5, 0.3, 0.7],
            noise: 0.2,
        }
    }

    /// Simulate taking an action and get a noisy reward.
    fn step(&self, action: u32, rng: &mut impl Rng) -> f64 {
        let mean = self.action_means.get(action as usize).copied().unwrap_or(0.0);
        let noise = (rng.gen::<f64>() - 0.5) * 2.0 * self.noise;
        (mean + noise).clamp(0.0, 1.0)
    }

    /// Get mock logits for expansion (higher value = more likely action).
    fn get_logits(&self) -> Vec<f32> {
        // Prior probabilities roughly match true values
        vec![0.0, -1.0, 0.5] // Slightly favor action 2
    }
}

fn main() {
    println!("=== MCTS Search Demo ===\n");

    let mut rng = rand::thread_rng();
    let env = SimpleEnvironment::new();

    // Create MCTS with UCT scoring
    let config = MctsConfig {
        max_iterations: 500,
        max_depth: 10,
        exploration_constant: 1.4, // Standard UCT exploration
        num_parallel: 1,
        reuse_tree: true,
        discount: 0.99,
    };

    let mut mcts = MctsSearch::new(NodeId::ROOT, config);

    println!("Environment: 3 actions with different reward distributions");
    println!("  Action 0: mean=0.5, noise=±0.2");
    println!("  Action 1: mean=0.3, noise=±0.2");
    println!("  Action 2: mean=0.7, noise=±0.2");
    println!("\nRunning {} MCTS iterations...\n", 500);

    // Run MCTS iterations
    for iteration in 0..500 {
        // 1. Select a leaf node
        let selected = mcts.select();

        // 2. Expand if not terminal
        let node = mcts.get_node(selected).unwrap();
        if !node.stats.is_terminal && node.children.is_empty() {
            let logits = env.get_logits();

            // Create expansion from logits
            let expansion = ExpansionResult {
                children: Vec::new(),
                actions: vec![0, 1, 2],
                log_probs: Some(logits.iter().copied().collect()),
                priors: Some(vec![0.33, 0.33, 0.34]),
            };

            let children = mcts.expand(selected, &expansion);

            // Pick first child for simulation
            if !children.is_empty() {
                let child = mcts.get_node(children[0]).unwrap();
                let action = child.action.unwrap_or(0);
                let reward = env.step(action, &mut rng);
                mcts.backpropagate(children[0], reward);
            }
        } else if !node.children.is_empty() {
            // Already expanded - simulate from this node
            let action = node.action.unwrap_or(0);
            let reward = env.step(action, &mut rng);
            mcts.backpropagate(selected, reward);
        }

        // Print progress
        if (iteration + 1) % 100 == 0 {
            let stats = mcts.stats();
            println!(
                "Iteration {}: {} nodes, best value = {:.3}",
                iteration + 1,
                stats.nodes_created,
                stats.best_value
            );
        }
    }

    // Get results
    println!("\n=== Results ===\n");

    let best_action = mcts.best_action();
    println!("Best action (by visits): {:?}", best_action);

    // Print action statistics
    let root = mcts.root();
    println!("\nAction statistics from root:");
    for &child_idx in &root.children {
        let child = mcts.get_node(child_idx).unwrap();
        println!(
            "  Action {:?}: {} visits, mean reward = {:.3}",
            child.action, child.stats.visits, child.stats.mean_reward
        );
    }

    let probs = mcts.action_probabilities();
    println!("\nAction probabilities:");
    for (action, prob) in probs {
        println!("  Action {}: {:.1}%", action, prob * 100.0);
    }

    println!("\nSearch statistics:");
    let stats = mcts.stats();
    println!("  Total nodes created: {}", stats.nodes_created);
    println!("  Max depth reached: {}", stats.max_depth_reached);
    println!("  Iterations: {}", stats.iterations);
    println!("  Best value found: {:.3}", stats.best_value);

    // Principal variation (best path from root)
    println!("\nPrincipal variation (best actions from root):");
    let pv = mcts.principal_variation();
    for (depth, action) in pv.iter().enumerate() {
        println!("  Depth {}: Action {}", depth + 1, action);
    }

    println!("\n=== MCTS Demo Complete ===");
}
