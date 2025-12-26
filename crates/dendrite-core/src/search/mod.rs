//! Tree search algorithms for inference.
//!
//! This module provides tree search capabilities for branching inference:
//! - **MCTS** (Monte Carlo Tree Search) with UCT selection
//! - **Beam Search** with length normalization and early stopping
//! - **Scorers** for branch evaluation (UCT, Greedy, PUCT)
//! - **Expanders** for generating child nodes
//!
//! # Architecture
//!
//! The search system is built on modular components:
//!
//! | Component | Purpose |
//! |-----------|---------|
//! | [`Scorer`] | Evaluates branches (UCT, Greedy, PUCT) |
//! | [`Expander`] | Generates child nodes from logits |
//! | [`MctsSearch`] | Monte Carlo Tree Search with backpropagation |
//! | [`BeamSearch`] | Beam search with configurable width |
//!
//! # Example: MCTS
//!
//! ```
//! use dendrite_core::search::{MctsSearch, MctsConfig, ExpansionResult};
//! use dendrite_core::tree::NodeId;
//!
//! // Create MCTS with default config
//! let config = MctsConfig::default();
//! let mut mcts = MctsSearch::new(NodeId::ROOT, config);
//!
//! // Select a node, expand it, and backpropagate
//! let selected = mcts.select();
//!
//! // Expand with candidate actions
//! let expansion = ExpansionResult {
//!     children: Vec::new(),
//!     actions: vec![1, 2, 3],
//!     log_probs: None,
//!     priors: Some(vec![0.5, 0.3, 0.2]),
//! };
//! let children = mcts.expand(selected, &expansion);
//!
//! // Backpropagate a reward
//! if !children.is_empty() {
//!     mcts.backpropagate(children[0], 0.8);
//! }
//!
//! // Get best action by visit count
//! let best = mcts.best_action();
//! ```
//!
//! # Example: Beam Search
//!
//! ```
//! use dendrite_core::search::{BeamSearch, BeamConfig};
//! use dendrite_core::tree::NodeId;
//!
//! // Create beam search
//! let config = BeamConfig {
//!     beam_width: 4,
//!     max_length: 100,
//!     eos_token_id: Some(1),
//!     ..Default::default()
//! };
//! let mut beam = BeamSearch::new(NodeId::ROOT, config);
//!
//! // Expand candidates with logits
//! let logits = vec![0.1, 0.5, 0.3, 0.1]; // vocab size 4
//! beam.expand_with_logits(0, &logits, 3); // top-3 tokens
//!
//! // Step to select top candidates
//! beam.step();
//!
//! // Get best sequence
//! if let Some(best) = beam.best_sequence() {
//!     println!("Best: {:?}", best.tokens);
//! }
//! ```
//!
//! # Scoring Algorithms
//!
//! | Scorer | Formula | Use Case |
//! |--------|---------|----------|
//! | [`UctScorer`] | `Q + C * sqrt(ln(N)/n)` | Exploration-exploitation balance |
//! | `GreedyScorer` | `Q` | Pure exploitation |
//! | `PuctScorer` | `Q + C * P * sqrt(N)/(1+n)` | AlphaZero-style with priors |

mod scorer;
mod expander;
mod mcts;
mod beam;
mod integrated;

pub use scorer::{Scorer, UctScorer, GreedyScorer, PuctScorer, NodeStats, ScorerConfig};
pub use expander::{Expander, TokenExpander, TokenExpanderConfig, UniformExpander, ExpansionResult};
pub use mcts::{MctsSearch, MctsConfig, MctsNode, MctsStats};
pub use beam::{BeamSearch, BeamConfig, BeamCandidate};
pub use integrated::{TreeMcts, TreeBeam, TreeSearchContext};
