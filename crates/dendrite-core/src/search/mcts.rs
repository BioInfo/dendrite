//! Monte Carlo Tree Search implementation.
//!
//! MCTS for LLM inference with UCT selection and parallel rollouts.
//!
//! # Algorithm
//!
//! Each iteration of MCTS consists of four phases:
//! 1. **Selection**: Traverse tree using UCT to find a promising leaf
//! 2. **Expansion**: Create child nodes for unexplored actions
//! 3. **Simulation**: Evaluate the leaf (via model forward pass)
//! 4. **Backpropagation**: Update statistics along the path
//!
//! # Example
//!
//! ```ignore
//! let config = MctsConfig::default();
//! let mut mcts = MctsSearch::new(tree_state, config);
//!
//! for _ in 0..num_iterations {
//!     let leaf = mcts.select();
//!     let value = evaluate(&model, leaf).await;
//!     mcts.backpropagate(leaf, value);
//! }
//!
//! let best = mcts.best_action();
//! ```

use super::expander::ExpansionResult;
use super::scorer::{NodeStats, Scorer, UctScorer};
use crate::tree::NodeId;
use std::collections::HashMap;

/// Configuration for MCTS.
#[derive(Debug, Clone)]
pub struct MctsConfig {
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Maximum depth to search.
    pub max_depth: usize,
    /// Exploration constant for UCT.
    pub exploration_constant: f64,
    /// Number of parallel simulations (if supported).
    pub num_parallel: usize,
    /// Whether to reuse tree between moves.
    pub reuse_tree: bool,
    /// Discount factor for rewards (gamma).
    pub discount: f64,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            max_depth: 50,
            exploration_constant: std::f64::consts::SQRT_2,
            num_parallel: 1,
            reuse_tree: true,
            discount: 1.0,
        }
    }
}

/// Statistics about MCTS search.
#[derive(Debug, Clone, Default)]
pub struct MctsStats {
    /// Total iterations performed.
    pub iterations: usize,
    /// Total nodes created.
    pub nodes_created: usize,
    /// Maximum depth reached.
    pub max_depth_reached: usize,
    /// Average depth of simulations.
    pub avg_depth: f64,
    /// Best value found.
    pub best_value: f64,
}

/// A node in the MCTS tree.
#[derive(Debug, Clone)]
pub struct MctsNode {
    /// Node ID in the underlying tree.
    pub node_id: NodeId,
    /// Action that led to this node.
    pub action: Option<u32>,
    /// Statistics for this node.
    pub stats: NodeStats,
    /// Children of this node.
    pub children: Vec<usize>, // Indices into MctsSearch.nodes
    /// Parent node index.
    pub parent: Option<usize>,
    /// Depth in the search tree.
    pub depth: usize,
    /// Prior probability from policy.
    pub prior: f32,
}

impl MctsNode {
    /// Create a root node.
    fn root(node_id: NodeId) -> Self {
        Self {
            node_id,
            action: None,
            stats: NodeStats::new(),
            children: Vec::new(),
            parent: None,
            depth: 0,
            prior: 1.0,
        }
    }

    /// Create a child node.
    fn child(node_id: NodeId, action: u32, parent_idx: usize, depth: usize, prior: f32) -> Self {
        Self {
            node_id,
            action: Some(action),
            stats: NodeStats::new(),
            children: Vec::new(),
            parent: Some(parent_idx),
            depth,
            prior,
        }
    }
}

/// Monte Carlo Tree Search.
pub struct MctsSearch {
    /// Configuration.
    config: MctsConfig,
    /// Scorer for node selection.
    scorer: Box<dyn Scorer>,
    /// All nodes in the tree.
    nodes: Vec<MctsNode>,
    /// Map from NodeId to node index.
    node_map: HashMap<NodeId, usize>,
    /// Search statistics.
    stats: MctsStats,
}

impl MctsSearch {
    /// Create a new MCTS search.
    pub fn new(root_node_id: NodeId, config: MctsConfig) -> Self {
        let scorer = Box::new(UctScorer::with_exploration(config.exploration_constant));

        let root = MctsNode::root(root_node_id);
        let mut node_map = HashMap::new();
        node_map.insert(root_node_id, 0);

        Self {
            config,
            scorer,
            nodes: vec![root],
            node_map,
            stats: MctsStats::default(),
        }
    }

    /// Create with custom scorer.
    pub fn with_scorer(root_node_id: NodeId, config: MctsConfig, scorer: Box<dyn Scorer>) -> Self {
        let root = MctsNode::root(root_node_id);
        let mut node_map = HashMap::new();
        node_map.insert(root_node_id, 0);

        Self {
            config,
            scorer,
            nodes: vec![root],
            node_map,
            stats: MctsStats::default(),
        }
    }

    /// Get the root node.
    pub fn root(&self) -> &MctsNode {
        &self.nodes[0]
    }

    /// Get a node by index.
    pub fn get_node(&self, idx: usize) -> Option<&MctsNode> {
        self.nodes.get(idx)
    }

    /// Get a node by NodeId.
    pub fn get_node_by_id(&self, node_id: NodeId) -> Option<&MctsNode> {
        self.node_map.get(&node_id).map(|&idx| &self.nodes[idx])
    }

    /// Get mutable node by index.
    #[allow(dead_code)]
    fn get_node_mut(&mut self, idx: usize) -> Option<&mut MctsNode> {
        self.nodes.get_mut(idx)
    }

    /// Get search statistics.
    pub fn stats(&self) -> &MctsStats {
        &self.stats
    }

    /// Select a leaf node for expansion using UCT.
    ///
    /// Returns the index of the selected node.
    pub fn select(&self) -> usize {
        let mut current = 0;

        loop {
            let node = &self.nodes[current];

            // If terminal or not fully expanded, return this node
            if node.stats.is_terminal || !node.stats.is_fully_expanded {
                return current;
            }

            // If no children, return this node
            if node.children.is_empty() {
                return current;
            }

            // Select best child according to scorer
            let parent_visits = node.stats.visits;
            let mut best_score = f64::NEG_INFINITY;
            let mut best_child = node.children[0];

            for &child_idx in &node.children {
                let child = &self.nodes[child_idx];
                let score = self.scorer.score(&child.stats, parent_visits);

                if score > best_score {
                    best_score = score;
                    best_child = child_idx;
                }
            }

            current = best_child;

            // Depth limit check
            if self.nodes[current].depth >= self.config.max_depth {
                return current;
            }
        }
    }

    /// Expand a node by adding children for the given actions.
    ///
    /// Returns indices of the new children.
    pub fn expand(&mut self, node_idx: usize, expansion: &ExpansionResult) -> Vec<usize> {
        if expansion.is_empty() {
            self.nodes[node_idx].stats.is_terminal = true;
            return Vec::new();
        }

        let parent_depth = self.nodes[node_idx].depth;
        let priors = expansion.priors.as_ref();

        let mut child_indices = Vec::with_capacity(expansion.actions.len());

        for (i, &action) in expansion.actions.iter().enumerate() {
            let prior = priors.map(|p| p[i]).unwrap_or(1.0 / expansion.actions.len() as f32);

            // Create placeholder NodeId (actual tree node created externally)
            let child_node_id = NodeId::new();

            let child = MctsNode::child(
                child_node_id,
                action,
                node_idx,
                parent_depth + 1,
                prior,
            );

            let child_idx = self.nodes.len();
            self.node_map.insert(child_node_id, child_idx);
            self.nodes.push(child);
            child_indices.push(child_idx);

            self.stats.nodes_created += 1;
            self.stats.max_depth_reached = self.stats.max_depth_reached.max(parent_depth + 1);
        }

        self.nodes[node_idx].children = child_indices.clone();
        self.nodes[node_idx].stats.is_fully_expanded = true;

        child_indices
    }

    /// Backpropagate a value through the tree.
    pub fn backpropagate(&mut self, leaf_idx: usize, value: f64) {
        let mut current = Some(leaf_idx);
        let mut current_value = value;

        while let Some(idx) = current {
            self.nodes[idx].stats.update(current_value);
            current = self.nodes[idx].parent;
            current_value *= self.config.discount;
        }

        self.stats.iterations += 1;
        self.stats.best_value = self.stats.best_value.max(value);
    }

    /// Get the best action from the root.
    ///
    /// Uses visit count as the selection criterion (more robust than value).
    pub fn best_action(&self) -> Option<u32> {
        let root = &self.nodes[0];

        if root.children.is_empty() {
            return None;
        }

        let mut best_visits = 0;
        let mut best_action = None;

        for &child_idx in &root.children {
            let child = &self.nodes[child_idx];
            if child.stats.visits > best_visits {
                best_visits = child.stats.visits;
                best_action = child.action;
            }
        }

        best_action
    }

    /// Get action probabilities from root (based on visit counts).
    pub fn action_probabilities(&self) -> Vec<(u32, f32)> {
        let root = &self.nodes[0];

        if root.children.is_empty() {
            return Vec::new();
        }

        let total_visits: u32 = root.children.iter()
            .map(|&idx| self.nodes[idx].stats.visits)
            .sum();

        if total_visits == 0 {
            return Vec::new();
        }

        root.children.iter()
            .filter_map(|&idx| {
                let child = &self.nodes[idx];
                child.action.map(|a| (a, child.stats.visits as f32 / total_visits as f32))
            })
            .collect()
    }

    /// Get the principal variation (best path from root).
    pub fn principal_variation(&self) -> Vec<u32> {
        let mut pv = Vec::new();
        let mut current = 0;

        loop {
            let node = &self.nodes[current];

            if node.children.is_empty() {
                break;
            }

            // Find most visited child
            let mut best_visits = 0;
            let mut best_child = None;

            for &child_idx in &node.children {
                let child = &self.nodes[child_idx];
                if child.stats.visits > best_visits {
                    best_visits = child.stats.visits;
                    best_child = Some(child_idx);
                }
            }

            match best_child {
                Some(child_idx) => {
                    if let Some(action) = self.nodes[child_idx].action {
                        pv.push(action);
                    }
                    current = child_idx;
                }
                None => break,
            }
        }

        pv
    }

    /// Reset the search tree, keeping only the root.
    pub fn reset(&mut self) {
        let root_id = self.nodes[0].node_id;
        self.nodes.clear();
        self.node_map.clear();
        self.stats = MctsStats::default();

        let root = MctsNode::root(root_id);
        self.node_map.insert(root_id, 0);
        self.nodes.push(root);
    }

    /// Number of nodes in the tree.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_mcts() -> MctsSearch {
        MctsSearch::new(NodeId::ROOT, MctsConfig::default())
    }

    #[test]
    fn mcts_creation() {
        let mcts = create_mcts();
        assert_eq!(mcts.num_nodes(), 1);
        assert_eq!(mcts.root().node_id, NodeId::ROOT);
    }

    #[test]
    fn mcts_select_returns_root_when_unexpanded() {
        let mcts = create_mcts();
        let selected = mcts.select();
        assert_eq!(selected, 0); // Root
    }

    #[test]
    fn mcts_expand_creates_children() {
        let mut mcts = create_mcts();

        let expansion = ExpansionResult {
            children: Vec::new(),
            actions: vec![1, 2, 3],
            log_probs: None,
            priors: Some(vec![0.5, 0.3, 0.2]),
        };

        let children = mcts.expand(0, &expansion);

        assert_eq!(children.len(), 3);
        assert_eq!(mcts.num_nodes(), 4);
        assert_eq!(mcts.root().children.len(), 3);
    }

    #[test]
    fn mcts_backpropagate_updates_stats() {
        let mut mcts = create_mcts();

        // Expand root
        let expansion = ExpansionResult {
            children: Vec::new(),
            actions: vec![1, 2],
            log_probs: None,
            priors: None,
        };
        let children = mcts.expand(0, &expansion);

        // Backprop from first child
        mcts.backpropagate(children[0], 0.8);

        assert_eq!(mcts.root().stats.visits, 1);
        assert_eq!(mcts.nodes[children[0]].stats.visits, 1);
        assert_eq!(mcts.nodes[children[0]].stats.mean_reward, 0.8);
    }

    #[test]
    fn mcts_best_action_uses_visit_count() {
        let mut mcts = create_mcts();

        let expansion = ExpansionResult {
            children: Vec::new(),
            actions: vec![10, 20, 30],
            log_probs: None,
            priors: None,
        };
        let children = mcts.expand(0, &expansion);

        // Visit second action more
        mcts.backpropagate(children[0], 0.9); // High reward
        mcts.backpropagate(children[1], 0.5);
        mcts.backpropagate(children[1], 0.5);
        mcts.backpropagate(children[1], 0.5); // More visits
        mcts.backpropagate(children[2], 0.3);

        // Should select action 20 (most visits)
        assert_eq!(mcts.best_action(), Some(20));
    }

    #[test]
    fn mcts_principal_variation() {
        let mut mcts = create_mcts();

        // Build a small tree
        let expansion1 = ExpansionResult {
            children: Vec::new(),
            actions: vec![1, 2],
            log_probs: None,
            priors: None,
        };
        let level1 = mcts.expand(0, &expansion1);

        let expansion2 = ExpansionResult {
            children: Vec::new(),
            actions: vec![3, 4],
            log_probs: None,
            priors: None,
        };
        let level2 = mcts.expand(level1[0], &expansion2);

        // Visit path: root -> action 1 -> action 4
        mcts.backpropagate(level2[1], 1.0);
        mcts.backpropagate(level2[1], 1.0);
        mcts.backpropagate(level2[0], 0.5);
        mcts.backpropagate(level1[1], 0.3);

        let pv = mcts.principal_variation();
        assert_eq!(pv, vec![1, 4]);
    }

    #[test]
    fn mcts_action_probabilities() {
        let mut mcts = create_mcts();

        let expansion = ExpansionResult {
            children: Vec::new(),
            actions: vec![1, 2],
            log_probs: None,
            priors: None,
        };
        let children = mcts.expand(0, &expansion);

        mcts.backpropagate(children[0], 1.0);
        mcts.backpropagate(children[0], 1.0);
        mcts.backpropagate(children[0], 1.0);
        mcts.backpropagate(children[1], 1.0);

        let probs = mcts.action_probabilities();

        assert_eq!(probs.len(), 2);
        // Action 1 has 3 visits, action 2 has 1 visit
        // Probabilities: 0.75 and 0.25
        let (a1, p1) = probs.iter().find(|(a, _)| *a == 1).unwrap();
        let (a2, p2) = probs.iter().find(|(a, _)| *a == 2).unwrap();
        assert!((p1 - 0.75).abs() < 0.01);
        assert!((p2 - 0.25).abs() < 0.01);
    }

    #[test]
    fn mcts_reset() {
        let mut mcts = create_mcts();

        let expansion = ExpansionResult {
            children: Vec::new(),
            actions: vec![1, 2, 3],
            log_probs: None,
            priors: None,
        };
        mcts.expand(0, &expansion);

        assert_eq!(mcts.num_nodes(), 4);

        mcts.reset();

        assert_eq!(mcts.num_nodes(), 1);
        assert_eq!(mcts.root().children.len(), 0);
    }

    #[test]
    fn mcts_stats_tracking() {
        let mut mcts = create_mcts();

        let expansion = ExpansionResult {
            children: Vec::new(),
            actions: vec![1, 2],
            log_probs: None,
            priors: None,
        };
        let children = mcts.expand(0, &expansion);

        mcts.backpropagate(children[0], 0.8);
        mcts.backpropagate(children[1], 0.9);

        let stats = mcts.stats();
        assert_eq!(stats.iterations, 2);
        assert_eq!(stats.nodes_created, 2);
        assert_eq!(stats.best_value, 0.9);
    }
}
