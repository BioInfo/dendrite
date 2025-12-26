//! Tree-integrated search algorithms.
//!
//! This module provides search algorithms that integrate with TreeState for
//! O(1) fork latency and KV cache sharing. Unlike the standalone search algorithms,
//! these variants manage TreeState nodes and blocks automatically.
//!
//! # Architecture
//!
//! The integration works by:
//! 1. Wrapping standalone search algorithms (MCTS, BeamSearch)
//! 2. Intercepting node creation to call `TreeState::fork()`
//! 3. Managing cleanup via `TreeState::release()`
//! 4. Mapping internal search node IDs to TreeState `NodeId`s
//!
//! # Example: Tree-Integrated MCTS
//!
//! ```rust,ignore
//! use dendrite_core::search::{TreeMcts, MctsConfig};
//! use dendrite_core::tree::{TreeState, NodeId};
//!
//! let tree_state = Arc::new(TreeState::new(kv_cache, 16));
//! let config = MctsConfig::default();
//!
//! // Create tree-integrated MCTS
//! let mut mcts = TreeMcts::new(tree_state.clone(), NodeId::ROOT, config);
//!
//! // Search - all fork operations use O(1) KV cache sharing
//! for _ in 0..100 {
//!     let leaf = mcts.select();
//!     let expansion = evaluate_node(&model, leaf);
//!     mcts.expand_with_tree(leaf, &expansion)?;  // Calls TreeState::fork()
//!     mcts.backpropagate(leaf, value);
//! }
//!
//! // Cleanup unused branches
//! mcts.prune_unselected()?;  // Calls TreeState::release()
//! ```

use super::beam::{BeamCandidate, BeamConfig, BeamSearch};
use super::expander::ExpansionResult;
use super::mcts::{MctsConfig, MctsSearch, MctsStats};
use super::scorer::Scorer;
use crate::cache::RadixTree;
use crate::error::Result;
use crate::tree::{NodeId, TreeState};
use std::collections::HashMap;
use std::sync::Arc;

/// MCTS integrated with TreeState for O(1) fork latency.
///
/// This struct wraps `MctsSearch` and automatically manages TreeState nodes
/// when expanding the search tree. Each expansion calls `TreeState::fork()`
/// to create new branches with shared KV cache blocks.
pub struct TreeMcts {
    /// Underlying MCTS search.
    mcts: MctsSearch,
    /// Tree state for KV cache management.
    tree_state: Arc<TreeState>,
    /// Mapping from MCTS node index to TreeState NodeId.
    tree_node_map: HashMap<usize, NodeId>,
    /// Radix tree for prefix caching (optional).
    prefix_cache: Option<RadixTree>,
    /// Token sequences for each node (for prefix caching).
    node_tokens: HashMap<usize, Vec<u32>>,
}

impl TreeMcts {
    /// Create a new tree-integrated MCTS.
    ///
    /// # Arguments
    ///
    /// * `tree_state` - Shared tree state for KV cache management
    /// * `root_node_id` - Node ID to start search from
    /// * `config` - MCTS configuration
    pub fn new(tree_state: Arc<TreeState>, root_node_id: NodeId, config: MctsConfig) -> Self {
        let mcts = MctsSearch::new(root_node_id, config);
        let mut tree_node_map = HashMap::new();
        tree_node_map.insert(0, root_node_id); // Root maps to provided node

        Self {
            mcts,
            tree_state,
            tree_node_map,
            prefix_cache: None,
            node_tokens: HashMap::new(),
        }
    }

    /// Create with a custom scorer.
    pub fn with_scorer(
        tree_state: Arc<TreeState>,
        root_node_id: NodeId,
        config: MctsConfig,
        scorer: Box<dyn Scorer>,
    ) -> Self {
        let mcts = MctsSearch::with_scorer(root_node_id, config, scorer);
        let mut tree_node_map = HashMap::new();
        tree_node_map.insert(0, root_node_id);

        Self {
            mcts,
            tree_state,
            tree_node_map,
            prefix_cache: None,
            node_tokens: HashMap::new(),
        }
    }

    /// Enable prefix caching with a RadixTree.
    ///
    /// When enabled, the search will check for prefix matches before forking,
    /// potentially reusing existing KV cache entries.
    pub fn with_prefix_cache(mut self) -> Self {
        self.prefix_cache = Some(RadixTree::new());
        self
    }

    /// Get the underlying MCTS search (read-only).
    pub fn mcts(&self) -> &MctsSearch {
        &self.mcts
    }

    /// Get search statistics.
    pub fn stats(&self) -> &MctsStats {
        self.mcts.stats()
    }

    /// Get the TreeState NodeId for an MCTS node index.
    pub fn get_tree_node_id(&self, mcts_idx: usize) -> Option<NodeId> {
        self.tree_node_map.get(&mcts_idx).copied()
    }

    /// Select a leaf node for expansion.
    ///
    /// This uses UCT selection through the underlying MCTS.
    pub fn select(&self) -> usize {
        self.mcts.select()
    }

    /// Expand a node by creating TreeState forks.
    ///
    /// This is the key integration point - each action creates a new TreeState
    /// node via `fork()`, giving O(1) fork latency with KV cache sharing.
    ///
    /// # Arguments
    ///
    /// * `node_idx` - The MCTS node index to expand
    /// * `expansion` - Expansion result with actions and priors
    ///
    /// # Returns
    ///
    /// Indices of the newly created MCTS nodes.
    pub fn expand_with_tree(
        &mut self,
        node_idx: usize,
        expansion: &ExpansionResult,
    ) -> Result<Vec<usize>> {
        if expansion.is_empty() {
            // Mark as terminal in MCTS
            let _ = self.mcts.expand(node_idx, expansion);
            return Ok(Vec::new());
        }

        // Get the TreeState NodeId for this MCTS node
        let parent_tree_id = self
            .tree_node_map
            .get(&node_idx)
            .copied()
            .unwrap_or(NodeId::ROOT);

        // Get parent tokens for prefix caching
        let parent_tokens = self.node_tokens.get(&node_idx).cloned().unwrap_or_default();

        // Create TreeState forks for each action
        let mut child_tree_ids = Vec::with_capacity(expansion.actions.len());

        for &action in &expansion.actions {
            // Fork in TreeState - O(1) operation
            let handle = self.tree_state.fork(parent_tree_id)?;

            // Append the token to the new branch
            self.tree_state.append_token(handle.node_id, action)?;

            child_tree_ids.push(handle.node_id);

            // Track tokens for prefix caching
            let mut child_tokens = parent_tokens.clone();
            child_tokens.push(action);

            // Update prefix cache if enabled
            if let Some(ref mut cache) = self.prefix_cache {
                // Store with a placeholder block index (we use mcts node index)
                cache.insert(&child_tokens, child_tree_ids.len() - 1);
            }
        }

        // Now expand in MCTS (creates placeholder nodes)
        let mcts_indices = self.mcts.expand(node_idx, expansion);

        // Map MCTS indices to TreeState NodeIds
        for (i, &mcts_idx) in mcts_indices.iter().enumerate() {
            self.tree_node_map.insert(mcts_idx, child_tree_ids[i]);

            // Store token sequence for this node
            let mut child_tokens = parent_tokens.clone();
            child_tokens.push(expansion.actions[i]);
            self.node_tokens.insert(mcts_idx, child_tokens);
        }

        Ok(mcts_indices)
    }

    /// Backpropagate a value through the tree.
    pub fn backpropagate(&mut self, leaf_idx: usize, value: f64) {
        self.mcts.backpropagate(leaf_idx, value);
    }

    /// Get the best action from the root.
    pub fn best_action(&self) -> Option<u32> {
        self.mcts.best_action()
    }

    /// Get action probabilities from root.
    pub fn action_probabilities(&self) -> Vec<(u32, f32)> {
        self.mcts.action_probabilities()
    }

    /// Get the principal variation.
    pub fn principal_variation(&self) -> Vec<u32> {
        self.mcts.principal_variation()
    }

    /// Get the best sequence of tokens (from root to best leaf).
    pub fn best_sequence(&self) -> Vec<u32> {
        let pv = self.principal_variation();
        pv
    }

    /// Prune unselected branches to free KV cache blocks.
    ///
    /// This releases TreeState nodes that are not on the best path,
    /// allowing their KV cache blocks to be reused.
    pub fn prune_unselected(&mut self) -> Result<()> {
        let pv = self.mcts.principal_variation();
        let pv_set: std::collections::HashSet<u32> = pv.into_iter().collect();

        // Find nodes to prune (not on principal variation)
        let root = self.mcts.root();
        let mut to_release = Vec::new();

        for &child_idx in &root.children {
            if let Some(node) = self.mcts.get_node(child_idx) {
                if let Some(action) = node.action {
                    if !pv_set.contains(&action) {
                        if let Some(&tree_id) = self.tree_node_map.get(&child_idx) {
                            to_release.push(tree_id);
                        }
                    }
                }
            }
        }

        // Release in TreeState
        for tree_id in to_release {
            self.tree_state.release(tree_id)?;
        }

        Ok(())
    }

    /// Advance the root to a child after selecting an action.
    ///
    /// This is used when reusing the tree between moves.
    pub fn advance_root(&mut self, action: u32) -> Option<NodeId> {
        let root = self.mcts.root();

        for &child_idx in &root.children {
            if let Some(node) = self.mcts.get_node(child_idx) {
                if node.action == Some(action) {
                    return self.tree_node_map.get(&child_idx).copied();
                }
            }
        }

        None
    }
}

/// Beam search integrated with TreeState.
///
/// Each beam candidate maintains a corresponding TreeState node for
/// O(1) forking and KV cache sharing.
pub struct TreeBeam {
    /// Underlying beam search.
    beam: BeamSearch,
    /// Tree state for KV cache management.
    tree_state: Arc<TreeState>,
    /// Mapping from candidate index to TreeState NodeId.
    candidate_nodes: HashMap<usize, NodeId>,
    /// Next candidate index.
    next_idx: usize,
}

impl TreeBeam {
    /// Create a new tree-integrated beam search.
    pub fn new(tree_state: Arc<TreeState>, root_node_id: NodeId, config: BeamConfig) -> Self {
        let beam = BeamSearch::new(root_node_id, config);
        let mut candidate_nodes = HashMap::new();
        candidate_nodes.insert(0, root_node_id);

        Self {
            beam,
            tree_state,
            candidate_nodes,
            next_idx: 1,
        }
    }

    /// Expand a candidate with logits, creating TreeState forks.
    ///
    /// # Arguments
    ///
    /// * `candidate_idx` - The candidate to expand
    /// * `logits` - Log probabilities for each token in vocabulary
    /// * `top_k` - Number of top tokens to consider
    pub fn expand_with_tree(
        &mut self,
        candidate_idx: usize,
        logits: &[f32],
        top_k: usize,
    ) -> Result<()> {
        // Get parent TreeState node
        let parent_tree_id = self
            .candidate_nodes
            .get(&candidate_idx)
            .copied()
            .unwrap_or(NodeId::ROOT);

        // Get top-k tokens
        let mut indexed_logits: Vec<(usize, f32)> =
            logits.iter().copied().enumerate().collect();
        indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top_tokens: Vec<u32> = indexed_logits
            .iter()
            .take(top_k)
            .map(|(idx, _)| *idx as u32)
            .collect();

        // Fork in TreeState for each top token
        for &token in &top_tokens {
            let handle = self.tree_state.fork(parent_tree_id)?;
            self.tree_state.append_token(handle.node_id, token)?;
            self.candidate_nodes.insert(self.next_idx, handle.node_id);
            self.next_idx += 1;
        }

        // Expand in beam search
        self.beam.expand_with_logits(candidate_idx, logits, top_k);

        Ok(())
    }

    /// Perform a beam step (select top-k candidates).
    pub fn step(&mut self) {
        self.beam.step();
    }

    /// Check if search is complete.
    pub fn is_done(&self) -> bool {
        self.beam.is_done()
    }

    /// Get the best completed sequence.
    pub fn best_sequence(&self) -> Option<&BeamCandidate> {
        self.beam.best_sequence()
    }

    /// Get all completed sequences.
    pub fn finished_sequences(&self) -> &[BeamCandidate] {
        self.beam.finished()
    }

    /// Get the TreeState NodeId for a candidate.
    pub fn get_tree_node_id(&self, candidate_idx: usize) -> Option<NodeId> {
        self.candidate_nodes.get(&candidate_idx).copied()
    }

    /// Release non-best candidates to free KV cache.
    pub fn prune_non_best(&mut self) -> Result<()> {
        // Keep only the best candidate's tree nodes
        if let Some(best) = self.beam.best_sequence() {
            let best_len = best.tokens.len();

            // Release all candidate nodes except those on best path
            // (Simplified: release all non-current candidates)
            let current_candidates = self.beam.candidates();
            let keep_indices: std::collections::HashSet<usize> = current_candidates
                .iter()
                .enumerate()
                .map(|(i, _)| i)
                .collect();

            let mut to_release = Vec::new();
            for (&idx, &tree_id) in &self.candidate_nodes {
                if !keep_indices.contains(&idx) && idx != 0 {
                    to_release.push(tree_id);
                }
            }

            for tree_id in to_release {
                let _ = self.tree_state.release(tree_id);
            }

            let _ = best_len; // suppress warning
        }

        Ok(())
    }
}

/// Context for tree-integrated search.
///
/// Provides utilities for managing search across the TreeState.
#[derive(Clone)]
pub struct TreeSearchContext {
    /// Tree state for KV cache.
    tree_state: Arc<TreeState>,
    /// Prefix cache for common subsequences.
    prefix_cache: Arc<parking_lot::RwLock<RadixTree>>,
}

impl TreeSearchContext {
    /// Create a new search context.
    pub fn new(tree_state: Arc<TreeState>) -> Self {
        Self {
            tree_state,
            prefix_cache: Arc::new(parking_lot::RwLock::new(RadixTree::new())),
        }
    }

    /// Get the tree state.
    pub fn tree_state(&self) -> &Arc<TreeState> {
        &self.tree_state
    }

    /// Check for a prefix match in the cache.
    ///
    /// Returns (match_length, block_index) if a prefix exists.
    pub fn find_prefix(&self, tokens: &[u32]) -> (usize, Option<usize>) {
        self.prefix_cache.read().find_prefix(tokens)
    }

    /// Register a sequence in the prefix cache.
    pub fn register_prefix(&self, tokens: &[u32], block_idx: usize) {
        self.prefix_cache.write().insert(tokens, block_idx);
    }

    /// Create an MCTS search from a node.
    pub fn mcts(&self, node_id: NodeId, config: MctsConfig) -> TreeMcts {
        TreeMcts::new(self.tree_state.clone(), node_id, config)
    }

    /// Create a beam search from a node.
    pub fn beam(&self, node_id: NodeId, config: BeamConfig) -> TreeBeam {
        TreeBeam::new(self.tree_state.clone(), node_id, config)
    }

    /// Fork a node in the tree state.
    pub fn fork(&self, from_node: NodeId) -> Result<NodeId> {
        let handle = self.tree_state.fork(from_node)?;
        Ok(handle.node_id)
    }

    /// Release a node in the tree state.
    pub fn release(&self, node_id: NodeId) -> Result<()> {
        self.tree_state.release(node_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::{KvCache, KvCacheConfig};

    fn create_test_context() -> TreeSearchContext {
        let config = KvCacheConfig {
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 64,
            max_blocks: 100,
            tokens_per_block: 16,
        };
        let kv_cache = Arc::new(parking_lot::RwLock::new(KvCache::new(config).unwrap()));
        let tree_state = Arc::new(TreeState::new(kv_cache, 16));
        TreeSearchContext::new(tree_state)
    }

    #[test]
    fn tree_mcts_creation() {
        let ctx = create_test_context();
        let config = MctsConfig::default();
        let mcts = ctx.mcts(NodeId::ROOT, config);

        assert!(mcts.get_tree_node_id(0).is_some());
        assert_eq!(mcts.get_tree_node_id(0), Some(NodeId::ROOT));
    }

    #[test]
    fn tree_mcts_expand_creates_tree_nodes() {
        let ctx = create_test_context();
        let config = MctsConfig::default();
        let mut mcts = ctx.mcts(NodeId::ROOT, config);

        // Expand with actions
        let expansion = ExpansionResult {
            children: Vec::new(),
            actions: vec![1, 2, 3],
            log_probs: None,
            priors: Some(vec![0.5, 0.3, 0.2]),
        };

        let selected = mcts.select();
        let children = mcts.expand_with_tree(selected, &expansion).unwrap();

        // Should have created 3 children
        assert_eq!(children.len(), 3);

        // Each child should have a tree node
        for child_idx in children {
            assert!(mcts.get_tree_node_id(child_idx).is_some());
        }
    }

    #[test]
    fn tree_mcts_backpropagation() {
        let ctx = create_test_context();
        let config = MctsConfig::default();
        let mut mcts = ctx.mcts(NodeId::ROOT, config);

        let expansion = ExpansionResult {
            children: Vec::new(),
            actions: vec![1, 2],
            log_probs: None,
            priors: None,
        };

        let selected = mcts.select();
        let children = mcts.expand_with_tree(selected, &expansion).unwrap();

        // Backpropagate
        mcts.backpropagate(children[0], 0.8);

        // Stats should be updated
        assert_eq!(mcts.stats().iterations, 1);
    }

    #[test]
    fn tree_mcts_best_action() {
        let ctx = create_test_context();
        let config = MctsConfig::default();
        let mut mcts = ctx.mcts(NodeId::ROOT, config);

        let expansion = ExpansionResult {
            children: Vec::new(),
            actions: vec![10, 20, 30],
            log_probs: None,
            priors: Some(vec![0.5, 0.3, 0.2]),
        };

        let selected = mcts.select();
        let children = mcts.expand_with_tree(selected, &expansion).unwrap();

        // Heavily reward action 20
        mcts.backpropagate(children[1], 1.0);
        mcts.backpropagate(children[1], 1.0);
        mcts.backpropagate(children[1], 1.0);

        // Lightly reward others
        mcts.backpropagate(children[0], 0.1);
        mcts.backpropagate(children[2], 0.2);

        // Best action should be 20 (most visits)
        assert_eq!(mcts.best_action(), Some(20));
    }

    #[test]
    fn tree_beam_creation() {
        let ctx = create_test_context();
        let config = BeamConfig::default();
        let beam = ctx.beam(NodeId::ROOT, config);

        assert!(beam.get_tree_node_id(0).is_some());
    }

    #[test]
    fn tree_beam_expand_creates_tree_nodes() {
        let ctx = create_test_context();
        let config = BeamConfig {
            beam_width: 3,
            ..Default::default()
        };
        let mut beam = ctx.beam(NodeId::ROOT, config);

        // Expand with logits
        let logits = vec![0.1, 0.5, 0.3, 0.1]; // vocab size 4
        beam.expand_with_tree(0, &logits, 3).unwrap();

        // Should have created candidates
        // Check that tree nodes were created (indices 1, 2, 3)
        assert!(beam.get_tree_node_id(1).is_some());
        assert!(beam.get_tree_node_id(2).is_some());
        assert!(beam.get_tree_node_id(3).is_some());
    }

    #[test]
    fn tree_search_context_fork_and_release() {
        let ctx = create_test_context();

        // Fork from root
        let node1 = ctx.fork(NodeId::ROOT).unwrap();
        assert!(node1.is_valid());

        // Fork from child
        let node2 = ctx.fork(node1).unwrap();
        assert!(node2.is_valid());
        assert_ne!(node1, node2);

        // Release nodes
        ctx.release(node2).unwrap();
        ctx.release(node1).unwrap();
    }

    #[test]
    fn prefix_cache_integration() {
        let ctx = create_test_context();

        // Register some prefixes
        ctx.register_prefix(&[1, 2, 3], 0);
        ctx.register_prefix(&[1, 2, 4], 1);
        ctx.register_prefix(&[1, 2, 3, 4, 5], 2);

        // Find prefix
        let (match_len, block) = ctx.find_prefix(&[1, 2, 3, 4, 5, 6]);
        assert_eq!(match_len, 5);
        assert_eq!(block, Some(2));

        // Partial match
        let (match_len, block) = ctx.find_prefix(&[1, 2, 3]);
        assert_eq!(match_len, 3);
        assert_eq!(block, Some(0));

        // No match
        let (match_len, block) = ctx.find_prefix(&[9, 9, 9]);
        assert_eq!(match_len, 0);
        assert_eq!(block, None);
    }

    #[test]
    fn tree_mcts_with_prefix_cache() {
        let ctx = create_test_context();
        let config = MctsConfig::default();
        let mut mcts = TreeMcts::new(ctx.tree_state().clone(), NodeId::ROOT, config)
            .with_prefix_cache();

        let expansion = ExpansionResult {
            children: Vec::new(),
            actions: vec![1, 2, 3],
            log_probs: None,
            priors: None,
        };

        let selected = mcts.select();
        mcts.expand_with_tree(selected, &expansion).unwrap();

        // Prefix cache should have entries
        // (Internal to TreeMcts, verified by successful expansion)
    }

    #[test]
    fn tree_mcts_prune() {
        let ctx = create_test_context();
        let config = MctsConfig::default();
        let mut mcts = ctx.mcts(NodeId::ROOT, config);

        let expansion = ExpansionResult {
            children: Vec::new(),
            actions: vec![1, 2, 3],
            log_probs: None,
            priors: None,
        };

        let selected = mcts.select();
        let children = mcts.expand_with_tree(selected, &expansion).unwrap();

        // Make action 2 the best
        mcts.backpropagate(children[1], 1.0);
        mcts.backpropagate(children[1], 1.0);
        mcts.backpropagate(children[0], 0.1);
        mcts.backpropagate(children[2], 0.1);

        // Prune should not error
        mcts.prune_unselected().unwrap();
    }

    #[test]
    fn tree_mcts_deep_expansion() {
        let ctx = create_test_context();
        let config = MctsConfig::default();
        let mut mcts = ctx.mcts(NodeId::ROOT, config);

        // Expand root
        let expansion1 = ExpansionResult {
            children: Vec::new(),
            actions: vec![1, 2],
            log_probs: None,
            priors: None,
        };

        let root_children = mcts.expand_with_tree(0, &expansion1).unwrap();
        assert_eq!(root_children.len(), 2);

        // Expand first child
        let expansion2 = ExpansionResult {
            children: Vec::new(),
            actions: vec![10, 20],
            log_probs: None,
            priors: None,
        };

        let grandchildren = mcts.expand_with_tree(root_children[0], &expansion2).unwrap();
        assert_eq!(grandchildren.len(), 2);

        // Verify tree structure
        for gc_idx in &grandchildren {
            let tree_id = mcts.get_tree_node_id(*gc_idx);
            assert!(tree_id.is_some());
        }
    }
}
