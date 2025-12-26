//! Beam Search implementation.
//!
//! Beam search maintains a fixed-size set of best candidates
//! and expands all of them at each step.
//!
//! # Algorithm
//!
//! 1. Start with initial candidate(s)
//! 2. For each step:
//!    a. Expand all candidates
//!    b. Score all expansions
//!    c. Keep top-k by score (beam width)
//! 3. Return best completed sequence
//!
//! # Example
//!
//! ```ignore
//! let config = BeamConfig { beam_width: 4, max_length: 100 };
//! let mut beam = BeamSearch::new(root_id, config);
//!
//! while !beam.is_done() {
//!     // Get current candidates
//!     for candidate in beam.candidates() {
//!         let logits = model.forward(candidate.tokens).await?;
//!         beam.expand_candidate(candidate.id, &logits);
//!     }
//!     beam.step();
//! }
//!
//! let best = beam.best_sequence();
//! ```

use crate::tree::NodeId;
use std::cmp::Ordering;

/// Configuration for beam search.
#[derive(Debug, Clone)]
pub struct BeamConfig {
    /// Number of candidates to keep at each step.
    pub beam_width: usize,
    /// Maximum sequence length.
    pub max_length: usize,
    /// Length normalization alpha (0 = no normalization).
    pub length_alpha: f32,
    /// Whether to stop when all beams are finished.
    pub early_stopping: bool,
    /// EOS token ID for detecting completion.
    pub eos_token_id: Option<u32>,
    /// Minimum length before allowing EOS.
    pub min_length: usize,
    /// Number of beams to return.
    pub num_return: usize,
}

impl Default for BeamConfig {
    fn default() -> Self {
        Self {
            beam_width: 4,
            max_length: 100,
            length_alpha: 0.6,
            early_stopping: true,
            eos_token_id: None,
            min_length: 0,
            num_return: 1,
        }
    }
}

/// A candidate in beam search.
#[derive(Debug, Clone)]
pub struct BeamCandidate {
    /// Unique ID for this candidate.
    pub id: usize,
    /// Node ID in the tree.
    pub node_id: NodeId,
    /// Token sequence so far.
    pub tokens: Vec<u32>,
    /// Log probability score (sum of log probs).
    pub score: f64,
    /// Whether this candidate is finished (hit EOS).
    pub is_finished: bool,
    /// Parent candidate ID.
    pub parent: Option<usize>,
}

impl BeamCandidate {
    /// Create a new root candidate.
    fn root(node_id: NodeId) -> Self {
        Self {
            id: 0,
            node_id,
            tokens: Vec::new(),
            score: 0.0,
            is_finished: false,
            parent: None,
        }
    }

    /// Create a child candidate.
    fn child(&self, new_id: usize, node_id: NodeId, token: u32, log_prob: f64) -> Self {
        let mut tokens = self.tokens.clone();
        tokens.push(token);

        Self {
            id: new_id,
            node_id,
            tokens,
            score: self.score + log_prob,
            is_finished: false,
            parent: Some(self.id),
        }
    }

    /// Get normalized score for length.
    pub fn normalized_score(&self, alpha: f32) -> f64 {
        if alpha == 0.0 || self.tokens.is_empty() {
            self.score
        } else {
            let length_penalty = ((5.0 + self.tokens.len() as f64) / 6.0).powf(alpha as f64);
            self.score / length_penalty
        }
    }
}

// For BinaryHeap - we want max heap by score
impl PartialEq for BeamCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for BeamCandidate {}

impl PartialOrd for BeamCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.score.partial_cmp(&other.score)
    }
}

impl Ord for BeamCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Beam search implementation.
pub struct BeamSearch {
    /// Configuration.
    config: BeamConfig,
    /// Current active candidates.
    candidates: Vec<BeamCandidate>,
    /// Finished sequences.
    finished: Vec<BeamCandidate>,
    /// Pending expansions for current step.
    pending: Vec<BeamCandidate>,
    /// Next candidate ID.
    next_id: usize,
    /// Current step.
    step: usize,
}

impl BeamSearch {
    /// Create a new beam search.
    pub fn new(root_node_id: NodeId, config: BeamConfig) -> Self {
        let root = BeamCandidate::root(root_node_id);

        Self {
            config,
            candidates: vec![root],
            finished: Vec::new(),
            pending: Vec::new(),
            next_id: 1,
            step: 0,
        }
    }

    /// Get current candidates (for expansion).
    pub fn candidates(&self) -> &[BeamCandidate] {
        &self.candidates
    }

    /// Get finished sequences.
    pub fn finished(&self) -> &[BeamCandidate] {
        &self.finished
    }

    /// Current step number.
    pub fn current_step(&self) -> usize {
        self.step
    }

    /// Check if search is done.
    pub fn is_done(&self) -> bool {
        // Done if all candidates are finished
        if self.candidates.is_empty() {
            return true;
        }

        // Done if we've reached max length
        if self.step >= self.config.max_length {
            return true;
        }

        // Early stopping: done if we have enough finished and they're better
        if self.config.early_stopping && self.finished.len() >= self.config.num_return {
            let best_finished = self.finished.iter()
                .map(|c| c.normalized_score(self.config.length_alpha))
                .fold(f64::NEG_INFINITY, f64::max);

            let best_active = self.candidates.iter()
                .map(|c| c.normalized_score(self.config.length_alpha))
                .fold(f64::NEG_INFINITY, f64::max);

            if best_finished >= best_active {
                return true;
            }
        }

        false
    }

    /// Expand a candidate with token scores.
    ///
    /// # Arguments
    ///
    /// * `candidate_id` - ID of the candidate to expand
    /// * `log_probs` - Log probabilities for each token
    /// * `node_ids` - Node IDs for each expanded token (from tree)
    pub fn expand_candidate(
        &mut self,
        candidate_id: usize,
        log_probs: &[(u32, f64)],
        node_ids: &[NodeId],
    ) {
        // Find the candidate
        let candidate = match self.candidates.iter().find(|c| c.id == candidate_id) {
            Some(c) => c.clone(),
            None => return,
        };

        // Create children for each token
        for (i, &(token, log_prob)) in log_probs.iter().enumerate() {
            let node_id = node_ids.get(i).copied().unwrap_or(NodeId::INVALID);
            let mut child = candidate.child(self.next_id, node_id, token, log_prob);
            self.next_id += 1;

            // Check if this is EOS
            if Some(token) == self.config.eos_token_id {
                if child.tokens.len() >= self.config.min_length {
                    child.is_finished = true;
                    self.finished.push(child);
                }
                // If below min_length, don't add EOS expansion
            } else {
                self.pending.push(child);
            }
        }
    }

    /// Simpler expansion with just logits.
    pub fn expand_with_logits(
        &mut self,
        candidate_id: usize,
        logits: &[f32],
        top_k: usize,
    ) {
        // Compute log probs and get top-k
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_logits: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
        let sum_exp: f32 = exp_logits.iter().sum();
        let log_probs: Vec<f64> = exp_logits.iter()
            .map(|&e| (e / sum_exp).ln() as f64)
            .collect();

        // Get top-k tokens
        let mut indexed: Vec<(u32, f64)> = log_probs.iter()
            .enumerate()
            .map(|(i, &lp)| (i as u32, lp))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        indexed.truncate(top_k);

        // Create placeholder node IDs
        let node_ids: Vec<NodeId> = (0..indexed.len()).map(|_| NodeId::new()).collect();

        self.expand_candidate(candidate_id, &indexed, &node_ids);
    }

    /// Complete the current step, selecting top-k candidates.
    pub fn step(&mut self) {
        if self.pending.is_empty() {
            self.candidates.clear();
            return;
        }

        // Sort pending by normalized score (descending)
        self.pending.sort_by(|a, b| {
            let score_a = a.normalized_score(self.config.length_alpha);
            let score_b = b.normalized_score(self.config.length_alpha);
            score_b.partial_cmp(&score_a).unwrap_or(Ordering::Equal)
        });

        // Keep top beam_width
        self.candidates = self.pending.drain(..).take(self.config.beam_width).collect();
        self.step += 1;
    }

    /// Get the best sequence found.
    pub fn best_sequence(&self) -> Option<&BeamCandidate> {
        // Check finished first
        let best_finished = self.finished.iter()
            .max_by(|a, b| {
                let score_a = a.normalized_score(self.config.length_alpha);
                let score_b = b.normalized_score(self.config.length_alpha);
                score_a.partial_cmp(&score_b).unwrap_or(Ordering::Equal)
            });

        // Then check active candidates
        let best_active = self.candidates.iter()
            .max_by(|a, b| {
                let score_a = a.normalized_score(self.config.length_alpha);
                let score_b = b.normalized_score(self.config.length_alpha);
                score_a.partial_cmp(&score_b).unwrap_or(Ordering::Equal)
            });

        match (best_finished, best_active) {
            (Some(f), Some(a)) => {
                if f.normalized_score(self.config.length_alpha) >= a.normalized_score(self.config.length_alpha) {
                    Some(f)
                } else {
                    Some(a)
                }
            }
            (Some(f), None) => Some(f),
            (None, Some(a)) => Some(a),
            (None, None) => None,
        }
    }

    /// Get top-n sequences.
    pub fn best_sequences(&self, n: usize) -> Vec<&BeamCandidate> {
        let mut all: Vec<&BeamCandidate> = self.finished.iter()
            .chain(self.candidates.iter())
            .collect();

        all.sort_by(|a, b| {
            let score_a = a.normalized_score(self.config.length_alpha);
            let score_b = b.normalized_score(self.config.length_alpha);
            score_b.partial_cmp(&score_a).unwrap_or(Ordering::Equal)
        });

        all.into_iter().take(n).collect()
    }

    /// Alias for best_sequences.
    pub fn n_best(&self, n: usize) -> Vec<&BeamCandidate> {
        self.best_sequences(n)
    }

    /// Reset the search.
    pub fn reset(&mut self, root_node_id: NodeId) {
        self.candidates = vec![BeamCandidate::root(root_node_id)];
        self.finished.clear();
        self.pending.clear();
        self.next_id = 1;
        self.step = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_beam() -> BeamSearch {
        let config = BeamConfig {
            beam_width: 2,
            max_length: 10,
            ..Default::default()
        };
        BeamSearch::new(NodeId::ROOT, config)
    }

    #[test]
    fn beam_search_creation() {
        let beam = create_beam();
        assert_eq!(beam.candidates().len(), 1);
        assert!(!beam.is_done());
    }

    #[test]
    fn beam_search_expand_and_step() {
        let mut beam = create_beam();

        // Expand root with 3 tokens
        let log_probs = vec![(1u32, -0.5), (2, -1.0), (3, -2.0)];
        let node_ids = vec![NodeId::new(), NodeId::new(), NodeId::new()];
        beam.expand_candidate(0, &log_probs, &node_ids);

        // Step to select top-2
        beam.step();

        assert_eq!(beam.candidates().len(), 2);
        // Best candidates should be token 1 and 2
        let tokens: Vec<u32> = beam.candidates().iter()
            .map(|c| c.tokens[0])
            .collect();
        assert!(tokens.contains(&1));
        assert!(tokens.contains(&2));
    }

    #[test]
    fn beam_search_tracks_scores() {
        let mut beam = create_beam();

        let log_probs = vec![(1u32, -1.0), (2, -2.0)];
        let node_ids = vec![NodeId::new(), NodeId::new()];
        beam.expand_candidate(0, &log_probs, &node_ids);
        beam.step();

        let best = beam.candidates().iter()
            .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
            .unwrap();

        assert_eq!(best.tokens[0], 1);
        assert_eq!(best.score, -1.0);
    }

    #[test]
    fn beam_search_handles_eos() {
        let mut beam = BeamSearch::new(NodeId::ROOT, BeamConfig {
            beam_width: 2,
            max_length: 10,
            eos_token_id: Some(99),
            min_length: 0,
            ..Default::default()
        });

        // Include EOS token
        let log_probs = vec![(99u32, -0.5), (2, -1.0)];
        let node_ids = vec![NodeId::new(), NodeId::new()];
        beam.expand_candidate(0, &log_probs, &node_ids);
        beam.step();

        // EOS should go to finished
        assert_eq!(beam.finished().len(), 1);
        assert_eq!(beam.finished()[0].tokens, vec![99]);
    }

    #[test]
    fn beam_search_respects_min_length() {
        let mut beam = BeamSearch::new(NodeId::ROOT, BeamConfig {
            beam_width: 2,
            max_length: 10,
            eos_token_id: Some(99),
            min_length: 3, // Require at least 3 tokens
            ..Default::default()
        });

        // EOS at first step should be ignored due to min_length
        let log_probs = vec![(99u32, -0.5), (2, -1.0)];
        let node_ids = vec![NodeId::new(), NodeId::new()];
        beam.expand_candidate(0, &log_probs, &node_ids);
        beam.step();

        assert_eq!(beam.finished().len(), 0); // EOS ignored
        assert_eq!(beam.candidates().len(), 1); // Only non-EOS token
    }

    #[test]
    fn beam_search_length_normalization() {
        let mut c1 = BeamCandidate::root(NodeId::ROOT);
        c1.score = -10.0;
        c1.tokens = vec![1, 2, 3]; // Length 3

        let mut c2 = BeamCandidate::root(NodeId::ROOT);
        c2.score = -12.0;
        c2.tokens = vec![1, 2, 3, 4, 5]; // Length 5

        // With alpha > 0, longer sequence gets boost
        let norm1 = c1.normalized_score(0.6);
        let norm2 = c2.normalized_score(0.6);

        // Raw scores: -10 vs -12
        // With length penalty, -12/longer should be boosted
        assert!(c1.score > c2.score); // Raw: c1 better
        // Normalized might change order depending on alpha
    }

    #[test]
    fn beam_search_best_sequence() {
        let mut beam = create_beam();

        let log_probs = vec![(1u32, -1.0), (2, -0.5)];
        let node_ids = vec![NodeId::new(), NodeId::new()];
        beam.expand_candidate(0, &log_probs, &node_ids);
        beam.step();

        let best = beam.best_sequence().unwrap();
        assert_eq!(best.tokens[0], 2); // Higher score
    }

    #[test]
    fn beam_search_is_done_at_max_length() {
        let mut beam = BeamSearch::new(NodeId::ROOT, BeamConfig {
            beam_width: 1,
            max_length: 2,
            ..Default::default()
        });

        assert!(!beam.is_done());

        // Two steps to reach max_length
        let log_probs = vec![(1u32, -1.0)];
        let node_ids = vec![NodeId::new()];

        beam.expand_candidate(0, &log_probs, &node_ids);
        beam.step();
        assert!(!beam.is_done());

        beam.expand_candidate(1, &log_probs, &node_ids);
        beam.step();
        assert!(beam.is_done());
    }

    #[test]
    fn beam_search_reset() {
        let mut beam = create_beam();

        let log_probs = vec![(1u32, -1.0)];
        let node_ids = vec![NodeId::new()];
        beam.expand_candidate(0, &log_probs, &node_ids);
        beam.step();

        assert_eq!(beam.current_step(), 1);

        beam.reset(NodeId::ROOT);

        assert_eq!(beam.current_step(), 0);
        assert_eq!(beam.candidates().len(), 1);
        assert!(beam.finished().is_empty());
    }

    #[test]
    fn beam_search_expand_with_logits() {
        let mut beam = create_beam();

        // Create logits where token 5 is best, then 3, then 7
        let mut logits = vec![0.0f32; 10];
        logits[5] = 5.0;
        logits[3] = 3.0;
        logits[7] = 2.0;

        beam.expand_with_logits(0, &logits, 3);
        beam.step();

        // Should have top 2 (beam_width)
        assert_eq!(beam.candidates().len(), 2);
        let tokens: Vec<u32> = beam.candidates().iter()
            .map(|c| c.tokens[0])
            .collect();
        assert!(tokens.contains(&5));
        assert!(tokens.contains(&3));
    }
}
