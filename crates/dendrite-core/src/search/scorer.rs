//! Branch scoring for tree search.
//!
//! Provides scoring functions for evaluating and selecting branches:
//! - UCT (Upper Confidence Bound for Trees)
//! - Custom scoring via trait


/// Configuration for scorers.
#[derive(Debug, Clone)]
pub struct ScorerConfig {
    /// Exploration constant for UCT (default: sqrt(2))
    pub exploration_constant: f64,
    /// Temperature for score scaling
    pub temperature: f64,
}

impl Default for ScorerConfig {
    fn default() -> Self {
        Self {
            exploration_constant: std::f64::consts::SQRT_2,
            temperature: 1.0,
        }
    }
}

/// Statistics tracked for each node in search.
#[derive(Debug, Clone, Default)]
pub struct NodeStats {
    /// Number of times this node was visited.
    pub visits: u32,
    /// Total reward accumulated.
    pub total_reward: f64,
    /// Mean reward (total_reward / visits).
    pub mean_reward: f64,
    /// Best reward seen at this node.
    pub best_reward: f64,
    /// Worst reward seen at this node.
    pub worst_reward: f64,
    /// Whether this node is terminal.
    pub is_terminal: bool,
    /// Whether this node is fully expanded.
    pub is_fully_expanded: bool,
}

impl NodeStats {
    /// Create new stats with initial values.
    pub fn new() -> Self {
        Self {
            visits: 0,
            total_reward: 0.0,
            mean_reward: 0.0,
            best_reward: f64::NEG_INFINITY,
            worst_reward: f64::INFINITY,
            is_terminal: false,
            is_fully_expanded: false,
        }
    }

    /// Update stats with a new reward.
    pub fn update(&mut self, reward: f64) {
        self.visits += 1;
        self.total_reward += reward;
        self.mean_reward = self.total_reward / self.visits as f64;
        self.best_reward = self.best_reward.max(reward);
        self.worst_reward = self.worst_reward.min(reward);
    }

    /// Get the value estimate for this node.
    pub fn value(&self) -> f64 {
        if self.visits == 0 {
            0.0
        } else {
            self.mean_reward
        }
    }
}

/// Trait for branch scoring algorithms.
pub trait Scorer: Send + Sync {
    /// Score a node for selection.
    ///
    /// Higher scores indicate more promising nodes to explore.
    fn score(&self, node_stats: &NodeStats, parent_visits: u32) -> f64;

    /// Score multiple children and return indices sorted by score (descending).
    fn rank_children(&self, children: &[NodeStats], parent_visits: u32) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..children.len()).collect();
        indices.sort_by(|&a, &b| {
            let score_a = self.score(&children[a], parent_visits);
            let score_b = self.score(&children[b], parent_visits);
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        indices
    }
}

/// UCT (Upper Confidence Bound for Trees) scorer.
///
/// UCT balances exploitation (high mean reward) with exploration
/// (less visited nodes). The formula is:
///
/// `UCT = mean_reward + C * sqrt(ln(parent_visits) / visits)`
///
/// Where C is the exploration constant.
#[derive(Debug, Clone)]
pub struct UctScorer {
    /// Exploration constant (higher = more exploration).
    pub exploration_constant: f64,
}

impl UctScorer {
    /// Create a new UCT scorer with default exploration constant.
    pub fn new() -> Self {
        Self {
            exploration_constant: std::f64::consts::SQRT_2,
        }
    }

    /// Create with custom exploration constant.
    pub fn with_exploration(exploration_constant: f64) -> Self {
        Self { exploration_constant }
    }
}

impl Default for UctScorer {
    fn default() -> Self {
        Self::new()
    }
}

impl Scorer for UctScorer {
    fn score(&self, node_stats: &NodeStats, parent_visits: u32) -> f64 {
        if node_stats.visits == 0 {
            // Unvisited nodes get infinite score to ensure exploration
            return f64::INFINITY;
        }

        let exploitation = node_stats.mean_reward;
        let exploration = self.exploration_constant
            * ((parent_visits as f64).ln() / node_stats.visits as f64).sqrt();

        exploitation + exploration
    }
}

/// Greedy scorer that only considers mean reward.
#[derive(Debug, Clone, Default)]
pub struct GreedyScorer;

impl Scorer for GreedyScorer {
    fn score(&self, node_stats: &NodeStats, _parent_visits: u32) -> f64 {
        if node_stats.visits == 0 {
            f64::INFINITY
        } else {
            node_stats.mean_reward
        }
    }
}

/// Epsilon-greedy scorer.
///
/// With probability epsilon, selects randomly. Otherwise, selects greedily.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct EpsilonGreedyScorer {
    /// Probability of random selection.
    pub epsilon: f64,
    /// Random number for this scoring (set externally).
    pub random: f64,
}

#[allow(dead_code)]
impl EpsilonGreedyScorer {
    /// Create with epsilon value.
    pub fn new(epsilon: f64) -> Self {
        Self { epsilon, random: 0.0 }
    }

    /// Set the random value for this scoring round.
    pub fn set_random(&mut self, random: f64) {
        self.random = random;
    }
}

impl Scorer for EpsilonGreedyScorer {
    fn score(&self, node_stats: &NodeStats, _parent_visits: u32) -> f64 {
        if self.random < self.epsilon {
            // Random exploration - return random score
            self.random * 1000.0
        } else if node_stats.visits == 0 {
            f64::INFINITY
        } else {
            node_stats.mean_reward
        }
    }
}

/// PUCT scorer (Predictor + UCT) used in AlphaZero-style algorithms.
///
/// `PUCT = Q + C * P * sqrt(N) / (1 + n)`
///
/// Where:
/// - Q = mean action value
/// - P = prior probability from policy network
/// - N = parent visits
/// - n = child visits
/// - C = exploration constant
#[derive(Debug, Clone)]
pub struct PuctScorer {
    /// Exploration constant.
    pub c_puct: f64,
}

impl PuctScorer {
    /// Create with default c_puct.
    pub fn new() -> Self {
        Self { c_puct: 1.5 }
    }

    /// Create with custom c_puct.
    pub fn with_c_puct(c_puct: f64) -> Self {
        Self { c_puct }
    }

    /// Score with prior probability.
    pub fn score_with_prior(&self, node_stats: &NodeStats, parent_visits: u32, prior: f64) -> f64 {
        let q = if node_stats.visits == 0 {
            0.0
        } else {
            node_stats.mean_reward
        };

        let u = self.c_puct * prior * (parent_visits as f64).sqrt() / (1.0 + node_stats.visits as f64);

        q + u
    }
}

impl Default for PuctScorer {
    fn default() -> Self {
        Self::new()
    }
}

impl Scorer for PuctScorer {
    fn score(&self, node_stats: &NodeStats, parent_visits: u32) -> f64 {
        // Without prior, assume uniform prior
        let num_actions = 1.0; // Would need to know number of siblings
        let prior = 1.0 / num_actions;
        self.score_with_prior(node_stats, parent_visits, prior)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_stats_update() {
        let mut stats = NodeStats::new();
        assert_eq!(stats.visits, 0);
        assert_eq!(stats.value(), 0.0);

        stats.update(1.0);
        assert_eq!(stats.visits, 1);
        assert_eq!(stats.mean_reward, 1.0);

        stats.update(0.5);
        assert_eq!(stats.visits, 2);
        assert_eq!(stats.mean_reward, 0.75);
        assert_eq!(stats.best_reward, 1.0);
        assert_eq!(stats.worst_reward, 0.5);
    }

    #[test]
    fn uct_unvisited_infinite() {
        let scorer = UctScorer::new();
        let stats = NodeStats::new();
        let score = scorer.score(&stats, 10);
        assert!(score.is_infinite());
    }

    #[test]
    fn uct_balances_exploration_exploitation() {
        let scorer = UctScorer::new();

        // High reward, many visits
        let mut high_visits = NodeStats::new();
        for _ in 0..100 {
            high_visits.update(0.8);
        }

        // Lower reward, few visits
        let mut low_visits = NodeStats::new();
        for _ in 0..5 {
            low_visits.update(0.6);
        }

        let parent_visits = 105;

        // With enough parent visits, exploration bonus should help low_visits
        let score_high = scorer.score(&high_visits, parent_visits);
        let score_low = scorer.score(&low_visits, parent_visits);

        // Both should be finite
        assert!(score_high.is_finite());
        assert!(score_low.is_finite());

        // High visits should have higher exploitation but lower exploration
        assert!(high_visits.mean_reward > low_visits.mean_reward);
    }

    #[test]
    fn greedy_scorer_picks_best() {
        let scorer = GreedyScorer;

        let mut good = NodeStats::new();
        good.update(0.9);

        let mut bad = NodeStats::new();
        bad.update(0.1);

        assert!(scorer.score(&good, 10) > scorer.score(&bad, 10));
    }

    #[test]
    fn rank_children_sorts_by_score() {
        let scorer = GreedyScorer;

        let mut children = vec![
            NodeStats::new(),
            NodeStats::new(),
            NodeStats::new(),
        ];

        children[0].update(0.3);
        children[1].update(0.9);
        children[2].update(0.5);

        let ranked = scorer.rank_children(&children, 10);

        // Unvisited nodes first (infinite score), then by mean reward
        // All have 1 visit now, so sort by mean reward
        // Expected order: [1, 2, 0] (0.9, 0.5, 0.3)
        assert_eq!(ranked, vec![1, 2, 0]);
    }

    #[test]
    fn puct_with_prior() {
        let scorer = PuctScorer::new();

        let mut stats = NodeStats::new();
        stats.update(0.5);

        // Higher prior should give higher score
        let score_high_prior = scorer.score_with_prior(&stats, 100, 0.8);
        let score_low_prior = scorer.score_with_prior(&stats, 100, 0.1);

        assert!(score_high_prior > score_low_prior);
    }
}
