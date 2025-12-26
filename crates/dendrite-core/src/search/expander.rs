//! Tree expansion for generating child nodes.
//!
//! Expanders generate new branches from a parent node by:
//! 1. Getting candidate actions (tokens)
//! 2. Creating child nodes for each action
//! 3. Optionally running the model to get logits

use crate::error::Result;
use crate::tree::NodeId;

/// Result of expanding a node.
#[derive(Debug, Clone)]
pub struct ExpansionResult {
    /// Child node IDs created.
    pub children: Vec<NodeId>,
    /// Actions (tokens) corresponding to each child.
    pub actions: Vec<u32>,
    /// Log probabilities for each action (if available).
    pub log_probs: Option<Vec<f32>>,
    /// Prior probabilities for each action (for PUCT).
    pub priors: Option<Vec<f32>>,
}

impl ExpansionResult {
    /// Create an empty expansion result.
    pub fn empty() -> Self {
        Self {
            children: Vec::new(),
            actions: Vec::new(),
            log_probs: None,
            priors: None,
        }
    }

    /// Number of actions to expand.
    pub fn len(&self) -> usize {
        self.actions.len()
    }

    /// Check if no actions to expand.
    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }
}

/// Configuration for token-based expansion.
#[derive(Debug, Clone)]
pub struct TokenExpanderConfig {
    /// Maximum number of children to create per expansion.
    pub max_children: usize,
    /// Minimum probability threshold for a token to be considered.
    pub min_prob: f32,
    /// Temperature for sampling (0 = greedy top-k).
    pub temperature: f32,
    /// Whether to use nucleus (top-p) sampling.
    pub top_p: Option<f32>,
}

impl Default for TokenExpanderConfig {
    fn default() -> Self {
        Self {
            max_children: 5,
            min_prob: 0.01,
            temperature: 1.0,
            top_p: Some(0.9),
        }
    }
}

/// Trait for tree expansion strategies.
pub trait Expander: Send + Sync {
    /// Expand a node, creating child nodes for possible actions.
    ///
    /// # Arguments
    ///
    /// * `node_id` - The node to expand
    /// * `logits` - Model logits for action selection (optional)
    ///
    /// # Returns
    ///
    /// ExpansionResult containing the created children and their actions.
    fn expand(&self, node_id: NodeId, logits: Option<&[f32]>) -> Result<ExpansionResult>;

    /// Get the maximum number of children this expander can create.
    fn max_children(&self) -> usize;
}

/// Token-based expander using model logits.
///
/// Selects top-k tokens based on logits and creates a child node for each.
#[derive(Debug, Clone)]
pub struct TokenExpander {
    /// Configuration.
    config: TokenExpanderConfig,
    /// Vocabulary size.
    vocab_size: usize,
}

impl TokenExpander {
    /// Create a new token expander.
    pub fn new(vocab_size: usize, config: TokenExpanderConfig) -> Self {
        Self { config, vocab_size }
    }

    /// Create with default config.
    pub fn with_vocab_size(vocab_size: usize) -> Self {
        Self::new(vocab_size, TokenExpanderConfig::default())
    }

    /// Select top-k tokens from logits.
    pub fn select_tokens(&self, logits: &[f32]) -> Vec<(u32, f32)> {
        // Apply temperature
        let scaled_logits: Vec<f32> = if self.config.temperature != 1.0 {
            logits.iter().map(|&l| l / self.config.temperature).collect()
        } else {
            logits.to_vec()
        };

        // Compute softmax
        let max_logit = scaled_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_logits: Vec<f32> = scaled_logits.iter().map(|&l| (l - max_logit).exp()).collect();
        let sum_exp: f32 = exp_logits.iter().sum();
        let probs: Vec<f32> = exp_logits.iter().map(|&e| e / sum_exp).collect();

        // Get top-k by probability
        let mut indexed_probs: Vec<(u32, f32)> = probs
            .iter()
            .enumerate()
            .map(|(i, &p)| (i as u32, p))
            .collect();
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Apply top-p (nucleus sampling) if configured
        let selected = if let Some(top_p) = self.config.top_p {
            let mut cumsum = 0.0;
            indexed_probs
                .into_iter()
                .take_while(|(_, p)| {
                    let include = cumsum < top_p;
                    cumsum += p;
                    include || cumsum <= top_p
                })
                .take(self.config.max_children)
                .filter(|(_, p)| *p >= self.config.min_prob)
                .collect()
        } else {
            indexed_probs
                .into_iter()
                .take(self.config.max_children)
                .filter(|(_, p)| *p >= self.config.min_prob)
                .collect()
        };

        selected
    }
}

impl Expander for TokenExpander {
    fn expand(&self, _node_id: NodeId, logits: Option<&[f32]>) -> Result<ExpansionResult> {
        let logits = match logits {
            Some(l) => l,
            None => {
                // No logits provided - return empty
                return Ok(ExpansionResult::empty());
            }
        };

        if logits.len() != self.vocab_size {
            return Err(crate::error::DendriteError::ShapeMismatch(format!(
                "Expected vocab_size {}, got {}",
                self.vocab_size,
                logits.len()
            )));
        }

        let selected = self.select_tokens(logits);

        let actions: Vec<u32> = selected.iter().map(|(t, _)| *t).collect();
        let priors: Vec<f32> = selected.iter().map(|(_, p)| *p).collect();
        let log_probs: Vec<f32> = priors.iter().map(|p| p.ln()).collect();

        // Note: Actual node creation would be done by the search algorithm
        // using TreeState. Here we just return the actions.
        Ok(ExpansionResult {
            children: Vec::new(), // Filled in by search algorithm
            actions,
            log_probs: Some(log_probs),
            priors: Some(priors),
        })
    }

    fn max_children(&self) -> usize {
        self.config.max_children
    }
}

/// Uniform expander that creates children for specified actions.
///
/// Useful for testing or when actions are predetermined.
#[derive(Debug, Clone)]
pub struct UniformExpander {
    /// Fixed actions to expand.
    actions: Vec<u32>,
}

impl UniformExpander {
    /// Create with fixed actions.
    pub fn new(actions: Vec<u32>) -> Self {
        Self { actions }
    }

    /// Create for a range of tokens.
    pub fn from_range(start: u32, end: u32) -> Self {
        Self {
            actions: (start..end).collect(),
        }
    }
}

impl Expander for UniformExpander {
    fn expand(&self, _node_id: NodeId, _logits: Option<&[f32]>) -> Result<ExpansionResult> {
        let n = self.actions.len();
        let uniform_prior = 1.0 / n as f32;

        Ok(ExpansionResult {
            children: Vec::new(),
            actions: self.actions.clone(),
            log_probs: Some(vec![uniform_prior.ln(); n]),
            priors: Some(vec![uniform_prior; n]),
        })
    }

    fn max_children(&self) -> usize {
        self.actions.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_expander_selects_top_k() {
        let expander = TokenExpander::new(10, TokenExpanderConfig {
            max_children: 3,
            min_prob: 0.0,
            temperature: 1.0,
            top_p: None,
        });

        // Logits where token 5 is highest, then 3, then 7
        let logits = [0.0, 0.0, 0.0, 2.0, 0.0, 5.0, 0.0, 1.5, 0.0, 0.0];

        let selected = expander.select_tokens(&logits);

        assert_eq!(selected.len(), 3);
        assert_eq!(selected[0].0, 5); // Highest logit
        assert_eq!(selected[1].0, 3); // Second highest
        assert_eq!(selected[2].0, 7); // Third highest
    }

    #[test]
    fn token_expander_respects_min_prob() {
        let expander = TokenExpander::new(10, TokenExpanderConfig {
            max_children: 10,
            min_prob: 0.1,
            temperature: 1.0,
            top_p: None,
        });

        // Only one very high logit, rest are very low
        let mut logits = [-10.0f32; 10];
        logits[5] = 10.0;

        let selected = expander.select_tokens(&logits);

        // Only token 5 should have prob > 0.1
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].0, 5);
    }

    #[test]
    fn token_expander_applies_temperature() {
        let hot_expander = TokenExpander::new(10, TokenExpanderConfig {
            max_children: 10,
            min_prob: 0.0,
            temperature: 2.0, // High temperature = more uniform
            top_p: None,
        });

        let cold_expander = TokenExpander::new(10, TokenExpanderConfig {
            max_children: 10,
            min_prob: 0.0,
            temperature: 0.5, // Low temperature = more peaked
            top_p: None,
        });

        let logits = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let hot_selected = hot_expander.select_tokens(&logits);
        let cold_selected = cold_expander.select_tokens(&logits);

        // With high temperature, top prob should be lower (more uniform)
        // With low temperature, top prob should be higher (more peaked)
        assert!(cold_selected[0].1 > hot_selected[0].1);
    }

    #[test]
    fn token_expander_applies_top_p() {
        let expander = TokenExpander::new(10, TokenExpanderConfig {
            max_children: 10,
            min_prob: 0.0,
            temperature: 1.0,
            top_p: Some(0.5),
        });

        // Roughly uniform logits
        let logits = [1.0f32; 10];

        let selected = expander.select_tokens(&logits);

        // With top_p=0.5, should select about half
        assert!(selected.len() <= 6);
    }

    #[test]
    fn uniform_expander_creates_uniform_priors() {
        let expander = UniformExpander::new(vec![1, 2, 3, 4]);

        let result = expander.expand(NodeId::ROOT, None).unwrap();

        assert_eq!(result.actions, vec![1, 2, 3, 4]);
        assert_eq!(result.priors.unwrap(), vec![0.25, 0.25, 0.25, 0.25]);
    }

    #[test]
    fn expansion_result_empty() {
        let result = ExpansionResult::empty();
        assert!(result.is_empty());
        assert_eq!(result.len(), 0);
    }
}
