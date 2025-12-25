//! Scheduling policies.

/// Scheduling policy for request ordering.
#[derive(Debug, Clone, Copy, Default)]
pub enum SchedulingPolicy {
    /// First-come, first-served.
    #[default]
    Fcfs,
    /// Shortest job first (by input length).
    ShortestJobFirst,
    /// Longest job first (by input length).
    LongestJobFirst,
    /// Priority-based scheduling.
    Priority,
}

impl SchedulingPolicy {
    /// Compare two requests according to this policy.
    pub fn compare(&self, a: &super::Request, b: &super::Request) -> std::cmp::Ordering {
        match self {
            Self::Fcfs => a.arrival_time.cmp(&b.arrival_time),
            Self::ShortestJobFirst => a.input_tokens.len().cmp(&b.input_tokens.len()),
            Self::LongestJobFirst => b.input_tokens.len().cmp(&a.input_tokens.len()),
            Self::Priority => b.priority.cmp(&a.priority),
        }
    }
}
