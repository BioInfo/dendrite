//! Golden token test harness for model validation.
//!
//! This module provides infrastructure for validating model outputs against
//! known-good token sequences. Golden tests ensure deterministic correctness
//! across code changes, platforms, and precision formats.
//!
//! # Architecture
//!
//! The harness consists of:
//! - [`GoldenCase`] - A single test case with input/expected output
//! - [`GoldenTestHarness`] - Collection of test cases with validation
//! - [`GoldenResult`] - Comparison result with detailed metrics
//!
//! # Example
//!
//! ```rust,ignore
//! use dendrite_core::model::{GoldenTestHarness, GoldenCase};
//!
//! // Create test harness
//! let mut harness = GoldenTestHarness::new();
//!
//! // Add golden cases
//! harness.add_case(GoldenCase {
//!     name: "simple_greeting".to_string(),
//!     input_tokens: vec![1, 2, 3],  // "Hello"
//!     expected_tokens: vec![4, 5],   // "Hi there"
//!     temperature: 0.0,  // Greedy decoding
//!     ..Default::default()
//! });
//!
//! // Run validation
//! let results = harness.validate(&model, max_tokens)?;
//!
//! // Check all passed
//! assert!(results.all_passed());
//! ```
//!
//! # Determinism
//!
//! For golden tests to be reliable, sampling must be deterministic:
//! - Use `temperature = 0.0` for greedy decoding
//! - Or use fixed seeds with `top_k = 1`
//! - Results should match across platforms when using FP32

use std::collections::HashMap;
use std::fmt;

/// A single golden test case.
#[derive(Debug, Clone)]
pub struct GoldenCase {
    /// Test case name for identification.
    pub name: String,
    /// Input token sequence.
    pub input_tokens: Vec<u32>,
    /// Expected output token sequence.
    pub expected_tokens: Vec<u32>,
    /// Temperature for sampling (0.0 for greedy).
    pub temperature: f32,
    /// Top-k for sampling (0 for all).
    pub top_k: usize,
    /// Top-p for nucleus sampling (1.0 for no filtering).
    pub top_p: f32,
    /// Maximum tokens to generate.
    pub max_tokens: usize,
    /// Allowed tolerance for floating-point comparison (logits).
    pub tolerance: f32,
    /// Optional description.
    pub description: Option<String>,
    /// Tags for filtering tests.
    pub tags: Vec<String>,
}

impl Default for GoldenCase {
    fn default() -> Self {
        Self {
            name: String::new(),
            input_tokens: Vec::new(),
            expected_tokens: Vec::new(),
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            max_tokens: 10,
            tolerance: 1e-5,
            description: None,
            tags: Vec::new(),
        }
    }
}

impl GoldenCase {
    /// Create a new greedy decoding test case.
    pub fn greedy(name: impl Into<String>, input: Vec<u32>, expected: Vec<u32>) -> Self {
        Self {
            name: name.into(),
            input_tokens: input,
            expected_tokens: expected,
            temperature: 0.0,
            top_k: 1,
            ..Default::default()
        }
    }

    /// Create a case from token strings (for readability).
    ///
    /// Tokens are space-separated decimal numbers.
    pub fn from_str(
        name: impl Into<String>,
        input: &str,
        expected: &str,
    ) -> Self {
        let parse_tokens = |s: &str| -> Vec<u32> {
            s.split_whitespace()
                .filter_map(|t| t.parse().ok())
                .collect()
        };

        Self {
            name: name.into(),
            input_tokens: parse_tokens(input),
            expected_tokens: parse_tokens(expected),
            temperature: 0.0,
            top_k: 1,
            ..Default::default()
        }
    }

    /// Add a tag to the test case.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Set description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set max tokens.
    pub fn with_max_tokens(mut self, max: usize) -> Self {
        self.max_tokens = max;
        self
    }
}

/// Result of a single golden test.
#[derive(Debug, Clone)]
pub struct GoldenResult {
    /// Test case name.
    pub name: String,
    /// Whether the test passed.
    pub passed: bool,
    /// Expected tokens.
    pub expected: Vec<u32>,
    /// Actual tokens produced.
    pub actual: Vec<u32>,
    /// Token-by-token comparison.
    pub token_matches: Vec<bool>,
    /// First divergence index (if any).
    pub first_divergence: Option<usize>,
    /// Error message (if failed).
    pub error: Option<String>,
    /// Execution time in milliseconds.
    pub time_ms: f64,
}

impl GoldenResult {
    /// Create a passed result.
    pub fn passed(name: String, expected: Vec<u32>, actual: Vec<u32>, time_ms: f64) -> Self {
        let len = expected.len().min(actual.len());
        let token_matches: Vec<bool> = (0..len)
            .map(|i| expected[i] == actual[i])
            .collect();

        Self {
            name,
            passed: true,
            expected,
            actual,
            token_matches,
            first_divergence: None,
            error: None,
            time_ms,
        }
    }

    /// Create a failed result.
    pub fn failed(
        name: String,
        expected: Vec<u32>,
        actual: Vec<u32>,
        time_ms: f64,
    ) -> Self {
        let len = expected.len().max(actual.len());
        let mut token_matches = Vec::with_capacity(len);
        let mut first_divergence = None;

        for i in 0..len {
            let exp = expected.get(i);
            let act = actual.get(i);
            let matches = exp == act;
            token_matches.push(matches);

            if !matches && first_divergence.is_none() {
                first_divergence = Some(i);
            }
        }

        let error = if let Some(idx) = first_divergence {
            let exp = expected.get(idx).map(|t| format!("{}", t)).unwrap_or_else(|| "EOF".to_string());
            let act = actual.get(idx).map(|t| format!("{}", t)).unwrap_or_else(|| "EOF".to_string());
            Some(format!(
                "Divergence at index {}: expected {}, got {}",
                idx, exp, act
            ))
        } else {
            None
        };

        Self {
            name,
            passed: false,
            expected,
            actual,
            token_matches,
            first_divergence,
            error,
            time_ms,
        }
    }

    /// Create an error result.
    pub fn error(name: String, error: String) -> Self {
        Self {
            name,
            passed: false,
            expected: Vec::new(),
            actual: Vec::new(),
            token_matches: Vec::new(),
            first_divergence: None,
            error: Some(error),
            time_ms: 0.0,
        }
    }

    /// Calculate match percentage.
    pub fn match_percentage(&self) -> f32 {
        if self.token_matches.is_empty() {
            return 0.0;
        }
        let matches = self.token_matches.iter().filter(|&&m| m).count();
        (matches as f32 / self.token_matches.len() as f32) * 100.0
    }
}

impl fmt::Display for GoldenResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = if self.passed { "PASS" } else { "FAIL" };
        write!(f, "[{}] {} ({:.1}ms)", status, self.name, self.time_ms)?;

        if !self.passed {
            if let Some(ref err) = self.error {
                write!(f, "\n  Error: {}", err)?;
            }
            if self.first_divergence.is_some() {
                write!(
                    f,
                    "\n  Match rate: {:.1}%",
                    self.match_percentage()
                )?;
            }
        }

        Ok(())
    }
}

/// Summary of golden test results.
#[derive(Debug, Clone, Default)]
pub struct GoldenSummary {
    /// Total number of tests.
    pub total: usize,
    /// Number of passed tests.
    pub passed: usize,
    /// Number of failed tests.
    pub failed: usize,
    /// Total execution time in milliseconds.
    pub total_time_ms: f64,
    /// Results by name.
    pub results: HashMap<String, GoldenResult>,
}

impl GoldenSummary {
    /// Create from results.
    pub fn from_results(results: Vec<GoldenResult>) -> Self {
        let total = results.len();
        let passed = results.iter().filter(|r| r.passed).count();
        let failed = total - passed;
        let total_time_ms = results.iter().map(|r| r.time_ms).sum();

        let results_map: HashMap<String, GoldenResult> = results
            .into_iter()
            .map(|r| (r.name.clone(), r))
            .collect();

        Self {
            total,
            passed,
            failed,
            total_time_ms,
            results: results_map,
        }
    }

    /// Check if all tests passed.
    pub fn all_passed(&self) -> bool {
        self.failed == 0
    }

    /// Get failure rate.
    pub fn failure_rate(&self) -> f32 {
        if self.total == 0 {
            return 0.0;
        }
        (self.failed as f32 / self.total as f32) * 100.0
    }
}

impl fmt::Display for GoldenSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Golden Test Summary")?;
        writeln!(f, "===================")?;
        writeln!(f, "Total:  {}", self.total)?;
        writeln!(f, "Passed: {} ({:.1}%)", self.passed,
            if self.total > 0 { (self.passed as f32 / self.total as f32) * 100.0 } else { 0.0 })?;
        writeln!(f, "Failed: {} ({:.1}%)", self.failed, self.failure_rate())?;
        writeln!(f, "Time:   {:.1}ms", self.total_time_ms)?;

        if self.failed > 0 {
            writeln!(f, "\nFailed tests:")?;
            for result in self.results.values() {
                if !result.passed {
                    writeln!(f, "  - {}", result)?;
                }
            }
        }

        Ok(())
    }
}

/// Golden test harness for model validation.
#[derive(Debug, Clone, Default)]
pub struct GoldenTestHarness {
    /// Test cases.
    cases: Vec<GoldenCase>,
    /// Strict mode - fail on first mismatch.
    strict: bool,
}

impl GoldenTestHarness {
    /// Create a new test harness.
    pub fn new() -> Self {
        Self {
            cases: Vec::new(),
            strict: false,
        }
    }

    /// Enable strict mode.
    pub fn strict(mut self) -> Self {
        self.strict = true;
        self
    }

    /// Add a test case.
    pub fn add_case(&mut self, case: GoldenCase) {
        self.cases.push(case);
    }

    /// Add multiple test cases.
    pub fn add_cases(&mut self, cases: impl IntoIterator<Item = GoldenCase>) {
        self.cases.extend(cases);
    }

    /// Get all test cases.
    pub fn cases(&self) -> &[GoldenCase] {
        &self.cases
    }

    /// Filter cases by tag.
    pub fn filter_by_tag(&self, tag: &str) -> Vec<&GoldenCase> {
        self.cases
            .iter()
            .filter(|c| c.tags.contains(&tag.to_string()))
            .collect()
    }

    /// Compare tokens for a single case.
    ///
    /// This is the core comparison logic, usable without a full model.
    pub fn compare(&self, case: &GoldenCase, actual: &[u32]) -> GoldenResult {
        let expected = &case.expected_tokens;

        if expected == actual {
            GoldenResult::passed(
                case.name.clone(),
                expected.clone(),
                actual.to_vec(),
                0.0,
            )
        } else {
            GoldenResult::failed(
                case.name.clone(),
                expected.clone(),
                actual.to_vec(),
                0.0,
            )
        }
    }

    /// Compare with prefix matching.
    ///
    /// Checks if actual output starts with expected tokens.
    pub fn compare_prefix(&self, case: &GoldenCase, actual: &[u32]) -> GoldenResult {
        let expected = &case.expected_tokens;

        let prefix_matches = actual
            .iter()
            .zip(expected.iter())
            .all(|(a, e)| a == e);

        if prefix_matches && actual.len() >= expected.len() {
            GoldenResult::passed(
                case.name.clone(),
                expected.clone(),
                actual.to_vec(),
                0.0,
            )
        } else {
            GoldenResult::failed(
                case.name.clone(),
                expected.clone(),
                actual.to_vec(),
                0.0,
            )
        }
    }

    /// Run all cases with a custom generator function.
    ///
    /// This is the main validation method. The generator function takes
    /// input tokens and returns generated output tokens.
    pub fn run_with<F>(&self, mut generator: F) -> GoldenSummary
    where
        F: FnMut(&[u32], usize) -> Result<Vec<u32>, String>,
    {
        let mut results = Vec::with_capacity(self.cases.len());

        for case in &self.cases {
            let start = std::time::Instant::now();

            let result = match generator(&case.input_tokens, case.max_tokens) {
                Ok(actual) => {
                    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

                    if case.expected_tokens == actual {
                        GoldenResult::passed(
                            case.name.clone(),
                            case.expected_tokens.clone(),
                            actual,
                            elapsed,
                        )
                    } else {
                        GoldenResult::failed(
                            case.name.clone(),
                            case.expected_tokens.clone(),
                            actual,
                            elapsed,
                        )
                    }
                }
                Err(err) => GoldenResult::error(case.name.clone(), err),
            };

            results.push(result);
        }

        GoldenSummary::from_results(results)
    }

    /// Create standard test cases for common patterns.
    pub fn standard_cases() -> Self {
        let mut harness = Self::new();

        // Add some basic test patterns
        harness.add_case(
            GoldenCase::greedy("empty_input", vec![], vec![])
                .with_description("Empty input should produce empty output")
                .with_tag("basic"),
        );

        harness.add_case(
            GoldenCase::greedy("single_token", vec![1], vec![])
                .with_description("Single token input")
                .with_tag("basic"),
        );

        harness.add_case(
            GoldenCase::greedy("short_sequence", vec![1, 2, 3], vec![])
                .with_description("Short input sequence")
                .with_tag("basic"),
        );

        harness
    }
}

/// Trait for models that support golden testing.
pub trait GoldenTestable {
    /// Generate tokens from input.
    fn generate(&mut self, input: &[u32], max_tokens: usize) -> crate::Result<Vec<u32>>;

    /// Run golden tests against this model.
    fn run_golden_tests(&mut self, harness: &GoldenTestHarness) -> GoldenSummary {
        harness.run_with(|input, max_tokens| {
            self.generate(input, max_tokens)
                .map_err(|e| e.to_string())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golden_case_creation() {
        let case = GoldenCase::greedy("test", vec![1, 2, 3], vec![4, 5]);
        assert_eq!(case.name, "test");
        assert_eq!(case.input_tokens, vec![1, 2, 3]);
        assert_eq!(case.expected_tokens, vec![4, 5]);
        assert_eq!(case.temperature, 0.0);
    }

    #[test]
    fn golden_case_from_str() {
        let case = GoldenCase::from_str("test", "1 2 3", "4 5");
        assert_eq!(case.input_tokens, vec![1, 2, 3]);
        assert_eq!(case.expected_tokens, vec![4, 5]);
    }

    #[test]
    fn golden_case_with_tags() {
        let case = GoldenCase::greedy("test", vec![], vec![])
            .with_tag("basic")
            .with_tag("regression");

        assert!(case.tags.contains(&"basic".to_string()));
        assert!(case.tags.contains(&"regression".to_string()));
    }

    #[test]
    fn golden_result_passed() {
        let result = GoldenResult::passed(
            "test".to_string(),
            vec![1, 2, 3],
            vec![1, 2, 3],
            10.0,
        );

        assert!(result.passed);
        assert!(result.error.is_none());
        assert_eq!(result.match_percentage(), 100.0);
    }

    #[test]
    fn golden_result_failed() {
        let result = GoldenResult::failed(
            "test".to_string(),
            vec![1, 2, 3],
            vec![1, 9, 3],  // Different at index 1
            10.0,
        );

        assert!(!result.passed);
        assert_eq!(result.first_divergence, Some(1));
        assert!(result.error.is_some());
    }

    #[test]
    fn golden_result_length_mismatch() {
        let result = GoldenResult::failed(
            "test".to_string(),
            vec![1, 2, 3],
            vec![1, 2],  // Too short
            10.0,
        );

        assert!(!result.passed);
        assert_eq!(result.first_divergence, Some(2));
    }

    #[test]
    fn golden_harness_compare_exact() {
        let harness = GoldenTestHarness::new();
        let case = GoldenCase::greedy("test", vec![1], vec![2, 3, 4]);

        let result = harness.compare(&case, &[2, 3, 4]);
        assert!(result.passed);

        let result = harness.compare(&case, &[2, 3, 5]);
        assert!(!result.passed);
    }

    #[test]
    fn golden_harness_compare_prefix() {
        let harness = GoldenTestHarness::new();
        let case = GoldenCase::greedy("test", vec![1], vec![2, 3]);

        // Prefix match - actual is longer but starts with expected
        let result = harness.compare_prefix(&case, &[2, 3, 4, 5]);
        assert!(result.passed);

        // Prefix mismatch
        let result = harness.compare_prefix(&case, &[2, 9, 4]);
        assert!(!result.passed);

        // Too short
        let result = harness.compare_prefix(&case, &[2]);
        assert!(!result.passed);
    }

    #[test]
    fn golden_harness_run_with() {
        let mut harness = GoldenTestHarness::new();

        harness.add_case(GoldenCase::greedy("pass", vec![1], vec![10, 20]));
        harness.add_case(GoldenCase::greedy("fail", vec![2], vec![30, 40]));

        // Mock generator that returns input[0] * 10, input[0] * 20
        let summary = harness.run_with(|input, _max| {
            if input.is_empty() {
                return Ok(vec![]);
            }
            let base = input[0];
            Ok(vec![base * 10, base * 20])
        });

        assert_eq!(summary.total, 2);
        assert_eq!(summary.passed, 1);
        assert_eq!(summary.failed, 1);
        assert!(summary.results.get("pass").unwrap().passed);
        assert!(!summary.results.get("fail").unwrap().passed);
    }

    #[test]
    fn golden_harness_filter_by_tag() {
        let mut harness = GoldenTestHarness::new();

        harness.add_case(
            GoldenCase::greedy("test1", vec![], vec![])
                .with_tag("regression"),
        );
        harness.add_case(
            GoldenCase::greedy("test2", vec![], vec![])
                .with_tag("basic"),
        );
        harness.add_case(
            GoldenCase::greedy("test3", vec![], vec![])
                .with_tag("regression")
                .with_tag("basic"),
        );

        let regression = harness.filter_by_tag("regression");
        assert_eq!(regression.len(), 2);

        let basic = harness.filter_by_tag("basic");
        assert_eq!(basic.len(), 2);
    }

    #[test]
    fn golden_summary_all_passed() {
        let results = vec![
            GoldenResult::passed("a".to_string(), vec![], vec![], 1.0),
            GoldenResult::passed("b".to_string(), vec![], vec![], 2.0),
        ];

        let summary = GoldenSummary::from_results(results);

        assert!(summary.all_passed());
        assert_eq!(summary.total, 2);
        assert_eq!(summary.passed, 2);
        assert_eq!(summary.failed, 0);
        assert_eq!(summary.total_time_ms, 3.0);
    }

    #[test]
    fn golden_summary_with_failures() {
        let results = vec![
            GoldenResult::passed("a".to_string(), vec![], vec![], 1.0),
            GoldenResult::failed("b".to_string(), vec![1], vec![2], 2.0),
        ];

        let summary = GoldenSummary::from_results(results);

        assert!(!summary.all_passed());
        assert_eq!(summary.total, 2);
        assert_eq!(summary.passed, 1);
        assert_eq!(summary.failed, 1);
        assert_eq!(summary.failure_rate(), 50.0);
    }

    #[test]
    fn golden_result_display() {
        let result = GoldenResult::passed(
            "test".to_string(),
            vec![1, 2, 3],
            vec![1, 2, 3],
            5.5,
        );

        let display = format!("{}", result);
        assert!(display.contains("PASS"));
        assert!(display.contains("test"));
        assert!(display.contains("5.5"));
    }

    #[test]
    fn standard_cases_exist() {
        let harness = GoldenTestHarness::standard_cases();
        assert!(!harness.cases().is_empty());

        let basic = harness.filter_by_tag("basic");
        assert!(!basic.is_empty());
    }

    #[test]
    fn golden_case_builder() {
        let case = GoldenCase::greedy("test", vec![1], vec![2])
            .with_max_tokens(50)
            .with_description("A test case")
            .with_tag("important");

        assert_eq!(case.max_tokens, 50);
        assert_eq!(case.description, Some("A test case".to_string()));
        assert!(case.tags.contains(&"important".to_string()));
    }
}
