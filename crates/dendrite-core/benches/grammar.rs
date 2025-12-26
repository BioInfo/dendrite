//! Benchmarks for grammar constraint performance.
//!
//! These benchmarks measure the overhead of grammar-constrained decoding,
//! particularly the mask computation time which is critical for achieving
//! the target of <50μs per token.
//!
//! # Key Metrics
//!
//! - Mask computation time: Target <50μs
//! - Constraint state forking (for tree search)
//! - Schema compilation time (one-time cost)
//!
//! # Benchmark Scenarios
//!
//! 1. Simple constraints (regex, short schemas)
//! 2. Complex JSON schemas (nested, with constraints)
//! 3. Forking for tree search
//! 4. llguidance integration (when tokenizer available)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use dendrite_core::grammar::{to_llguidance, Grammar, GrammarConstraint};

/// Sample JSON schemas for benchmarking.
mod schemas {
    pub const SIMPLE_STRING: &str = r#"{"type": "string"}"#;

    pub const PERSON: &str = r#"{
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"]
    }"#;

    pub const NESTED_OBJECT: &str = r#"{
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "profile": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "email": {"type": "string"},
                            "preferences": {
                                "type": "object",
                                "properties": {
                                    "theme": {"enum": ["light", "dark"]},
                                    "notifications": {"type": "boolean"}
                                }
                            }
                        }
                    }
                }
            }
        }
    }"#;

    pub const ARRAY_OF_OBJECTS: &str = r#"{
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "name": {"type": "string"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        }
    }"#;
}

/// Sample regex patterns.
mod patterns {
    pub const SIMPLE_WORD: &str = r"[a-z]+";
    pub const EMAIL: &str = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}";
    pub const UUID: &str = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}";
    pub const ISO_DATE: &str = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z?";
}

/// Benchmark: Grammar creation time.
fn bench_grammar_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("grammar_creation");

    // JSON Schema creation
    group.bench_function("json_simple", |b| {
        b.iter(|| {
            let grammar = Grammar::json_schema(black_box(schemas::SIMPLE_STRING));
            black_box(grammar)
        })
    });

    group.bench_function("json_person", |b| {
        b.iter(|| {
            let grammar = Grammar::json_schema(black_box(schemas::PERSON));
            black_box(grammar)
        })
    });

    group.bench_function("json_nested", |b| {
        b.iter(|| {
            let grammar = Grammar::json_schema(black_box(schemas::NESTED_OBJECT));
            black_box(grammar)
        })
    });

    // Regex creation
    group.bench_function("regex_simple", |b| {
        b.iter(|| {
            let grammar = Grammar::regex(black_box(patterns::SIMPLE_WORD));
            black_box(grammar)
        })
    });

    group.bench_function("regex_email", |b| {
        b.iter(|| {
            let grammar = Grammar::regex(black_box(patterns::EMAIL));
            black_box(grammar)
        })
    });

    group.finish();
}

/// Benchmark: Constraint initialization.
fn bench_constraint_init(c: &mut Criterion) {
    let mut group = c.benchmark_group("constraint_init");

    for (name, schema) in [
        ("simple", schemas::SIMPLE_STRING),
        ("person", schemas::PERSON),
        ("nested", schemas::NESTED_OBJECT),
        ("array", schemas::ARRAY_OF_OBJECTS),
    ] {
        group.bench_with_input(
            BenchmarkId::new("json_schema", name),
            &schema,
            |b, schema| {
                let grammar = Grammar::json_schema(*schema);
                b.iter(|| {
                    let constraint = GrammarConstraint::new(grammar.clone(), 50000).unwrap();
                    black_box(constraint)
                })
            },
        );
    }

    for (name, pattern) in [
        ("word", patterns::SIMPLE_WORD),
        ("email", patterns::EMAIL),
        ("uuid", patterns::UUID),
        ("date", patterns::ISO_DATE),
    ] {
        group.bench_with_input(
            BenchmarkId::new("regex", name),
            &pattern,
            |b, pattern| {
                let grammar = Grammar::regex(*pattern);
                b.iter(|| {
                    let constraint = GrammarConstraint::new(grammar.clone(), 50000).unwrap();
                    black_box(constraint)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark: Mask computation (critical path).
///
/// Target: <50μs per mask computation.
fn bench_mask_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("mask_computation");
    group.throughput(Throughput::Elements(1));

    // Test with different vocabulary sizes
    for vocab_size in [1000, 32000, 50000, 128000] {
        let grammar = Grammar::json_schema(schemas::PERSON);
        let constraint = GrammarConstraint::new(grammar, vocab_size).unwrap();

        group.bench_with_input(
            BenchmarkId::new("vocab_size", vocab_size),
            &vocab_size,
            |b, _| {
                b.iter(|| {
                    let mask = constraint.compute_mask().unwrap();
                    black_box(mask)
                })
            },
        );
    }

    // Test with different schema complexities
    for (name, schema) in [
        ("simple", schemas::SIMPLE_STRING),
        ("person", schemas::PERSON),
        ("nested", schemas::NESTED_OBJECT),
    ] {
        let grammar = Grammar::json_schema(schema);
        let constraint = GrammarConstraint::new(grammar, 50000).unwrap();

        group.bench_with_input(
            BenchmarkId::new("schema", name),
            &name,
            |b, _| {
                b.iter(|| {
                    let mask = constraint.compute_mask().unwrap();
                    black_box(mask)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark: Token acceptance (updating constraint state).
fn bench_token_acceptance(c: &mut Criterion) {
    let mut group = c.benchmark_group("token_acceptance");
    group.throughput(Throughput::Elements(1));

    let grammar = Grammar::json_schema(schemas::PERSON);
    let mut constraint = GrammarConstraint::new(grammar, 50000).unwrap();

    group.bench_function("accept_single", |b| {
        b.iter(|| {
            constraint.accept_token(black_box(42)).unwrap();
        })
    });

    // Accept a sequence of tokens
    group.bench_function("accept_sequence_100", |b| {
        b.iter(|| {
            let grammar = Grammar::json_schema(schemas::PERSON);
            let mut constraint = GrammarConstraint::new(grammar, 50000).unwrap();
            for i in 0..100 {
                constraint.accept_token(black_box(i as u32)).unwrap();
            }
        })
    });

    group.finish();
}

/// Benchmark: Constraint forking for tree search.
fn bench_constraint_fork(c: &mut Criterion) {
    let mut group = c.benchmark_group("constraint_fork");

    for num_tokens in [0, 10, 100, 500] {
        let grammar = Grammar::json_schema(schemas::NESTED_OBJECT);
        let mut constraint = GrammarConstraint::new(grammar, 50000).unwrap();

        // Accept some tokens
        for i in 0..num_tokens {
            constraint.accept_token(i as u32).unwrap();
        }

        group.bench_with_input(
            BenchmarkId::new("tokens", num_tokens),
            &num_tokens,
            |b, _| {
                b.iter(|| {
                    let forked = constraint.fork();
                    black_box(forked)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark: llguidance grammar conversion.
fn bench_llguidance_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("llguidance_conversion");

    for (name, schema) in [
        ("simple", schemas::SIMPLE_STRING),
        ("person", schemas::PERSON),
        ("nested", schemas::NESTED_OBJECT),
    ] {
        let grammar = Grammar::json_schema(schema);

        group.bench_with_input(BenchmarkId::new("to_llg", name), &grammar, |b, grammar| {
            b.iter(|| {
                let llg = to_llguidance(black_box(grammar));
                black_box(llg)
            })
        });
    }

    // Regex conversion
    for (name, pattern) in [("word", patterns::SIMPLE_WORD), ("email", patterns::EMAIL)] {
        let grammar = Grammar::regex(pattern);

        group.bench_with_input(BenchmarkId::new("regex_to_llg", name), &grammar, |b, grammar| {
            b.iter(|| {
                let llg = to_llguidance(black_box(grammar));
                black_box(llg)
            })
        });
    }

    group.finish();
}

/// Benchmark: Multi-branch constraint tracking (MCTS scenario).
fn bench_mcts_constraint_tracking(c: &mut Criterion) {
    let mut group = c.benchmark_group("mcts_constraints");

    for branching_factor in [2, 4, 8] {
        group.bench_with_input(
            BenchmarkId::new("branching", branching_factor),
            &branching_factor,
            |b, &bf| {
                b.iter(|| {
                    let grammar = Grammar::json_schema(schemas::PERSON);
                    let mut root = GrammarConstraint::new(grammar, 50000).unwrap();

                    // Simulate base generation
                    for i in 0..10 {
                        root.accept_token(i as u32).unwrap();
                    }

                    // Fork for MCTS branches
                    let mut branches: Vec<GrammarConstraint> = (0..bf)
                        .map(|_| root.fork())
                        .collect();

                    // Each branch continues independently
                    for (i, branch) in branches.iter_mut().enumerate() {
                        for j in 0..5 {
                            branch.accept_token(((i + 1) * 100 + j) as u32).unwrap();
                        }
                    }

                    black_box(branches)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark: Beam search constraint tracking.
fn bench_beam_constraint_tracking(c: &mut Criterion) {
    let mut group = c.benchmark_group("beam_constraints");

    for beam_width in [4, 8, 16] {
        group.bench_with_input(
            BenchmarkId::new("beam_width", beam_width),
            &beam_width,
            |b, &bw| {
                b.iter(|| {
                    let grammar = Grammar::json_schema(schemas::ARRAY_OF_OBJECTS);
                    let root = GrammarConstraint::new(grammar, 50000).unwrap();

                    // Simulate beam search steps
                    let mut beam: Vec<GrammarConstraint> = (0..bw)
                        .map(|_| root.fork())
                        .collect();

                    // 3 beam search steps
                    for step in 0..3 {
                        // Score and select (simulated)
                        let mut new_beam = Vec::new();
                        for (i, constraint) in beam.iter().enumerate() {
                            if i < bw / 2 {
                                let mut forked = constraint.fork();
                                forked.accept_token((step * 100 + i) as u32).unwrap();
                                new_beam.push(forked);
                            }
                        }
                        beam = new_beam;
                    }

                    black_box(beam)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_grammar_creation,
    bench_constraint_init,
    bench_mask_computation,
    bench_token_acceptance,
    bench_constraint_fork,
    bench_llguidance_conversion,
    bench_mcts_constraint_tracking,
    bench_beam_constraint_tracking,
);

criterion_main!(benches);
