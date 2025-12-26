//! Example: Grammar-Constrained JSON Generation
//!
//! This example demonstrates how to use Dendrite's grammar constraint system
//! for generating valid JSON output. The grammar module integrates with
//! llguidance for structured output generation.
//!
//! # Architecture
//!
//! Grammar-constrained generation works by:
//! 1. Defining a JSON schema or regex pattern
//! 2. Creating a constraint that tracks valid tokens
//! 3. Masking logits to only allow valid tokens
//! 4. Sampling from the masked distribution
//!
//! # Running
//!
//! ```bash
//! cargo run --example json_generation
//! ```

use dendrite_core::grammar::{Grammar, GrammarConstraint};

/// Example JSON schemas for different use cases.
mod schemas {
    /// Simple key-value object.
    pub const PERSON: &str = r#"{
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "email": {"type": "string", "format": "email"}
        },
        "required": ["name", "age"]
    }"#;

    /// Array of items.
    pub const SHOPPING_LIST: &str = r#"{
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "item": {"type": "string"},
                "quantity": {"type": "integer"},
                "priority": {"enum": ["low", "medium", "high"]}
            },
            "required": ["item", "quantity"]
        }
    }"#;

    /// Nested object with constraints.
    pub const API_RESPONSE: &str = r#"{
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "data": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "created_at": {"type": "string", "format": "date-time"},
                    "tags": {"type": "array", "items": {"type": "string"}}
                }
            },
            "error": {"type": "string"}
        },
        "required": ["success"]
    }"#;
}

/// Demonstrates basic JSON schema constraint creation.
fn demo_schema_creation() {
    println!("=== JSON Schema Constraint Creation ===\n");

    // Create a constraint from a JSON schema
    let grammar = Grammar::json_schema(schemas::PERSON);
    let constraint = GrammarConstraint::new(grammar, 50000).unwrap();

    println!("Created constraint for Person schema");
    println!("  Vocabulary size: {}", constraint.vocab_size());
    println!("  Grammar type: {:?}", constraint.grammar());
    println!();

    // Create from regex for simpler patterns
    let grammar = Grammar::regex(r#"\{"name": "[a-z]+"\}"#);
    let _constraint = GrammarConstraint::new(grammar, 50000).unwrap();

    println!("Created regex constraint for simple JSON");
    println!("  Pattern: {{\"name\": \"[a-z]+\"}}");
    println!();
}

/// Demonstrates constraint state management.
fn demo_state_management() {
    println!("=== Constraint State Management ===\n");

    let grammar = Grammar::json_schema(schemas::PERSON);
    let mut constraint = GrammarConstraint::new(grammar, 50000).unwrap();

    println!("Initial state:");
    println!("  Valid: {}", constraint.is_valid());
    println!("  Complete: {}", constraint.is_complete());

    // Simulate accepting tokens
    let tokens = [123, 456, 789]; // Placeholder tokens
    for token in tokens {
        constraint.accept_token(token).unwrap();
    }

    println!("\nAfter accepting 3 tokens:");
    println!("  Valid: {}", constraint.is_valid());

    // Reset for new generation
    constraint.reset();
    println!("\nAfter reset:");
    println!("  Valid: {}", constraint.is_valid());
    println!();
}

/// Demonstrates forking constraints for tree search.
fn demo_fork_for_tree_search() {
    println!("=== Forking for Tree Search ===\n");

    let grammar = Grammar::json_schema(schemas::SHOPPING_LIST);
    let mut base_constraint = GrammarConstraint::new(grammar, 50000).unwrap();

    // Simulate some base generation
    base_constraint.accept_token(100).unwrap();
    base_constraint.accept_token(200).unwrap();

    println!("Base constraint after 2 tokens: valid={}", base_constraint.is_valid());

    // Fork for tree search branches
    let branch1 = base_constraint.fork();
    let mut branch2 = base_constraint.fork();
    let mut branch3 = base_constraint.fork();

    println!("Forked 3 branches for parallel exploration");

    // Each branch can continue independently
    branch2.accept_token(301).unwrap();
    branch3.accept_token(302).unwrap();
    branch3.accept_token(303).unwrap();

    println!("  Branch 1: base state (unchanged)");
    println!("  Branch 2: +1 token");
    println!("  Branch 3: +2 tokens");
    println!();
    let _ = branch1; // Suppress unused warning
}

/// Demonstrates token mask computation.
fn demo_token_mask() {
    println!("=== Token Mask Computation ===\n");

    let grammar = Grammar::json_schema(schemas::API_RESPONSE);
    let constraint = GrammarConstraint::new(grammar, 1000).unwrap();

    // Compute the mask for current state
    let mask = constraint.compute_mask().unwrap();

    println!("Computed token mask:");
    println!("  Total vocabulary: {}", constraint.vocab_size());
    println!("  Allowed tokens: {}", mask.num_allowed());
    println!("  Blocked tokens: {}", mask.vocab_size() - mask.num_allowed());

    // Note: Without full llguidance integration, the mask allows all tokens
    // In production, LlgConstraint provides actual grammar-based masking
    println!("\nNote: Without tokenizer integration, mask allows all tokens.");
    println!("Use LlgConstraint with a tokenizer for actual grammar enforcement.");
    println!();
}

/// Demonstrates different grammar types.
fn demo_grammar_types() {
    println!("=== Grammar Types ===\n");

    // JSON Schema
    let json_grammar = Grammar::json_schema(r#"{"type": "string"}"#);
    println!("1. JSON Schema: {:?}", json_grammar);

    // Regex
    let regex_grammar = Grammar::regex(r"[0-9]{3}-[0-9]{4}");
    println!("2. Regex (phone format): {:?}", regex_grammar);

    // Lark CFG
    let lark_grammar = Grammar::lark(r#"
        start: sentence
        sentence: subject verb object
        subject: "The" NOUN
        verb: "ate" | "saw" | "liked"
        object: "the" NOUN
        NOUN: "cat" | "dog" | "mouse"
    "#);
    println!("3. Lark CFG (simple sentences): Lark(...)");

    // None (no constraint)
    let none = Grammar::None;
    println!("4. None: {:?}", none);
    println!();

    let _ = (json_grammar, regex_grammar, lark_grammar);
}

/// Demonstrates a conceptual generation loop.
fn demo_generation_concept() {
    println!("=== Conceptual Generation Loop ===\n");

    println!("In practice, grammar-constrained generation follows this pattern:");
    println!();
    println!("```rust");
    println!("// Setup");
    println!("let grammar = Grammar::json_schema(schema);");
    println!("let llg_grammar = to_llguidance(&grammar)?;");
    println!("let parser = factory.create_parser(llg_grammar)?;");
    println!("let mut constraint = LlgConstraint::new(parser);");
    println!();
    println!("// Generation loop");
    println!("loop {{");
    println!("    // 1. Compute mask for current state");
    println!("    let mask = constraint.compute_mask()?;");
    println!();
    println!("    // 2. Run model forward pass");
    println!("    let logits = model.forward(tokens)?;");
    println!();
    println!("    // 3. Apply mask to logits");
    println!("    let masked_logits = mask.apply(logits);");
    println!();
    println!("    // 4. Sample from masked distribution");
    println!("    let token = sample(masked_logits);");
    println!();
    println!("    // 5. Update constraint state");
    println!("    constraint.accept_token(token)?;");
    println!();
    println!("    if constraint.is_complete() {{");
    println!("        break;");
    println!("    }}");
    println!("}}");
    println!("```");
    println!();
}

fn main() {
    println!("╔════════════════════════════════════════════╗");
    println!("║  Dendrite: Grammar-Constrained JSON Gen    ║");
    println!("╚════════════════════════════════════════════╝\n");

    demo_schema_creation();
    demo_state_management();
    demo_fork_for_tree_search();
    demo_token_mask();
    demo_grammar_types();
    demo_generation_concept();

    println!("═══════════════════════════════════════════════");
    println!("Key Concepts:");
    println!("  • Grammar: Defines valid output structure (JSON schema, regex, CFG)");
    println!("  • Constraint: Tracks state and computes valid token masks");
    println!("  • Fork: Creates independent copies for tree search");
    println!("  • Mask: Filters logits to only allow valid continuations");
    println!();
    println!("For full integration with llguidance:");
    println!("  1. Load tokenizer to create TokEnv");
    println!("  2. Create ParserFactory from TokEnv");
    println!("  3. Use LlgConstraint for actual grammar enforcement");
    println!("═══════════════════════════════════════════════");
}
