#!/usr/bin/env python3
"""Generate golden token sequences from HuggingFace Transformers.

This script runs TinyLlama through HuggingFace's implementation and captures
the greedy-decoded output tokens. These become the ground truth for validating
Dendrite's implementation.

Usage:
    python generate_golden.py --model /path/to/tinyllama --output golden_cases.json

Output format:
    [
        {
            "name": "hello_world",
            "prompt": "Hello, world!",
            "input_tokens": [1, 15043, 29892, ...],
            "expected_tokens": [29871, 306, ...],
            "max_tokens": 20,
            "temperature": 0.0
        },
        ...
    ]
"""

import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def generate_golden_case(
    model,
    tokenizer,
    name: str,
    prompt: str,
    max_tokens: int = 20,
) -> dict:
    """Generate a single golden test case."""
    # Encode prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Greedy generation
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=False,  # Greedy
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    # Extract tokens
    input_tokens = input_ids[0].tolist()
    all_tokens = outputs[0].tolist()
    generated_tokens = all_tokens[len(input_tokens):]

    return {
        "name": name,
        "prompt": prompt,
        "input_tokens": input_tokens,
        "expected_tokens": generated_tokens,
        "full_sequence": all_tokens,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }


def format_rust_vec(tokens: list) -> str:
    """Format a list as a Rust vec![] literal."""
    return "vec![" + ", ".join(str(t) for t in tokens) + "]"


def main():
    parser = argparse.ArgumentParser(description="Generate golden token sequences")
    parser.add_argument("--model", type=str, required=True, help="Path to model directory")
    parser.add_argument("--output", type=str, default="golden_cases.json", help="Output JSON file")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, cuda)")
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,  # Use FP32 for determinism
        device_map=args.device,
    )
    model.set_default_dtype = torch.float32
    model.config.use_cache = True

    print(f"Model loaded: {model.config.model_type}")
    print(f"Vocab size: {model.config.vocab_size}")
    print(f"Hidden size: {model.config.hidden_size}")
    print(f"Layers: {model.config.num_hidden_layers}")

    # Define test cases
    test_prompts = [
        ("hello_greeting", "Hello, my name is"),
        ("count_numbers", "1, 2, 3, 4,"),
        ("capital_question", "The capital of France is"),
        ("code_completion", "def fibonacci(n):"),
        ("story_start", "Once upon a time,"),
        ("math_simple", "What is 2 + 2?"),
        ("translate_simple", "Translate to Spanish: Hello"),
        ("explain_ai", "Explain what AI is:"),
        ("list_colors", "List three colors:"),
        ("complete_sentence", "The quick brown fox"),
    ]

    cases = []
    for name, prompt in test_prompts:
        print(f"\nGenerating: {name}")
        print(f"  Prompt: {prompt}")

        case = generate_golden_case(model, tokenizer, name, prompt, max_tokens=20)

        print(f"  Input tokens: {case['input_tokens']}")
        print(f"  Generated: {case['expected_tokens'][:10]}...")

        # Decode for inspection
        generated_text = tokenizer.decode(case["expected_tokens"], skip_special_tokens=True)
        print(f"  Text: {generated_text[:50]}...")

        cases.append(case)

    # Save to JSON
    output_path = args.output
    with open(output_path, "w") as f:
        json.dump(cases, f, indent=2)

    print(f"\nSaved {len(cases)} golden cases to {output_path}")

    # Also generate a Rust constant for embedding in tests
    rust_path = output_path.replace(".json", ".rs.txt")
    with open(rust_path, "w") as f:
        f.write("// Auto-generated golden cases from HuggingFace\n")
        f.write("// Model: TinyLlama-1.1B\n")
        f.write("// Generated with: python generate_golden.py\n\n")
        f.write("fn tinyllama_golden_cases() -> Vec<GoldenCase> {\n")
        f.write("    vec![\n")
        for case in cases:
            input_vec = format_rust_vec(case["input_tokens"])
            expected_vec = format_rust_vec(case["expected_tokens"])
            f.write(f'        GoldenCase::greedy("{case["name"]}", {input_vec}, {expected_vec})')
            f.write(f'.with_max_tokens({case["max_tokens"]}),\n')
        f.write("    ]\n")
        f.write("}\n")

    print(f"Saved Rust snippet to {rust_path}")


if __name__ == "__main__":
    main()
