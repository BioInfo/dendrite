# Contributing to Dendrite

Thank you for your interest in contributing to Dendrite! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites

- Rust 1.75 or later
- CUDA 12.x (optional, for GPU features)

### Getting Started

```bash
# Clone the repository
git clone https://github.com/BioInfo/dendrite.git
cd dendrite

# Build the project
cargo build

# Run tests
cargo test

# Run with all checks (recommended before committing)
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
```

## Development Workflow

### Branching Strategy

We use the following branch naming conventions:

- `main` - Protected branch, always deployable
- `feature/*` - New features (e.g., `feature/add-beam-search`)
- `fix/*` - Bug fixes (e.g., `fix/memory-leak-in-cache`)
- `refactor/*` - Code refactoring
- `docs/*` - Documentation updates

### Making Changes

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards

3. **Write tests** for new functionality (TDD encouraged)

4. **Run the full check suite**:
   ```bash
   cargo fmt --all
   cargo clippy --all-targets --all-features -- -D warnings
   cargo test --all-features
   ```

5. **Commit your changes** with a descriptive message:
   ```bash
   git commit -m "feat: add beam search algorithm"
   ```

6. **Push and create a Pull Request**

### Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation only
- `refactor:` - Code refactoring (no functional change)
- `test:` - Adding or updating tests
- `perf:` - Performance improvement
- `chore:` - Maintenance tasks

Examples:
```
feat: implement O(1) fork for tree state
fix: resolve memory leak in KV cache pool
docs: add architecture diagram to README
test: add property tests for block allocator
perf: optimize attention kernel dispatch
```

## Coding Standards

### Formatting

- Use `cargo fmt` with our `rustfmt.toml` configuration
- Maximum line width: 100 characters
- Use 4 spaces for indentation

### Linting

- All code must pass `cargo clippy` with `-D warnings`
- Address clippy suggestions or document exceptions

### Documentation

- All public APIs must have doc comments
- Include examples in doc comments where helpful
- Document safety requirements for `unsafe` code

### Testing

We practice Test-Driven Development (TDD):

1. **Write a failing test** that defines the expected behavior
2. **Implement the minimum code** to make the test pass
3. **Refactor** while keeping tests green

#### Test Organization

```
crates/
  dendrite-core/
    src/
      cache/
        mod.rs
        pool.rs
    tests/           # Integration tests
      cache_tests.rs
```

- Unit tests: In `#[cfg(test)]` modules within source files
- Integration tests: In `tests/` directory
- Property tests: Use `proptest` for invariant testing
- Benchmarks: Use `criterion` in `benches/`

#### Test Naming

```rust
#[test]
fn fork_handle_shares_memory_with_parent() { ... }

#[test]
fn block_pool_returns_error_when_exhausted() { ... }
```

### Performance

- Profile before optimizing
- Document performance-critical code paths
- Add benchmarks for hot paths

## Pull Request Process

1. **Ensure CI passes** - All checks must be green
2. **Update documentation** if needed
3. **Add tests** for new functionality
4. **Keep PRs focused** - One feature/fix per PR
5. **Respond to feedback** promptly

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Tests added/updated for changes
- [ ] Documentation updated if needed
- [ ] Commit messages follow conventions
- [ ] CI passes

## Getting Help

- Open an issue for bugs or feature requests
- Use discussions for questions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
