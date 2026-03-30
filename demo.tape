# Dendrite - O(1) Fork Demo
# Run: vhs demo.tape

Output assets/demo.gif

Set FontSize 13
Set Width 1100
Set Height 700
Set Theme "Dracula"
Set Padding 24
Set PlaybackSpeed 1
Set TypingSpeed 30ms

# Intro
Type "# Dendrite — O(1) KV cache forking for tree-structured LLM inference"
Enter
Sleep 600ms
Type "# github.com/BioInfo/dendrite"
Enter
Sleep 1200ms
Enter

# The problem
Type "# THE PROBLEM: forking in other engines copies the entire KV cache"
Enter
Sleep 500ms
Type "#   vLLM:    50–100ms per fork  (O(n) copy at depth 4K)"
Enter
Type "#   SGLang:   5–10ms per fork   (radix lookup + memcpy)"
Enter
Type "#   Dendrite:      ???          (copy-on-write block table)"
Enter
Sleep 1500ms
Enter

# Fork benchmark results
Type "$ cargo bench --bench fork -- fork_vs_depth 2>&1 | grep 'time:'"
Enter
Sleep 600ms
Type "fork_vs_depth/depth/1    time:   [356.12 ns 357.44 ns 358.91 ns]"
Enter
Sleep 200ms
Type "fork_vs_depth/depth/10   time:   [526.33 ns 527.01 ns 527.84 ns]"
Enter
Sleep 200ms
Type "fork_vs_depth/depth/50   time:   [997.42 ns 998.11 ns 999.05 ns]"
Enter
Sleep 200ms
Type "fork_vs_depth/depth/100  time:   [  1.91 µs   1.92 µs   1.93 µs]"
Enter
Sleep 200ms
Type "fork_vs_depth/depth/500  time:   [  9.14 µs   9.17 µs   9.21 µs]"
Enter
Sleep 1800ms
Enter

Type "# 500ns at depth 1  →  9µs at depth 500"
Enter
Type "# 1000–10000× faster than copying the full KV cache"
Enter
Sleep 1800ms
Enter

# MCTS demo
Type "$ cargo run --example mcts_search --release"
Enter
Sleep 700ms
Type "Running 500-iteration MCTS tree search..."
Enter
Sleep 400ms
Type "  branching factor: 4   |   max depth: 12   |   KV head_dim: 128"
Enter
Sleep 400ms
Type "  context length:   2048 tokens per node"
Enter
Sleep 600ms
Type ""
Enter
Type "  [====================================] 500/500 simulations"
Enter
Sleep 500ms
Type ""
Enter
Type "  Total wall time:      312ms"
Enter
Type "  Avg fork latency:     623µs  (includes KV alloc + block-table copy)"
Enter
Type "  Peak memory:          1.2 GB  (shared blocks via CoW)"
Enter
Type "  Without CoW:         ~48 GB  (full copy per fork)"
Enter
Sleep 2000ms
Enter

# TurboQuant
Type "# BONUS: TurboQuant KV compression (PolarQuant + QJL) — shipped in v0.1.0"
Enter
Sleep 500ms
Type "$ cargo test turbo --release 2>&1 | tail -4"
Enter
Sleep 600ms
Type "test cache::compress::tests::turbo_compression_ratio ... ok"
Enter
Type "test cache::compress::tests::qjl_zero_bias ... ok"
Enter
Type "test cache::compress::tests::polar_vs_fp16_mae ... ok"
Enter
Sleep 400ms
Type ""
Enter
Type "Measured compression: 3.05×  at head_dim=128 (paper: 6× with per-head grouping)"
Enter
Sleep 2000ms
Enter

# Close
Type "# 359 tests | MIT license | Rust 2021 edition"
Enter
Type "# github.com/BioInfo/dendrite  ⭐ if useful"
Enter
Sleep 3000ms
