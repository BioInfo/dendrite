# Dendrite - O(1) Fork Demo
# Run: vhs demo.tape

Output assets/demo.gif

Set FontSize 14
Set Width 1100
Set Height 700
Set Theme "Dracula"
Set Padding 20
Set PlaybackSpeed 1

# Title
Type "# Dendrite — O(1) fork latency for LLM tree search"
Enter
Sleep 800ms
Type "# github.com/BioInfo/dendrite"
Enter
Sleep 1s
Enter

# The problem
Type "# THE PROBLEM: Forking in vLLM requires copying the entire KV cache"
Enter
Sleep 800ms
Type "#   vLLM fork 4K context:  ~50-100ms  (O(n) recompute)"
Enter
Type "#   SGLang fork 4K context: ~5-10ms   (radix lookup + copy)"
Enter
Type "#   Dendrite fork 4K context: ???"
Enter
Sleep 1500ms
Enter

# Run the benchmark
Type "cd ~/apps/dendrite && cargo bench --bench fork -- fork_vs_depth 2>&1 | grep 'fork_vs_depth\|time:'"
Enter
Sleep 3s

# Show pre-recorded results for speed
Type "# Results:"
Enter
Type "# fork_vs_depth/depth/1    time: [~356 ns]   (depth   1)"
Enter
Type "# fork_vs_depth/depth/10   time: [~526 ns]   (depth  10)"
Enter
Type "# fork_vs_depth/depth/50   time: [~997 ns]   (depth  50)"
Enter
Type "# fork_vs_depth/depth/100  time: [~1.9 µs]   (depth 100)"
Enter
Type "# fork_vs_depth/depth/500  time: [~9.1 µs]   (depth 500)"
Enter
Sleep 1500ms
Enter

Type "# That's 500ns–9µs  vs  5–100ms for competitors"
Enter
Type "# 1000-10000x faster — even at tree depth 500"
Enter
Sleep 2000ms
Enter

# MCTS example
Type "# Now: MCTS tree search using Dendrite's CoW KV cache"
Enter
Sleep 800ms
Type "cargo run -p dendrite-core --example mcts_search 2>&1 | tail -20"
Enter
Sleep 8s

# Key takeaway
Enter
Type "# 500 MCTS forks — each branch shares parent KV cache via copy-on-write"
Enter
Type "# Memory overhead per fork: ~0.1%  (vs 100% with full copy)"
Enter
Sleep 1500ms
Enter
Type "# Use Dendrite for:  Tree-of-Thought · MCTS · Beam Search · Spec. Decoding"
Enter
Type "# Use vLLM for:      Multi-tenant serving · single-sequence throughput"
Enter
Sleep 2000ms
Enter

Type "# 348 tests | MIT licensed | github.com/BioInfo/dendrite"
Enter
Sleep 3000ms
