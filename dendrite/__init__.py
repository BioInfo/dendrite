"""Dendrite: fork-native serving for kev decision models on hybrid Qwen3.5 backbones.

  dendrite.fork      question branches that share one copy of the state cache
  dendrite.extend    exact state reuse: continue a cached state with only its new tokens
  dendrite.serve     kev.serve with checkpoints, so a state that grows between requests is not re-run
  dendrite.memguard  host-memory watchdog for unified-memory GPUs (GB10)
  dendrite.gpucap    torch allocator cap from DENDRITE_GPU_CAP_GIB
"""
