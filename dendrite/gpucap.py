"""GPU memory cap from DENDRITE_GPU_CAP_GIB. On GB10 a cgroup MemoryMax does not limit CUDA allocations, so an overrun
reaches NVRM and can take the box down; the torch allocator cap turns it into a catchable OutOfMemoryError instead."""
import os

import torch

cap = os.environ.get("DENDRITE_GPU_CAP_GIB")
if cap:
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, float(cap) * 2**30 / total))
