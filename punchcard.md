### Checkpoints

- I'm working on fused swiglu where a single kernel does all the operations in [](/kernels/fused/glu.py)
- I was trying to figure out the offsets for z_ptr, offs_l and the strides. perhaps a toy example with triton-interpret might make things clearer
- The idea for the final step is that we need to use atomic_add, but we might need to work on the l-order for more efficient cache hit rate.