Quick Start
===========

Ops
---

All ops are importable directly from ``tritonix``:

.. code-block:: python

   import torch
   from tritonix import matmul, conv2d_forward

   a = torch.randn(1024, 512, device="cuda", dtype=torch.float16)
   b = torch.randn(512, 768, device="cuda", dtype=torch.float16)
   c = matmul(a, b)

   x = torch.randn(8, 64, 32, 32, device="cuda", dtype=torch.float16)
   w = torch.randn(128, 64, 3, 3, device="cuda", dtype=torch.float16)
   y = conv2d_forward(x, w, padding=(1, 1))

Dispatch
--------

``matmul`` is a :class:`~tritonix.dispatcher.DynamicDispatcher`. On the first call for a
given shape it benchmarks Triton and PyTorch, then caches the winner:

.. code-block:: python

   c = matmul(a, b)          # first call: benchmarks both backends
   c = matmul(a, b)          # subsequent calls: zero-overhead cache hit

   print(matmul.cache)       # {shape_key: "triton" | "pytorch"}
   print(matmul.timings)     # {shape_key: {"triton": ms, "pytorch": ms}}

   with matmul.force_backend("triton"):
       ref = matmul(a, b)    # bypass cache

Autotuning
----------

Every ``@tunable`` kernel exposes :meth:`~tritonix.autotune.TunableKernel.tune`:

.. code-block:: python

   from tritonix.ops.matmul import matmul_kernel

   def launcher(cfg):
       c = torch.empty((M, N), device="cuda", dtype=torch.float16)
       grid = (triton.cdiv(M, cfg["block_m"]), triton.cdiv(N, cfg["block_n"]))
       matmul_kernel[grid](a, b, c, M, N, K,
                           a.stride(0), a.stride(1),
                           b.stride(0), b.stride(1),
                           c.stride(0), c.stride(1),
                           **cfg)
       return c

   best = matmul_kernel.tune(launcher, method="grid", warmup=10, rep=20, verbose=True)
   # best = {"block_m": 128, "block_n": 128, "block_k": 32, "group_m": 8, ...}

See :doc:`autotuning` for a full explanation of the pruning strategies.
