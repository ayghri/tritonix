Backend Dispatch
================

Tritonix ops are :class:`~tritonix.dispatcher.DynamicDispatcher` instances that
automatically benchmark registered backends per input shape and cache the winner.

How it works
------------

On the first call for a given shape:

1. Each registered backend is timed with ``triton.testing.do_bench``.
2. The fastest backend (lowest median latency) is stored in an in-memory cache.
3. The result is optionally persisted to ``~/.triton/cache`` via Triton's
   ``get_cache_manager`` so subsequent processes skip the benchmark.

Subsequent calls for the same shape hit the in-memory cache with zero overhead.

Usage
-----

.. code-block:: python

   from tritonix import matmul

   c = matmul(a, b)              # first call benchmarks; cached after

   # inspect results
   print(matmul.cache)           # {shape_key: "triton" | "pytorch"}
   print(matmul.timings)         # {shape_key: {"triton": ms, "pytorch": ms}}

   # force a backend (bypass cache)
   with matmul.force_backend("pytorch"):
       ref = matmul(a, b)

   # clear in-memory cache (disk entries remain valid)
   matmul.clear_cache()

Creating a custom dispatcher
-----------------------------

Use :func:`~tritonix.dispatcher.dynamic_dispatch` to build a dispatcher for any op:

.. code-block:: python

   from tritonix.dispatcher import dynamic_dispatch

   def triton_add(x, y): ...
   def torch_add(x, y): return x + y

   vector_add = dynamic_dispatch(
       {"triton": triton_add, "pytorch": torch_add},
       key=[],           # key on all tensor shapes/dtypes automatically
       warmup=25,
       rep=100,
   )

   out = vector_add(x, y)

The ``key`` parameter lists scalar arguments to include in the cache key. Tensor
arguments are always keyed on shape + dtype regardless of ``key``.
