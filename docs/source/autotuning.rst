Autotuning
==========

Triton kernels have many tunable ``constexpr`` parameters — block sizes, pipeline
stages, warp counts — whose optimal values depend on problem shape and hardware.
Tritonix searches this space using two complementary pruning strategies.

Declaring a tunable kernel
--------------------------

Wrap a ``@triton.jit`` kernel with ``@tunable``:

.. code-block:: python

   from tritonix.autotune import tunable, PowerOfTwo, Choice, Range
   import triton, triton.language as tl

   @tunable(
       keys=["m", "n", "k"],
       space={
           "block_m": PowerOfTwo(32, 256),   # {32, 64, 128, 256}
           "block_n": PowerOfTwo(32, 256),
           "block_k": PowerOfTwo(16, 128),   # {16, 32, 64, 128}
           "group_m": Choice([4, 8]),
           "num_stages": Range(2, 5),        # {2, 3, 4}
           "num_warps": Choice([4, 8]),
       },
       memory_params={"block_m", "block_n", "block_k", "num_stages"},
   )
   @triton.jit
   def my_kernel(..., block_m: tl.constexpr, block_n: tl.constexpr, ...):
       ...

``keys`` are the problem-shape arguments used to cache the best config per shape.
``memory_params`` are the parameters that affect shared memory — these drive OOM
boundary detection (see below).

Config space types
------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Type
     - Constructor
     - Values generated
   * - ``PowerOfTwo``
     - ``PowerOfTwo(lo, hi)``
     - Powers of 2 in ``[lo, hi]``: ``{32, 64, 128, 256}``
   * - ``Range``
     - ``Range(lo, hi)``
     - Integers in ``[lo, hi)``: ``{2, 3, 4}``
   * - ``Choice``
     - ``Choice([a, b, c])``
     - Exactly the provided values

Grid search
-----------

Call ``kernel.tune(launcher, method="grid")`` to run the grid search:

.. code-block:: python

   best_cfg = my_kernel.tune(launcher, method="grid", warmup=10, rep=20, verbose=True)

The search proceeds in two nested loops:

1. **Outer loop — memory params (trie-guided, midpoint order).**
   :class:`~tritonix.utils.pruners.MonotonicCascadeTrie` yields memory configs in
   midpoint order. The first probe is the midpoint of the unpruned space; on OOM,
   the upper half is pruned and the trie skips all dominated configs automatically.

2. **Inner loop — non-memory params (perf-pruned).**
   For each valid memory config, non-memory params are swept.
   :class:`~tritonix.utils.pruners.CoordinateMonotonicFunction` prunes configs
   whose latency is guaranteed worse than the current best under the unimodality
   assumption.

Monotonic trie (OOM pruning)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Shared memory grows monotonically with ``memory_params``. If
``(block_m=128, block_n=128, block_k=64)`` causes an OOM, then any config with
values ≥ those in **all** memory dimensions will also OOM. The trie stores the
minimal set of failure points and prunes subtrees via prefix checks.

Because the search runs in **midpoint order**, the first OOM prunes roughly half
the remaining memory space. Subsequent OOMs narrow the feasible region in
O(log n) probes.

Unimodality pruner (performance pruning)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each parameter dimension, with all others fixed, latency is assumed unimodal
— a single minimum with non-decreasing tails on both sides. After each benchmark:

* If a value to the **right** of the current best is worse, the upper bound for
  that slice drops.
* If a value to the **left** of the current best is worse, the lower bound rises.

Any config outside those bounds is pruned without benchmarking.

Bayesian optimization
---------------------

Requires ``ax-platform``:

.. code-block:: python

   best_cfg = my_kernel.tune(
       launcher, method="bayesian", max_evals=60, warmup=10, rep=20
   )

Ax's Bayesian optimizer proposes configs. OOM configs are penalized with a large
latency value and the trie prunes dominated memory configs for subsequent trials.
