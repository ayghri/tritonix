"""Autotuning utilities for Triton kernels.

Uses a MonotonicCascadeTrie to prune configs that would OOM on shared memory.
The key insight: if (block_m=128, block_n=128, block_k=64) causes OOM, then
any config with equal-or-larger values in ALL smem dimensions will also OOM.
The trie tracks these minimal failure points and skips entire subtrees.

For Bayesian optimization, uses Meta's Ax platform (adaptive experimentation).

Usage:

    @tunable(
        keys=["m", "n", "k"],
        space={
            "block_m": PowerOfTwo(16, 256),
            "block_n": PowerOfTwo(16, 256),
            "block_k": PowerOfTwo(16, 128),
            "group_m": Choice([1, 2, 4, 8, 16, 24]),
            "num_stages": Range(1, 8),
            "num_warps": PowerOfTwo(1, 16),
        },
        # These params affect shared memory — monotonic OOM pruning applies
        smem_params=["block_m", "block_n", "block_k", "num_stages"],
    )
    @triton.jit
    def matmul_kernel(...):
        ...

    # Grid search with automatic OOM pruning
    best = matmul_kernel.tune(launcher_fn, method="grid")

    # Bayesian optimization via Ax
    best = matmul_kernel.tune(launcher_fn, method="bayesian", max_evals=100)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import triton

from tritonix.utils.trie import MonotonicCascadeTrie


# ---------------------------------------------------------------------------
# Search space primitives
# ---------------------------------------------------------------------------

@dataclass
class PowerOfTwo:
    """Powers of 2 in [lo, hi] inclusive.  E.g. PowerOfTwo(16, 128) -> {16,32,64,128}."""
    lo: int
    hi: int

    def values(self) -> list[int]:
        vals = []
        v = self.lo
        while v <= self.hi:
            vals.append(v)
            v *= 2
        return vals


@dataclass
class Range:
    """Integers in [lo, hi] inclusive."""
    lo: int
    hi: int
    step: int = 1

    def values(self) -> list[int]:
        return list(range(self.lo, self.hi + 1, self.step))


@dataclass
class Choice:
    """Explicit list of allowed values."""
    options: list[Any]

    def values(self) -> list[Any]:
        return list(self.options)


SearchParam = PowerOfTwo | Range | Choice


# ---------------------------------------------------------------------------
# OOM detection
# ---------------------------------------------------------------------------

_OOM_KEYWORDS = ("out of memory", "shared memory", "CUDA out of memory", "smem")


def _is_oom_error(exc: Exception) -> bool:
    """Check if an exception is an OOM / shared memory error."""
    msg = str(exc).lower()
    return any(kw.lower() in msg for kw in _OOM_KEYWORDS)


# ---------------------------------------------------------------------------
# Performance pruning (coordinate-wise unimodality)
# ---------------------------------------------------------------------------

class PerformancePruner:
    """Prune search space using coordinate-wise unimodality of kernel latency.

    Assumption: for each tunable parameter d, with all other params fixed
    (a "slice"), latency L(d) is unimodal — single minimum, non-decreasing
    tails on both sides.

    This holds because each parameter controls one hardware trade-off:
      - block sizes: data reuse vs occupancy/register pressure
      - num_stages: latency hiding vs shared memory consumption
      - num_warps: thread-level parallelism vs per-thread registers

    The joint surface is NOT unimodal (parameters interact), but along any
    single axis with others fixed, the trade-off has a natural sweet spot.

    Pruning rule — given two results in the same slice at indices a, b:
      L(a) < L(b) and a < b → optimum is below b → prune indices ≥ b
      L(a) < L(b) and a > b → optimum is above b → prune indices ≤ b
    """

    def __init__(self, n_dims: int, dim_sizes: list[int]):
        self.n_dims = n_dims
        self.dim_sizes = dim_sizes
        self._obs: list[dict] = [{} for _ in range(n_dims)]
        self._ranges: list[dict] = [{} for _ in range(n_dims)]

    def record(self, config_idx: tuple[int, ...], latency: float):
        """Record a benchmark and tighten per-dimension bounds."""
        for d in range(self.n_dims):
            slice_key = config_idx[:d] + config_idx[d + 1:]
            obs = self._obs[d].setdefault(slice_key, {})
            obs[config_idx[d]] = latency

            if len(obs) < 2:
                continue

            # Recompute bounds from the best observation in this slice
            best_idx = min(obs, key=obs.__getitem__)
            best_lat = obs[best_idx]

            lo, hi = 0, self.dim_sizes[d] - 1

            # Upper bound: first index right of best with worse latency
            for i in sorted(i for i in obs if i > best_idx):
                if obs[i] > best_lat:
                    hi = i - 1
                    break

            # Lower bound: first index left of best with worse latency
            for i in sorted((i for i in obs if i < best_idx), reverse=True):
                if obs[i] > best_lat:
                    lo = i + 1
                    break

            if lo > 0 or hi < self.dim_sizes[d] - 1:
                self._ranges[d][slice_key] = (lo, hi)

    def is_pruned(self, config_idx: tuple[int, ...]) -> bool:
        """Check if config is outside the valid range in any dimension."""
        for d in range(self.n_dims):
            slice_key = config_idx[:d] + config_idx[d + 1:]
            if slice_key in self._ranges[d]:
                lo, hi = self._ranges[d][slice_key]
                if config_idx[d] < lo or config_idx[d] > hi:
                    return True
        return False


# ---------------------------------------------------------------------------
# TunableKernel wrapper
# ---------------------------------------------------------------------------

@dataclass
class TunableKernel:
    """Wraps a Triton JIT kernel with tuning metadata and OOM-aware pruning."""
    kernel: Any  # triton.JITFunction or TunableKernel (auto-unwrapped)
    keys: list[str]
    space: dict[str, SearchParam]
    smem_params: list[str]
    _cache: dict[tuple, dict] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        # Unwrap nested TunableKernel (e.g. when user wraps an already-decorated kernel)
        if isinstance(self.kernel, TunableKernel):
            self.kernel = self.kernel.kernel

    def __getattr__(self, name: str):
        return getattr(self.kernel, name)

    def __getitem__(self, grid):
        return self.kernel[grid]

    def _build_trie(self) -> tuple[MonotonicCascadeTrie, list[str], list[list[Any]]]:
        """Build a trie over the smem_params dimensions.

        Returns (trie, param_names, param_values) where param_values[i]
        is the sorted list of values for smem_params[i], and the trie
        shape is (len(param_values[0]), len(param_values[1]), ...).
        Trie indices correspond to positions in the sorted value lists.
        """
        names = [p for p in self.smem_params if p in self.space]
        values = [sorted(self.space[p].values()) for p in names]
        shape = tuple(len(v) for v in values)
        trie = MonotonicCascadeTrie(shape)
        return trie, names, values

    def _config_to_trie_index(
        self, cfg: dict, names: list[str], values: list[list[Any]]
    ) -> tuple[int, ...]:
        """Convert a config dict to trie index tuple."""
        idx = []
        for name, vals in zip(names, values):
            idx.append(vals.index(cfg[name]))
        return tuple(idx)

    def configs(self, filter_fn: Callable[[dict], bool] | None = None) -> list[dict]:
        """Generate all config combinations from the search space."""
        keys = list(self.space.keys())
        all_values = [self.space[k].values() for k in keys]

        configs = [{}]
        for key, vals in zip(keys, all_values):
            configs = [{**c, key: v} for c in configs for v in vals]

        if filter_fn is not None:
            configs = [c for c in configs if filter_fn(c)]
        return configs

    def triton_configs(
        self, filter_fn: Callable[[dict], bool] | None = None
    ) -> list[triton.Config]:
        """Generate triton.Config objects from the search space."""
        result = []
        for cfg in self.configs(filter_fn):
            num_stages = cfg.pop("num_stages", 4)
            num_warps = cfg.pop("num_warps", 4)
            result.append(
                triton.Config(cfg, num_stages=num_stages, num_warps=num_warps)
            )
        return result

    def tune(
        self,
        launcher: Callable[[dict], None],
        method: str = "grid",
        max_evals: int = 100,
        filter_fn: Callable[[dict], bool] | None = None,
        warmup: int = 25,
        rep: int = 100,
        verbose: bool = False,
    ) -> dict:
        """Find the best config for a given launcher function.

        Both methods use the MonotonicCascadeTrie to prune configs that
        would OOM based on shared-memory-related parameters.

        Args:
            launcher: Callable that takes a config dict and launches the kernel.
            method: "grid" for exhaustive search, "bayesian" for Ax optimization.
            max_evals: Max evaluations for Bayesian search.
            filter_fn: Optional predicate to prune the search space.
            warmup: Warmup iterations for benchmarking.
            rep: Repetition iterations for benchmarking.
            verbose: Print progress during tuning.

        Returns:
            Best config dict.
        """
        if method == "grid":
            return self._tune_grid(launcher, filter_fn, warmup, rep, verbose)
        elif method == "bayesian":
            return self._tune_bayesian(
                launcher, max_evals, filter_fn, warmup, rep, verbose
            )
        else:
            raise ValueError(
                f"Unknown method: {method!r}. Use 'grid' or 'bayesian'."
            )

    def _tune_grid(
        self,
        launcher: Callable[[dict], None],
        filter_fn: Callable[[dict], bool] | None,
        warmup: int,
        rep: int,
        verbose: bool,
    ) -> dict:
        trie, trie_names, trie_values = self._build_trie()

        # Unified index space for all params (for performance pruner)
        all_names = list(self.space.keys())
        all_values = [sorted(self.space[k].values()) for k in all_names]
        all_sizes = [len(v) for v in all_values]
        all_inverse = [{v: i for i, v in enumerate(vals)} for vals in all_values]
        perf = PerformancePruner(len(all_names), all_sizes)

        def _cfg_to_full_idx(cfg):
            return tuple(
                all_inverse[d][cfg[all_names[d]]] for d in range(len(all_names))
            )

        best_time = float("inf")
        best_cfg = None
        n_oom = 0
        n_tested = 0
        n_perf_pruned = 0
        tested = set()

        def _bench(cfg):
            """Benchmark a config. Returns (is_oom, latency_ms or None)."""
            nonlocal best_time, best_cfg, n_tested
            key = tuple(sorted(cfg.items()))
            if key in tested:
                return False, None
            tested.add(key)
            try:
                ms = triton.testing.do_bench(
                    lambda: launcher(cfg), warmup=warmup, rep=rep,
                )
                if isinstance(ms, tuple):
                    ms = ms[0]
                n_tested += 1
                if ms < best_time:
                    best_time = ms
                    best_cfg = cfg
                    if verbose:
                        print(f"  New best: {ms:.3f}ms — {cfg}")
                return False, ms
            except Exception as e:
                if _is_oom_error(e):
                    return True, None
                return False, None

        if not trie_names:
            # No smem params — plain grid search with performance pruning
            for cfg in self.configs(filter_fn):
                full_idx = _cfg_to_full_idx(cfg)
                if perf.is_pruned(full_idx):
                    n_perf_pruned += 1
                    continue
                is_oom, ms = _bench(cfg)
                if ms is not None:
                    perf.record(full_idx, ms)
        else:
            # Split space into smem vs non-smem params
            smem_set = set(trie_names)
            non_smem_keys = [k for k in self.space if k not in smem_set]
            non_smem_combos = [{}]
            for key in non_smem_keys:
                vals = self.space[key].values()
                non_smem_combos = [
                    {**c, key: v} for c in non_smem_combos for v in vals
                ]

            def _idx_to_smem_cfg(idx):
                return {
                    n: v[i]
                    for n, v, i in zip(trie_names, trie_values, idx)
                }

            # Phase 1: Binary-search the OOM boundary.
            probed = set()
            probe_non_smem = non_smem_combos[0]
            n_dims = len(trie_values)

            def _probe_smem(smem_idx):
                """Probe one smem config. Returns True if OOM (and pruned)."""
                nonlocal n_oom
                if smem_idx in probed or trie.is_pruned(smem_idx):
                    return trie.is_pruned(smem_idx)
                probed.add(smem_idx)
                smem_cfg = _idx_to_smem_cfg(smem_idx)
                cfg = {**smem_cfg, **probe_non_smem}
                if filter_fn and not filter_fn(cfg):
                    return False
                is_oom, ms = _bench(cfg)
                if is_oom:
                    trie.prune(smem_idx)
                    n_oom += 1
                    if verbose:
                        print(f"  OOM: {smem_cfg} — pruned subtree")
                    return True
                if ms is not None:
                    perf.record(_cfg_to_full_idx(cfg), ms)
                return False

            while True:
                smem_idx = trie.get_mid_point_unpruned()
                if smem_idx is None or smem_idx in probed:
                    break
                if _probe_smem(smem_idx):
                    continue

                # Mid-point safe — escalate each dim toward max to find OOM.
                found_oom = False
                for d in range(n_dims):
                    hi = len(trie_values[d]) - 1
                    lo = smem_idx[d] + 1
                    while lo <= hi:
                        mid = (lo + hi) // 2
                        escalated = list(smem_idx)
                        escalated[d] = mid
                        escalated = tuple(escalated)
                        if _probe_smem(escalated):
                            found_oom = True
                            hi = mid - 1
                        else:
                            lo = mid + 1
                if not found_oom:
                    break

            # Phase 2: Benchmark remaining configs with OOM + perf pruning.
            for smem_idx in trie.generate_all_unpruned():
                smem_cfg = _idx_to_smem_cfg(smem_idx)
                for non_smem in non_smem_combos:
                    cfg = {**smem_cfg, **non_smem}
                    if filter_fn and not filter_fn(cfg):
                        continue
                    full_idx = _cfg_to_full_idx(cfg)
                    if perf.is_pruned(full_idx):
                        n_perf_pruned += 1
                        continue
                    is_oom, ms = _bench(cfg)
                    if is_oom:
                        trie.prune(smem_idx)
                        n_oom += 1
                        if verbose:
                            print(f"  OOM: {smem_cfg} — pruned subtree")
                        break
                    if ms is not None:
                        perf.record(full_idx, ms)

        if verbose:
            total = 1
            for p in self.space.values():
                total *= len(p.values())
            n_oom_pruned = total - n_tested - n_oom - n_perf_pruned
            print(
                f"Grid search: {n_tested} benchmarked, {n_oom} OOM, "
                f"{n_perf_pruned} perf-pruned, {n_oom_pruned} OOM-pruned, "
                f"{total} total"
            )

        if best_cfg is None:
            raise RuntimeError("All configs failed during grid tuning")
        return best_cfg

    def _tune_bayesian(
        self,
        launcher: Callable[[dict], None],
        max_evals: int,
        filter_fn: Callable[[dict], bool] | None,
        warmup: int,
        rep: int,
        verbose: bool,
        oom_penalty_ms: float = 1e6,
    ) -> dict:
        try:
            from ax.service.ax_client import AxClient
            from ax.service.utils.instantiation import ObjectiveProperties
        except ImportError:
            raise ImportError(
                "Bayesian tuning requires ax-platform: pip install ax-platform"
            )

        trie, trie_names, trie_values = self._build_trie()

        # Build Ax parameter space
        ax_params = []
        for name, param in self.space.items():
            vals = param.values()
            if len(vals) <= 1:
                continue
            ax_params.append(
                {
                    "name": name,
                    "type": "choice",
                    "values": vals,
                    "sort_values": True,
                    "is_ordered": isinstance(param, (PowerOfTwo, Range)),
                }
            )

        ax_client = AxClient(verbose_logging=verbose)
        ax_client.create_experiment(
            name="kernel_tune",
            parameters=ax_params,
            objectives={"latency_ms": ObjectiveProperties(minimize=True)},
        )

        n_oom = 0
        n_pruned = 0

        for i in range(max_evals):
            params, trial_index = ax_client.get_next_trial()

            # Check trie before launching — if pruned, report penalty loss
            # so Ax's surrogate learns "this region = bad" without a kernel launch
            if trie_names:
                idx = self._config_to_trie_index(params, trie_names, trie_values)
                if trie.is_pruned(idx):
                    ax_client.complete_trial(
                        trial_index=trial_index,
                        raw_data={"latency_ms": oom_penalty_ms},
                    )
                    n_pruned += 1
                    if verbose:
                        smem_vals = {n: params[n] for n in trie_names}
                        print(f"  [{i+1}/{max_evals}] pruned (trie): {smem_vals}")
                    continue

            # Apply user filter — also report penalty so Ax avoids these
            if filter_fn is not None and not filter_fn(params):
                ax_client.complete_trial(
                    trial_index=trial_index,
                    raw_data={"latency_ms": oom_penalty_ms},
                )
                continue

            try:
                ms = triton.testing.do_bench(
                    lambda: launcher(params),
                    warmup=warmup,
                    rep=rep,
                )
                if isinstance(ms, tuple):
                    ms = ms[0]
                ax_client.complete_trial(
                    trial_index=trial_index, raw_data={"latency_ms": ms}
                )
                if verbose:
                    print(f"  [{i+1}/{max_evals}] {ms:.3f}ms — {params}")
            except Exception as e:
                # Actual OOM: register in trie AND report penalty to Ax
                if _is_oom_error(e) and trie_names:
                    idx = self._config_to_trie_index(params, trie_names, trie_values)
                    trie.prune(idx)
                    n_oom += 1
                    if verbose:
                        smem_vals = {n: params[n] for n in trie_names}
                        print(f"  [{i+1}/{max_evals}] OOM: {smem_vals} — pruned")
                ax_client.complete_trial(
                    trial_index=trial_index,
                    raw_data={"latency_ms": oom_penalty_ms},
                )

        if verbose:
            print(
                f"Bayesian search: {max_evals} evals, {n_oom} OOM, {n_pruned} pruned"
            )

        best_params, _metrics = ax_client.get_best_parameters()
        return best_params


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def tunable(
    keys: list[str],
    space: dict[str, SearchParam],
    smem_params: list[str] | None = None,
) -> Callable:
    """Decorator that adds tuning metadata to a Triton kernel.

    Args:
        keys: Runtime parameter names that affect the optimal config
              (e.g. ["m", "n", "k"] for matmul).
        space: Dict mapping constexpr parameter names to their search space.
        smem_params: Subset of space keys that affect shared memory usage.
                     Configs where ALL smem_params are >= a known OOM failure
                     will be automatically pruned. If None, defaults to
                     params whose names contain "block" or "num_stages".
    """
    if smem_params is None:
        smem_params = [
            k for k in space
            if "block" in k.lower() or k == "num_stages"
        ]

    def wrapper(kernel):
        t = TunableKernel(
            kernel=kernel, keys=keys, space=space, smem_params=smem_params
        )
        t.kernel.specialize_keys = keys
        return t

    return wrapper
