import itertools
from dataclasses import dataclass
from typing import Any

import triton
from triton import OutOfResources

from tritonix.utils.pruners import (
    MonotonicCascadeTrie,
    CoordinateMonotonicFunction,
)
from tritonix.utils.spaces import (
    SpaceConfig,
    ConfigParam,
    Choice,
    PowerOfTwo,
    Range,
)


@dataclass
class TunableKernel:
    kernel: Any
    keys: list[str]
    space: dict[str, ConfigParam]
    memory_params: set[str]

    def __post_init__(self):
        if isinstance(self.kernel, TunableKernel):
            self.kernel = self.kernel.kernel

    def __getattr__(self, name):
        return getattr(self.kernel, name)

    def __getitem__(self, grid):
        return self.kernel[grid]

    def _build_trie(self):
        names = [p for p in self.memory_params if p in self.space]
        values = [sorted(self.space[p].values()) for p in names]
        trie = MonotonicCascadeTrie(
            SpaceConfig({n: Choice(v) for n, v in zip(names, values)})
        )
        return trie, names, values

    def configs(self, filter_fn=None):
        configs = [{}]
        for key, vals in zip(
            self.space, [p.values() for p in self.space.values()]
        ):
            configs = [{**c, key: v} for c in configs for v in vals]
        if filter_fn is not None:
            configs = [c for c in configs if filter_fn(c)]
        return configs

    def tune(
        self,
        launcher,
        method="grid",
        max_configs=None,
        max_evals=100,
        filter_fn=None,
        warmup=25,
        rep=100,
        verbose=False,
    ):
        if method == "grid":
            return self._tune_grid(
                launcher, max_configs, filter_fn, warmup, rep, verbose
            )
        return self._tune_bayesian(
            launcher, max_evals, filter_fn, warmup, rep, verbose
        )

    def _tune_grid(
        self, launcher, max_configs, filter_fn, warmup, rep, verbose
    ):
        trie, trie_names, trie_values = self._build_trie()

        all_names = list(self.space.keys())
        all_values = [sorted(self.space[k].values()) for k in all_names]
        all_inverse = [
            {v: i for i, v in enumerate(vals)} for vals in all_values
        ]
        perf = CoordinateMonotonicFunction(
            SpaceConfig({n: Choice(v) for n, v in zip(all_names, all_values)})
        )

        trie_name_set = set(trie_names)
        non_mem_names = [n for n in all_names if n not in trie_name_set]
        non_mem_values = [all_values[all_names.index(n)] for n in non_mem_names]

        def full_idx(cfg):
            return tuple(
                all_inverse[d][cfg[all_names[d]]] for d in range(len(all_names))
            )

        best_time, best_cfg = float("inf"), None
        n_evaluated = 0

        for mem_idx in trie.generate_all_unpruned_midpoint():
            if max_configs is not None and n_evaluated >= max_configs:
                break
            mem_cfg = {
                trie_names[i]: trie_values[i][mem_idx[i]]
                for i in range(len(trie_names))
            }

            for combo in itertools.product(*non_mem_values):
                if max_configs is not None and n_evaluated >= max_configs:
                    break
                cfg = {**mem_cfg, **dict(zip(non_mem_names, combo))}
                if filter_fn is not None and not filter_fn(cfg):
                    continue
                fidx = full_idx(cfg)
                if perf.is_pruned(fidx):
                    continue
                try:
                    ms = triton.testing.do_bench(
                        lambda: launcher(cfg), warmup=warmup, rep=rep
                    )
                    perf.record(fidx, ms)
                    n_evaluated += 1
                    is_best = ms < best_time
                    if is_best:
                        best_time, best_cfg = ms, cfg
                    if verbose:
                        marker = " *" if is_best else ""
                        print(f"  {ms:8.3f}ms  {cfg}{marker}", flush=True)
                except ValueError as e:
                    if verbose:
                        print(f"  SKIP      {cfg}  ({e})", flush=True)
                    continue
                except OutOfResources:
                    trie.prune(mem_idx)
                    if verbose:
                        print(f"  OOM       {cfg}", flush=True)
                    break

        return best_cfg

    def _tune_bayesian(
        self,
        launcher,
        max_evals,
        filter_fn,
        warmup,
        rep,
        verbose,
        oom_penalty=1e6,
    ):
        import logging
        from ax.service.ax_client import AxClient
        from ax.service.utils.instantiation import ObjectiveProperties
        from ax.utils.common.logger import set_ax_logger_levels

        set_ax_logger_levels(logging.WARNING)

        trie, trie_names, trie_values = self._build_trie()

        ax_params = [
            {
                "name": name,
                "type": "choice",
                "values": param.values(),
                "sort_values": True,
                "is_ordered": isinstance(param, (PowerOfTwo, Range)),
            }
            for name, param in self.space.items()
            if len(param.values()) > 1
        ]
        ax_client = AxClient(verbose_logging=False)
        ax_client.create_experiment(
            name="kernel_tune",
            parameters=ax_params,
            objectives={"latency_ms": ObjectiveProperties(minimize=True)},
        )

        def trie_idx(cfg):
            return tuple(
                trie_values[i].index(cfg[n]) for i, n in enumerate(trie_names)
            )

        for i in range(max_evals):
            params, trial_index = ax_client.get_next_trial()
            penalty = trie_names and trie.is_pruned(trie_idx(params))
            penalty = penalty or (
                filter_fn is not None and not filter_fn(params)
            )
            if penalty:
                ax_client.complete_trial(
                    trial_index=trial_index,
                    raw_data={"latency_ms": oom_penalty},
                )
                continue
            try:
                ms = triton.testing.do_bench(
                    lambda: launcher(params), warmup=warmup, rep=rep
                )
                ax_client.complete_trial(
                    trial_index=trial_index, raw_data={"latency_ms": ms}
                )
                if verbose:
                    print(f"  [{i+1}/{max_evals}] {ms:.3f}ms — {params}")
            except OutOfResources:
                if trie_names:
                    trie.prune(trie_idx(params))
                ax_client.complete_trial(
                    trial_index=trial_index,
                    raw_data={"latency_ms": oom_penalty},
                )

        best_params, _ = ax_client.get_best_parameters()
        return best_params


def tunable(keys, space, memory_params=None):
    memory_params = (memory_params or set()) | {"num_stages"}

    def wrapper(kernel):
        t = TunableKernel(
            kernel=kernel, keys=keys, space=space, memory_params=memory_params
        )
        t.kernel.specialize_keys = keys
        return t

    return wrapper
