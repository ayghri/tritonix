import random
from typing import Dict, Tuple

from tritonix.utils.spaces import SpaceConfig


def _compare_failure_config(failure: Tuple[int], candidate: Tuple[int]) -> bool:
    return all(c >= f for c, f in zip(candidate, failure))


def _compare_failure_dim(failure: int, candidate: int, rank: int = 0) -> bool:
    return candidate >= failure


class MonotonicCascadeTrie:
    """
    Trie data structure to efficiently track and prune a monotonic search space.
    """

    def __init__(
        self,
        space: SpaceConfig,
        config_comparison_func=_compare_failure_config,
        dimension_comparison_func=_compare_failure_dim,
    ):

        self.shape = space.shape
        self.dimensions = len(self.shape)
        self._minimal_failures = []
        self._failure_trie = {}
        self._FAILURE_LEAF = {"_fail_": True}
        self._config_compare = config_comparison_func
        self._dim_compare = dimension_comparison_func

    def is_pruned(self, config_idx):
        """
        Checks if a full configuration is pruned by any known failure.
        A configuration is pruned if it is element-wise >= any minimal failure.
        """
        if len(config_idx) != self.dimensions:
            raise ValueError(
                f"is_pruned expects a full-length config of length {self.dimensions}"
            )
        return self._recursive_check(config_idx, self._failure_trie)

    def _recursive_check(self, config: Tuple, node: Dict, rank=0) -> bool:
        if node.get("_fail_"):
            return True  # Dominated by this failure rule
        if not config:
            return False  # Reached end of candidate without being dominated

        idx, sub_config = config[0], config[1:]
        for fail_idx, child_node in node.items():
            if fail_idx == "_fail_":
                continue
            if self._dim_compare(fail_idx, idx, rank):
                if self._recursive_check(sub_config, child_node, rank + 1):
                    return True
        return False

    def _is_prefix_doomed(self, prefix):
        """
        Internal helper for generators. Checks if a prefix is a "dead end" by
        testing if its most optimistic completion (padded with zeros) is pruned.
        """
        if not prefix:  # An empty prefix is never doomed
            return False

        padding_len = self.dimensions - len(prefix)
        optimistic_config = prefix + (0,) * padding_len
        return self.is_pruned(optimistic_config)

    def prune(self, failed_config):
        # We check the full config here to see if it's already dominated.
        if self.is_pruned(failed_config):
            return

        self._minimal_failures = [
            f
            for f in self._minimal_failures
            if not self._config_compare(failed_config, f)
        ]
        self._minimal_failures.append(failed_config)

        self._failure_trie.clear()
        for f in self._minimal_failures:
            node = self._failure_trie
            for idx in f:
                node = node.setdefault(idx, {})
            node.update(self._FAILURE_LEAF)

    def _generate_configs_recursively(self, prefix, sampler_func):
        """Generic recursive backtracking engine for generation."""
        if self._is_prefix_doomed(prefix):
            return None

        if len(prefix) == self.dimensions:
            return prefix

        dim = len(prefix)
        valid_indices = [i for i in range(self.shape[dim])]
        search_order = sampler_func(valid_indices)

        for idx in search_order:
            result = self._generate_configs_recursively(
                prefix + (idx,), sampler_func
            )
            if result is not None:
                return result
        return None

    def get_random_unpruned(self):
        def random_sampler(indices):
            random.shuffle(indices)
            return indices

        return self._generate_configs_recursively(tuple(), random_sampler)

    def get_mid_point_unpruned(self):
        def mid_point_sampler(indices):
            mid_idx = len(indices) // 2
            search_order = [indices[mid_idx]]
            i, j = mid_idx - 1, mid_idx + 1
            while i >= 0 or j < len(indices):
                if i >= 0:
                    search_order.append(indices[i])
                    i -= 1
                if j < len(indices):
                    search_order.append(indices[j])
                    j += 1
            return search_order

        return self._generate_configs_recursively(tuple(), mid_point_sampler)

    def generate_all_unpruned(self):
        def _backtrack(prefix):
            if self._is_prefix_doomed(prefix):
                return
            if len(prefix) == self.dimensions:
                yield prefix
                return
            for i in range(self.shape[len(prefix)]):
                yield from _backtrack(prefix + (i,))

        yield from _backtrack(tuple())

    def generate_all_unpruned_midpoint(self):
        def _midpoint_order(n):
            mid = n // 2
            yield mid
            lo, hi = mid - 1, mid + 1
            while lo >= 0 or hi < n:
                if lo >= 0:
                    yield lo
                    lo -= 1
                if hi < n:
                    yield hi
                    hi += 1

        def _backtrack(prefix):
            if self._is_prefix_doomed(prefix):
                return
            if len(prefix) == self.dimensions:
                yield prefix
                return
            for i in _midpoint_order(self.shape[len(prefix)]):
                yield from _backtrack(prefix + (i,))

        yield from _backtrack(tuple())


class CoordinateUnimodalFunction:
    """Prune a search space using coordinate-wise unimodality.

    Assumes that for each parameter d, with all others fixed (a "slice"),
    the function value is unimodal — a single minimum with non-decreasing
    tails on both sides.

    After each observation, per-dimension bounds [lo, hi] are tightened:
      - If a value to the right of the best is worse, the upper bound drops.
      - If a value to the left of the best is worse, the lower bound rises.
    Any config outside those bounds in any dimension is pruned.
    """

    def __init__(self, space: SpaceConfig):
        self.n_dims = len(space.shape)
        self.dim_sizes = list(space.shape)
        self._obs: list[dict] = [{} for _ in range(self.n_dims)]
        self._ranges: list[dict] = [{} for _ in range(self.n_dims)]

    def record(self, config_idx: tuple[int, ...], value: float):
        for d in range(self.n_dims):
            slice_key = config_idx[:d] + config_idx[d + 1 :]
            obs = self._obs[d].setdefault(slice_key, {})
            obs[config_idx[d]] = value

            if len(obs) < 2:
                continue

            best_idx = min(obs, key=obs.__getitem__)
            best_val = obs[best_idx]

            lo, hi = 0, self.dim_sizes[d] - 1

            for i in sorted(i for i in obs if i > best_idx):
                if obs[i] > best_val:
                    hi = i - 1
                    break

            for i in sorted((i for i in obs if i < best_idx), reverse=True):
                if obs[i] > best_val:
                    lo = i + 1
                    break

            if lo > 0 or hi < self.dim_sizes[d] - 1:
                self._ranges[d][slice_key] = (lo, hi)

    def is_pruned(self, config_idx: tuple[int, ...]) -> bool:
        for d in range(self.n_dims):
            slice_key = config_idx[:d] + config_idx[d + 1 :]
            if slice_key in self._ranges[d]:
                lo, hi = self._ranges[d][slice_key]
                if config_idx[d] < lo or config_idx[d] > hi:
                    return True
        return False
