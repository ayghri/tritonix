import random
import itertools
from typing import Dict, Tuple


def compare_failure_config(failure: Tuple[int], candidate: Tuple[int]) -> bool:
    return all(c >= f for c, f in zip(candidate, failure))


def compare_failure_dim(failure: int, candidate: int, rank: int = 0) -> bool:
    return candidate >= failure


class MonotonicCascadeTrie:
    """
    Trie data structure to efficiently track and prune a monotonic search space.
    """

    def __init__(
        self,
        space_shape,
        config_comparison_func=compare_failure_config,
        dimension_comparison_func=compare_failure_dim,
    ):
        if not all(isinstance(s, int) and s > 0 for s in space_shape):
            raise ValueError(
                "space_shape must be a tuple of positive integers."
            )
        self.shape = space_shape
        self.dimensions = len(space_shape)
        self._minimal_failures = []
        self._failure_trie = {}
        self._FAILURE_LEAF = {"_fail_": True}
        self._config_compare = config_comparison_func
        self._dim_compare = dimension_comparison_func

    def is_pruned(self, config):
        """
        Checks if a full configuration is pruned by any known failure.
        A configuration is pruned if it is element-wise >= any minimal failure.
        """
        if len(config) != self.dimensions:
            raise ValueError(
                f"is_pruned expects a full-length config of length {self.dimensions}"
            )
        return self._recursive_check(config, self._failure_trie)

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


# --- Corrected Example Usage and Output ---
if __name__ == "__main__":
    shape = (4, 4, 3)
    trie = MonotonicCascadeTrie(shape)

    print(f"Initialized Trie for space of shape {shape}")
    print("Is (2, 2, 1) pruned initially?", trie.is_pruned((2, 2, 1)))

    print("\n--- Pruning a high-level config ---: ", (2, 1, 0))
    trie.prune((2, 1, 0))
    print("Is (1, 3, 2) pruned?", trie.is_pruned((1, 3, 2)))
    print("Is (3, 0, 1) pruned?", trie.is_pruned((3, 0, 1)))

    print("\n--- Pruning a more specific config ---:", (1, 2, 1))
    trie.prune((1, 2, 1))
    print("Is (1, 2, 0) pruned?", trie.is_pruned((1, 2, 0)))
    print("Is (1, 3, 1) pruned?", trie.is_pruned((1, 3, 1)))
    print("minimal failures after pruning:", trie._minimal_failures)

    print("\n--- Pruning a config that makes another redundant ---:", (1, 2, 0))
    # This new rule (1, 1, 0) makes the old rule (1, 2, 1) obsolete.
    trie.prune((1, 2, 0))
    print("minimal failures after pruning:", trie._minimal_failures)
    print(
        "Is (1, 2, 1) pruned now (by the new rule)?", trie.is_pruned((1, 2, 1))
    )

    print("\n--- Generation Methods Demo ---")

    k = 30
    print(f"\n1. Deterministic Generation (first {k} of all unpruned):")
    unpruned_generator = trie.generate_all_unpruned()
    for i, config in enumerate(itertools.islice(unpruned_generator, k)):
        print(f"  {i + 1}: {config}")

    print("\n2. Random Unpruned Generation:")
    for i in range(5):
        print(f"  Random sample {i + 1}: {trie.get_random_unpruned()}")

    print("\n3. Mid-Point Unpruned Generation:")
    print(f"  Mid-point sample: {trie.get_mid_point_unpruned()}")

    import unittest

    class TestTrieInitialization(unittest.TestCase):
        def test_valid_initialization(self):
            trie = MonotonicCascadeTrie((4, 5, 3))
            self.assertEqual(trie.shape, (4, 5, 3))

        def test_invalid_shape_zero(self):
            with self.assertRaises(ValueError):
                MonotonicCascadeTrie((4, 0, 3))

    class TestPruningLogic(unittest.TestCase):
        def setUp(self):
            self.trie = MonotonicCascadeTrie((4, 4, 4, 4))

        def test_simple_prune_and_check(self):
            self.trie.prune((2, 2, 0, 0))
            self.assertTrue(self.trie.is_pruned((2, 2, 0, 0)))
            self.assertTrue(self.trie.is_pruned((3, 3, 3, 3)))
            self.assertFalse(self.trie.is_pruned((2, 1, 3, 3)))

        def test_prune_replaces_specific_failure_with_general_one(self):
            self.trie.prune((2, 2, 2, 2))
            self.trie.prune((1, 1, 1, 1))
            self.assertEqual(self.trie._minimal_failures, [(1, 1, 1, 1)])

        def test_complex_pruning_reduction(self):
            """Should correctly reduce the minimal failures list."""
            self.trie.prune((1, 4, 4, 1))
            self.trie.prune((3, 3, 3, 0))
            self.trie.prune((0, 4, 0, 1))
            # This new failure should replace ALL previous failures.
            self.trie.prune((1, 3, 0, 0))
            self.assertEqual(
                set(self.trie._minimal_failures),
                set([(1, 3, 0, 0), (0, 4, 0, 1)]),
            )

        def test_generator_prefix_pruning_logic(self):
            """Tests the `_is_prefix_doomed` logic as you described."""
            self.trie.prune((1, 2, 0, 1))
            self.assertFalse(self.trie._is_prefix_doomed((1, 2)))
            self.assertFalse(self.trie._is_prefix_doomed((1, 2, 0)))

            # Now, pruning (1,1,3,3) should make (1,2) doomed, because its
            # optimistic completion (1,2,0,0) is NOT smaller than (1,1,3,3).
            # But pruning (1, 2, 0, 0) WILL make it doomed.
            self.trie.prune((1, 2, 0, 0))
            self.assertTrue(self.trie._is_prefix_doomed((1, 2)))
            self.assertTrue(self.trie._is_prefix_doomed((1, 2, 0)))
            # But a different prefix is still fine.
            self.assertFalse(self.trie._is_prefix_doomed((1, 1)))

    class TestGenerationMethods(unittest.TestCase):
        def test_generate_all_on_partially_pruned_trie(self):
            trie = MonotonicCascadeTrie((3, 3, 3))
            trie.prune((1, 1, 0))
            generated = list(trie.generate_all_unpruned())
            # Total=27. Pruned by (1,1,0) = 2*2*3 = 12. Valid = 27-12=15.
            self.assertEqual(len(generated), 15)
            self.assertNotIn((1, 1, 0), generated)
            self.assertNotIn((2, 2, 2), generated)
            self.assertIn((1, 0, 2), generated)
            self.assertIn((0, 2, 1), generated)

        def test_random_generator_returns_valid_configs(self):
            trie = MonotonicCascadeTrie((4, 4, 3))
            trie.prune((2, 1, 0))
            trie.prune((1, 2, 0))
            for _ in range(20):
                config = trie.get_random_unpruned()
                self.assertIsNotNone(config)
                self.assertFalse(trie.is_pruned(config))

        def test_generators_on_fully_pruned_trie(self):
            trie = MonotonicCascadeTrie((2, 2))
            trie.prune((0, 0))
            self.assertIsNone(trie.get_random_unpruned())
            self.assertIsNone(trie.get_mid_point_unpruned())
            self.assertEqual(list(trie.generate_all_unpruned()), [])

    if __name__ == "__main__":
        unittest.main(argv=["first-arg-is-ignored"], exit=False)
