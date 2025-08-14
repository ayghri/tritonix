import unittest
from tritonix.utils.trie import MonotonicCascadeTrie


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
            set(self.trie._minimal_failures), set([(1, 3, 0, 0), (0, 4, 0,1)])
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
