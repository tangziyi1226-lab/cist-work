import importlib
import sys
import types
import unittest

import torch


# snapkv_utils imports flash_attn at module import time; allocation tests do not
# execute its kernels, so a small stub keeps this CPU test dependency-free.
flash_attn_stub = types.ModuleType("flash_attn")
flash_attn_stub.flash_attn_func = lambda *args, **kwargs: None
sys.modules.setdefault("flash_attn", flash_attn_stub)

snapkv = importlib.import_module("adaptive_snapkv.monkeypatch.snapkv_utils")


class EntroKVAllocationTest(unittest.TestCase):
    def make_cluster(self):
        return snapkv.AdaptiveSnapKVCluster(
            window_size=32,
            kernel_size=7,
            pooling="maxpool",
            base_capacity=128,
            floor_alpha=0.2,
            skip=0,
            layer_idx=3,
            num_hidden_layers=32,
            gqa_support=True,
            num_key_value_groups=4,
            gqa_func="mean",
            allocation_mode="entropy",
            entropy_alpha=0.5,
            entropy_baseline=0.3,
        )

    def test_budget_is_conserved_and_favors_high_entropy(self):
        cluster = self.make_cluster()
        cluster.last_entropy = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]])
        capacity = cluster._entropy_capacities(candidate_length=256)

        self.assertEqual(capacity.shape, (1, 8))
        self.assertEqual(int(capacity.sum()), 8 * (128 - 32))
        self.assertGreater(int(capacity[0, -1]), int(capacity[0, 0]))

    def test_gqa_entropy_has_one_value_per_kv_head(self):
        cluster = self.make_cluster()
        query = torch.randn(1, 32, 128, 16)
        key = torch.randn(1, 32, 128, 16)
        score = cluster.calcul_attn_sore(key, query)

        self.assertEqual(score.shape, (1, 8, 96))
        self.assertEqual(cluster.last_entropy.shape, (1, 8))
        self.assertTrue(torch.isfinite(cluster.last_entropy).all())
        self.assertTrue(((cluster.last_entropy >= 0) & (cluster.last_entropy <= 1)).all())

    def test_ratio_mode_matches_thirty_percent_total(self):
        cluster = self.make_cluster()
        cluster.budget_ratio = 0.3
        cluster._set_ratio_capacity(4096)
        self.assertEqual(cluster.base_capacity + cluster.window_size, round(4096 * 0.3))


if __name__ == "__main__":
    unittest.main()
