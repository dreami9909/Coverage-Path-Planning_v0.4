"""MAPPO 학습 루프와 SPX 대비 비교가 주장하는 것만 주장하는지 검사한다."""

from __future__ import annotations

import math
import unittest

from cpp_search.learning.spx_env import SPXEnvironmentConfig
from cpp_search.learning.spx_policy import (
    StoneMAPPOSettings,
    compare_with_spx,
    mappo_path_solution,
    train_stone_mappo,
)
from cpp_search.planning.stone_spx import solve_stone_grid_paths
from tests.fixtures import small_instance


TRAINING = StoneMAPPOSettings(
    iterations=40, rollouts_per_iteration=6, hidden_size=32
)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.instance = small_instance()
        cls.result = train_stone_mappo(
            cls.instance,
            settings=TRAINING,
            environment_config=SPXEnvironmentConfig(action_count=9),
            seed=4_242,
        )

    def test_training_never_reports_worse_than_the_untrained_policy(self) -> None:
        """최선 경로는 미학습 평가를 포함해서 고른다. 따라서 단조여야 한다."""

        self.assertGreaterEqual(
            self.result.best_worst_target_detection_probability,
            self.result.untrained_worst_target_detection_probability,
        )
        self.assertGreaterEqual(self.result.learning_effect, 0.0)

    def test_the_policy_actually_learns_on_this_instance(self) -> None:
        """구현이 도는 것과 배우는 것은 다르다. 후자를 검사한다."""

        self.assertGreater(self.result.learning_effect, 0.0)

    def test_the_best_paths_are_scored_by_the_shared_function(self) -> None:
        scored = self.instance.score_paths(self.result.best_paths)
        self.assertAlmostEqual(
            min(scored),
            self.result.best_worst_target_detection_probability,
            places=12,
        )
        for left, right in zip(
            scored, self.result.best_target_detection_probabilities
        ):
            self.assertAlmostEqual(left, right, places=12)

    def test_the_learning_curve_has_one_row_per_iteration(self) -> None:
        self.assertEqual(len(self.result.learning_curve), TRAINING.iterations)
        self.assertEqual(self.result.learning_curve[0]["iteration"], 1)

    def test_the_policy_capacity_limit_is_recorded(self) -> None:
        capacity = self.result.policy_capacity
        self.assertIn("does_not_reproduce", capacity)
        self.assertIn("claim_limit", capacity)


class ClaimBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.instance = small_instance()
        cls.spx = solve_stone_grid_paths(cls.instance, method="stone-spx")
        cls.learned = train_stone_mappo(
            cls.instance, settings=TRAINING, seed=4_242
        )
        cls.mappo = mappo_path_solution(cls.instance, cls.learned)

    def test_mappo_never_claims_an_optimality_bound(self) -> None:
        self.assertTrue(math.isnan(self.mappo.lower_bound_nondetection))
        self.assertTrue(math.isnan(self.mappo.relative_optimality_gap))
        self.assertFalse(self.mappo.converged)

    def test_the_comparison_keeps_the_difference_direction_explicit(self) -> None:
        comparison = compare_with_spx(self.spx, self.mappo)
        self.assertAlmostEqual(
            comparison["spx_minus_mappo"],
            comparison["spx_worst_target_pd"] - comparison["mappo_worst_target_pd"],
            places=12,
        )
        self.assertFalse(comparison["mappo_certified"])

    def test_both_planners_saw_the_same_instance(self) -> None:
        """비교의 전제. 지문이 같지 않으면 비교값에 의미가 없다."""

        fingerprint = self.instance.fingerprint
        self.assertEqual(fingerprint, self.instance.fingerprint)
        self.assertEqual(len(self.spx.paths), len(self.mappo.paths))
        self.assertEqual(
            len(self.spx.paths[0]), len(self.mappo.paths[0])
        )

    def test_the_certified_spx_solution_is_not_beaten_by_the_learned_policy(self) -> None:
        """SPX 인증이 닫혔으면 그 값이 최적이다 — 넘을 수 있으면 버그다."""

        if not self.spx.converged:
            self.skipTest("SPX certificate is open on this instance")
        self.assertLessEqual(
            min(self.mappo.target_detection_probabilities),
            min(self.spx.target_detection_probabilities) + 1e-9,
        )


if __name__ == "__main__":
    unittest.main()
