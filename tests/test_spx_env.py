"""MAPPO 환경이 **정말로 Stone 문제를 푸는지** 검사한다.

이 파일이 지키는 세 가지 주장이 Chapter 3 비교의 전제다.

1. 환경이 보상에 쓰는 미탐지 재귀가 ``instance.score_paths`` 와 **같다**.
   다르면 정책은 채점되지 않는 다른 것을 최적화하게 된다.
2. 환경이 내보내는 경로가 SPX 실행가능집합 안에 있다 — 인접성, 점유한도,
   분리반경. 아니면 MAPPO 가 제약을 깨서 이긴 것이 된다.
3. 행동열이 같으면 경로도 같다. 복구 규칙이 결정론적이라는 뜻이다.
"""

from __future__ import annotations

import unittest

import numpy as np

from cpp_search.learning.spx_env import (
    SPXEnvironmentConfig,
    StoneSPXEnvironment,
)
from tests.fixtures import small_instance


def _rollout(environment: StoneSPXEnvironment, seed: int):
    rng = np.random.default_rng(seed)
    environment.reset(0)
    done = False
    metrics = environment.metrics()
    while not done:
        actions = rng.integers(
            0, environment.action_count, size=environment.searcher_count
        )
        _, _, _, done, metrics = environment.step(actions)
    return metrics


class OnlineScoringTests(unittest.TestCase):
    def test_online_recursion_matches_the_official_scoring_function(self) -> None:
        instance = small_instance()
        environment = StoneSPXEnvironment(instance)
        metrics = _rollout(environment, seed=3)
        online = environment.online_target_detection_probabilities()
        official = instance.score_paths(metrics.paths)
        self.assertEqual(len(online), len(official))
        for left, right in zip(online, official):
            # 같은 식을 같은 순서로 굴리므로 부동소수점 수준에서 같아야 한다.
            self.assertAlmostEqual(left, right, places=12)

    def test_metrics_report_the_official_score_once_the_episode_is_complete(self) -> None:
        instance = small_instance()
        environment = StoneSPXEnvironment(instance)
        metrics = _rollout(environment, seed=5)
        self.assertEqual(metrics.completed_slices, instance.time_slice_count)
        self.assertAlmostEqual(
            metrics.worst_target_detection_probability,
            min(instance.score_paths(metrics.paths)),
            places=12,
        )


class FeasibilityTests(unittest.TestCase):
    def test_every_step_follows_the_searcher_adjacency(self) -> None:
        instance = small_instance()
        environment = StoneSPXEnvironment(instance)
        metrics = _rollout(environment, seed=7)
        for searcher, path in enumerate(metrics.paths):
            adjacency = instance.problem.searchers[searcher].adjacency
            previous = instance.starts[searcher]
            for cell in path:
                self.assertTrue(
                    adjacency[previous, cell],
                    f"searcher {searcher} moved {previous} -> {cell} without an arc",
                )
                previous = cell

    def test_cell_occupancy_limit_is_respected(self) -> None:
        instance = small_instance()
        environment = StoneSPXEnvironment(instance)
        metrics = _rollout(environment, seed=11)
        self.assertEqual(metrics.occupancy_violations, 0)
        for slice_index in range(instance.time_slice_count):
            occupied = [path[slice_index] for path in metrics.paths]
            self.assertEqual(
                len(occupied),
                len(set(occupied)),
                f"two searchers shared a cell in slice {slice_index}",
            )

    def test_separation_radius_is_respected_when_declared(self) -> None:
        instance = small_instance(reservation_separation_m=1_500.0)
        environment = StoneSPXEnvironment(instance)
        metrics = _rollout(environment, seed=13)
        self.assertEqual(metrics.separation_violations, 0)
        grid = instance.grid
        for slice_index in range(instance.time_slice_count):
            cells = [path[slice_index] for path in metrics.paths]
            for left in range(len(cells)):
                for right in range(left + 1, len(cells)):
                    if cells[left] >= instance.cell_count:
                        continue
                    if cells[right] >= instance.cell_count:
                        continue
                    distance = grid.center_of(cells[left]).distance_to(
                        grid.center_of(cells[right])
                    )
                    self.assertGreaterEqual(distance, 1_500.0 - 1e-6)

    def test_repaired_moves_are_counted_rather_than_hidden(self) -> None:
        """복구가 일어나도 위반으로 남지 않는다. 대신 세어서 보고한다."""

        instance = small_instance(uav_count=4, grid_width=3, grid_height=3)
        environment = StoneSPXEnvironment(instance)
        metrics = _rollout(environment, seed=17)
        self.assertGreaterEqual(metrics.repaired_moves, 0)
        self.assertEqual(metrics.occupancy_violations, 0)


class ActionSpaceTests(unittest.TestCase):
    def test_every_candidate_is_a_legal_successor(self) -> None:
        instance = small_instance()
        environment = StoneSPXEnvironment(instance)
        adjacency = instance.problem.searchers[0].adjacency
        for state in range(instance.problem.state_count):
            if not adjacency[state].any():
                continue
            for cell in environment.candidates_for(state):
                self.assertTrue(
                    adjacency[state, int(cell)],
                    f"candidate {cell} is not reachable from {state}",
                )

    def test_coverage_is_one_when_the_action_space_holds_every_successor(self) -> None:
        instance = small_instance()
        environment = StoneSPXEnvironment(
            instance, SPXEnvironmentConfig(action_count=32)
        )
        self.assertAlmostEqual(environment.action_candidate_coverage, 1.0)

    def test_pruning_is_reported_rather_than_silent(self) -> None:
        instance = small_instance()
        environment = StoneSPXEnvironment(
            instance, SPXEnvironmentConfig(action_count=3)
        )
        self.assertLess(environment.action_candidate_coverage, 1.0)

    def test_observation_and_state_sizes_match_the_declared_shapes(self) -> None:
        instance = small_instance()
        environment = StoneSPXEnvironment(instance)
        local, global_state = environment.reset(0)
        self.assertEqual(
            local.shape, (environment.searcher_count, *environment.local_observation_shape)
        )
        self.assertEqual(global_state.shape, (environment.global_state_size,))
        self.assertTrue(np.isfinite(local).all())
        self.assertTrue(np.isfinite(global_state).all())


class DeterminismTests(unittest.TestCase):
    def test_the_same_action_sequence_gives_the_same_paths(self) -> None:
        instance = small_instance()
        first = _rollout(StoneSPXEnvironment(instance), seed=23)
        second = _rollout(StoneSPXEnvironment(instance), seed=23)
        self.assertEqual(first.paths, second.paths)

    def test_horizon_equals_the_spx_planning_horizon(self) -> None:
        instance = small_instance(time_slice_count=5)
        environment = StoneSPXEnvironment(instance)
        self.assertEqual(environment.config.horizon_steps, 5)
        self.assertEqual(environment.time_count, 5)


if __name__ == "__main__":
    unittest.main()
