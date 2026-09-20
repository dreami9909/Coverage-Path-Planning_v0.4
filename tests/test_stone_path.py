"""Stone SP1/SPX mathematical-programming and cutting-plane tests."""

from __future__ import annotations

from itertools import product
from math import log
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import cpp_search.theory.stone_path as stone_path_module

from cpp_search.theory.path_constrained import (
    PathConstrainedProblem,
    SearcherModel,
    exact_best_paths,
    path_detection_probability,
)
from cpp_search.theory.stone_path import (
    StoneTargetModel,
    _nondetection_value_gradient,
    stone_sp1_cutting_plane,
    stone_spx_cutting_plane,
)


TRANSITIONS = np.array(
    [[0.7, 0.3, 0.0], [0.2, 0.6, 0.2], [0.0, 0.3, 0.7]],
    dtype=float,
)
ADJACENCY = np.array(
    [[1, 1, 0], [1, 1, 1], [0, 1, 1]],
    dtype=bool,
)
INITIAL = np.array([0.6, 0.3, 0.1], dtype=float)


def _problem(*, searchers: int = 2) -> PathConstrainedProblem:
    rate = np.full((3, 3), -log(0.4), dtype=float)
    return PathConstrainedProblem(
        initial_mass=INITIAL,
        transitions=np.stack([TRANSITIONS, TRANSITIONS]),
        searchers=tuple(
            SearcherModel(0, rate, ADJACENCY) for _ in range(searchers)
        ),
    )


class StoneSP1Tests(unittest.TestCase):
    def test_sp1_matches_complete_enumeration_and_certifies_zero_gap(self) -> None:
        problem = _problem()
        exact = exact_best_paths(problem)
        stone = stone_sp1_cutting_plane(problem)

        self.assertTrue(stone.converged)
        self.assertLessEqual(stone.relative_optimality_gap, 1e-8)
        self.assertAlmostEqual(
            stone.probability_of_detection,
            exact.probability_of_detection,
            places=10,
        )
        self.assertAlmostEqual(
            path_detection_probability(problem, stone.paths),
            stone.probability_of_detection,
            places=12,
        )
        self.assertLessEqual(
            stone.lower_bound_nondetection,
            stone.upper_bound_nondetection + 1e-12,
        )

    def test_sp1_rejects_heterogeneous_searchers(self) -> None:
        base = _problem(searchers=1)
        fast_rate = np.full((3, 3), -log(0.2), dtype=float)
        heterogeneous = PathConstrainedProblem(
            base.initial_mass,
            base.transitions,
            (base.searchers[0], SearcherModel(0, fast_rate, ADJACENCY)),
        )

        with self.assertRaisesRegex(ValueError, "homogeneous detection rates"):
            stone_sp1_cutting_plane(heterogeneous)

    def test_sp1_directs_arc_dependent_detection_to_spx(self) -> None:
        base = _problem(searchers=1)
        searcher = base.searchers[0]
        transit = np.ones((3, 3), dtype=float)
        transit[0, 1] = 0.5
        problem = PathConstrainedProblem(
            base.initial_mass,
            base.transitions,
            (
                SearcherModel(
                    searcher.start_state,
                    searcher.detection_rate,
                    searcher.adjacency,
                    transit,
                ),
            ),
        )

        with self.assertRaisesRegex(ValueError, "use SPX"):
            stone_sp1_cutting_plane(problem)
        self.assertTrue(stone_spx_cutting_plane(problem).converged)

    def test_markov_gradient_matches_finite_difference(self) -> None:
        problem = _problem(searchers=1)
        target = StoneTargetModel.from_problem(problem)
        hazard = np.array(
            [[0.2, 0.0, 0.4], [0.1, 0.3, 0.0], [0.0, 0.2, 0.1]],
            dtype=float,
        )
        value, gradient = _nondetection_value_gradient(target, hazard)
        epsilon = 1e-6
        perturbed = hazard.copy()
        perturbed[1, 1] += epsilon
        shifted, _ = _nondetection_value_gradient(target, perturbed)

        self.assertAlmostEqual(
            gradient[1, 1],
            (shifted - value) / epsilon,
            places=6,
        )

    def test_solver_retries_without_presolve_after_highs_failure(self) -> None:
        real_milp = stone_path_module.milp
        options_seen: list[dict] = []

        def fail_once(*args, **kwargs):
            options_seen.append(dict(kwargs.get("options", {})))
            if len(options_seen) == 1:
                return SimpleNamespace(success=False, message="presolve failure")
            return real_milp(*args, **kwargs)

        with patch.object(stone_path_module, "milp", side_effect=fail_once):
            result = stone_sp1_cutting_plane(_problem())

        self.assertTrue(result.converged)
        self.assertFalse(options_seen[1]["presolve"])

    def test_sp1_retains_a_feasible_path_when_best_detection_is_zero(self) -> None:
        problem = PathConstrainedProblem(
            initial_mass=np.array([0.0, 1.0]),
            transitions=np.array([np.eye(2)]),
            searchers=(
                SearcherModel(
                    0,
                    np.ones((2, 2)),
                    np.eye(2, dtype=bool),
                ),
            ),
        )

        result = stone_sp1_cutting_plane(problem)

        self.assertTrue(result.converged)
        self.assertEqual(result.paths, ((0, 0),))
        self.assertAlmostEqual(result.probability_of_detection, 0.0)
        self.assertAlmostEqual(result.upper_bound_nondetection, 1.0)

    def test_sp1_uses_an_explicit_exact_fallback_after_a_master_failure(self) -> None:
        with patch.object(
            stone_path_module,
            "stone_spx_cutting_plane",
            side_effect=RuntimeError("synthetic HiGHS failure"),
        ):
            result = stone_sp1_cutting_plane(_problem(searchers=1))

        exact = exact_best_paths(_problem(searchers=1))
        self.assertTrue(result.fallback_used)
        self.assertIn("HiGHS", result.fallback_reason)
        self.assertIn("exact-enumeration-fallback", result.method)
        self.assertAlmostEqual(
            result.probability_of_detection,
            exact.probability_of_detection,
        )
        self.assertEqual(
            result.lower_bound_nondetection,
            result.upper_bound_nondetection,
        )


class StoneSPXTests(unittest.TestCase):
    def test_spx_reduces_to_sp1_for_one_target_and_homogeneous_searchers(self) -> None:
        problem = _problem()
        sp1 = stone_sp1_cutting_plane(problem)
        spx = stone_spx_cutting_plane(problem)

        self.assertAlmostEqual(
            spx.probability_of_detection,
            sp1.probability_of_detection,
            places=10,
        )

    def test_spx_supports_heterogeneous_searchers_and_multiple_targets(self) -> None:
        base = _problem(searchers=1)
        second_rate = np.full((3, 3), -log(0.65), dtype=float)
        problem = PathConstrainedProblem(
            base.initial_mass,
            base.transitions,
            (
                base.searchers[0],
                SearcherModel(2, second_rate, ADJACENCY),
            ),
        )
        targets = (
            StoneTargetModel(INITIAL, base.transitions),
            StoneTargetModel(INITIAL[::-1], base.transitions, 0.75),
        )
        result = stone_spx_cutting_plane(
            problem,
            targets=targets,
            occupancy_limits=1,
        )

        self.assertTrue(result.converged)
        self.assertEqual(len(result.target_nondetection_probabilities), 2)
        self.assertAlmostEqual(
            result.upper_bound_nondetection,
            max(result.target_nondetection_probabilities),
            places=12,
        )
        for time in range(problem.time_count):
            self.assertEqual(len({path[time] for path in result.paths}), 2)

    def test_multitarget_spx_matches_independent_complete_enumeration(self) -> None:
        """Check the minimax SPX result with a separate path enumerator/evaluator."""

        base = _problem(searchers=1)
        second_rate = np.full((3, 3), -log(0.65), dtype=float)
        problem = PathConstrainedProblem(
            base.initial_mass,
            base.transitions,
            (base.searchers[0], SearcherModel(2, second_rate, ADJACENCY)),
        )
        targets = (
            StoneTargetModel(INITIAL, base.transitions),
            StoneTargetModel(INITIAL[::-1], base.transitions, 0.75),
        )

        def feasible_paths(searcher: SearcherModel) -> tuple[tuple[int, ...], ...]:
            partial = ((searcher.start_state, ()),)
            for _ in range(problem.time_count):
                partial = tuple(
                    (destination, (*path, destination))
                    for state, path in partial
                    for destination in searcher.successors(state)
                )
            return tuple(path for _, path in partial)

        target_problems = tuple(
            PathConstrainedProblem(
                target.initial_mass,
                target.transitions,
                tuple(
                    SearcherModel(
                        searcher.start_state,
                        searcher.detection_rate * float(target.hazard_multiplier),
                        searcher.adjacency,
                        searcher.transit_fraction,
                    )
                    for searcher in problem.searchers
                ),
            )
            for target in targets
        )
        exact_worst_pnd = 1.0
        for paths in product(*(feasible_paths(s) for s in problem.searchers)):
            if any(
                len({path[time] for path in paths}) != len(paths)
                for time in range(problem.time_count)
            ):
                continue
            exact_worst_pnd = min(
                exact_worst_pnd,
                max(
                    1.0 - path_detection_probability(target_problem, paths)
                    for target_problem in target_problems
                ),
            )

        result = stone_spx_cutting_plane(
            problem,
            targets=targets,
            occupancy_limits=1,
        )

        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.upper_bound_nondetection, exact_worst_pnd, places=10)

    def test_spx_enforces_incompatible_moves(self) -> None:
        problem = _problem()
        # Both searchers could choose 0->1 at t=0.  SPX equation (4.54)
        # forbids this particular pair while leaving the model feasible.
        result = stone_spx_cutting_plane(
            problem,
            incompatible_moves=(((0, 0, 0, 1), (1, 0, 0, 1)),),
        )

        self.assertFalse(result.paths[0][0] == result.paths[1][0] == 1)

    def test_aggregated_identical_searchers_matches_disaggregated_spx(self) -> None:
        problem = _problem()
        separate = stone_spx_cutting_plane(problem, occupancy_limits=1)
        aggregated = stone_spx_cutting_plane(
            problem,
            occupancy_limits=1,
            aggregate_identical_searchers=True,
        )

        self.assertTrue(aggregated.converged)
        self.assertAlmostEqual(
            aggregated.probability_of_detection,
            separate.probability_of_detection,
            places=10,
        )
        self.assertAlmostEqual(
            path_detection_probability(problem, aggregated.paths),
            aggregated.probability_of_detection,
            places=12,
        )

    def test_lazy_destination_conflict_is_enforced(self) -> None:
        problem = _problem()
        result = stone_spx_cutting_plane(
            problem,
            occupancy_limits=1,
            destination_conflicts=((1, 2),),
            max_iterations=100,
        )

        for time in range(problem.time_count):
            self.assertNotEqual(
                {result.paths[0][time], result.paths[1][time]},
                {1, 2},
            )

    def test_feasible_initial_paths_preserve_the_exact_optimum(self) -> None:
        problem = _problem()
        exact = stone_spx_cutting_plane(problem, occupancy_limits=1)
        warmed = stone_spx_cutting_plane(
            problem,
            occupancy_limits=1,
            initial_paths=exact.paths,
        )

        self.assertTrue(warmed.converged)
        self.assertAlmostEqual(
            warmed.probability_of_detection,
            exact.probability_of_detection,
            places=10,
        )

    def test_persistent_highs_master_matches_scipy_master(self) -> None:
        problem = _problem()
        scipy_result = stone_spx_cutting_plane(problem, occupancy_limits=1)
        persistent = stone_spx_cutting_plane(
            problem,
            occupancy_limits=1,
            persistent_master=True,
            relative_tolerance=1e-5,
        )

        self.assertTrue(persistent.converged)
        self.assertAlmostEqual(
            persistent.probability_of_detection,
            scipy_result.probability_of_detection,
            places=10,
        )

    def test_exact_survival_milp_matches_cutting_plane_optimum(self) -> None:
        problem = _problem()
        cutting_plane = stone_spx_cutting_plane(problem, occupancy_limits=1)
        exact_milp = stone_spx_cutting_plane(
            problem,
            occupancy_limits=1,
            exact_survival_milp=True,
            relative_tolerance=1e-8,
        )

        self.assertTrue(exact_milp.converged)
        self.assertEqual(exact_milp.method, "stone-spx-exact-survival-milp")
        self.assertAlmostEqual(
            exact_milp.probability_of_detection,
            cutting_plane.probability_of_detection,
            places=8,
        )
        self.assertLessEqual(exact_milp.relative_optimality_gap, 1e-8)


if __name__ == "__main__":
    unittest.main()


class IncompatibleMoveValidationTests(unittest.TestCase):
    """(4.54) 양립불가 이동을 넘길 때의 회귀 검사.

    v0.3 의 결함: ``_build_group_arcs`` 는 **그 시점에 도달 가능한** source 에서만
    arc 를 만드는데, 운용격자 호출부는 반대방향 edge swap 을 막으려고 모든 셀을
    source 로 열거했다. 그래서 존재하지 않는 arc 가 제약으로 들어와
    ``KeyError -> ValueError`` 가 났고, 그 호출부는 SPX 를 아예 돌릴 수 없었다
    (Ch5-b-2 와 관련 테스트 6건이 이 하나 때문에 실패했다).

    고친 방향: 도달 불가 arc 는 **공허한 제약**이므로 건너뛴다 — 비집계 SPX 의
    arc 변수 상한이 1 이라 남은 한 변수만으로 합이 1 을 넘지 못한다. 대신
    호출자가 문제를 잘못 기술한 경우(없는 탐색자, 범위 밖 상태·시각, 인접하지
    않은 이동)는 조용히 넘기지 않고 실패한다.
    """

    def test_an_unreachable_arc_makes_the_constraint_vacuous_not_an_error(self) -> None:
        problem = _problem()
        # 두 탐색자 모두 상태 0 에서 출발하므로 t=1 에 상태 2 에는 있을 수 없다.
        # 따라서 (searcher, t=1, 2 -> 1) arc 는 만들어지지 않는다.
        result = stone_spx_cutting_plane(
            problem,
            incompatible_moves=(((0, 1, 2, 1), (1, 1, 1, 2)),),
        )

        self.assertTrue(result.converged)
        unconstrained = stone_spx_cutting_plane(problem)
        self.assertAlmostEqual(
            result.probability_of_detection,
            unconstrained.probability_of_detection,
            places=10,
        )

    def test_a_nonadjacent_move_is_rejected(self) -> None:
        problem = _problem()
        with self.assertRaisesRegex(ValueError, "not an adjacent step"):
            stone_spx_cutting_plane(
                problem,
                incompatible_moves=(((0, 0, 0, 2), (1, 0, 0, 1)),),
            )

    def test_an_unknown_searcher_is_rejected(self) -> None:
        problem = _problem()
        with self.assertRaisesRegex(ValueError, "unknown searcher"):
            stone_spx_cutting_plane(
                problem,
                incompatible_moves=(((5, 0, 0, 1), (1, 0, 0, 1)),),
            )

    def test_a_time_outside_the_horizon_is_rejected(self) -> None:
        problem = _problem()
        with self.assertRaisesRegex(ValueError, "time outside the horizon"):
            stone_spx_cutting_plane(
                problem,
                incompatible_moves=(((0, 9, 0, 1), (1, 0, 0, 1)),),
            )

    def test_a_state_outside_the_grid_is_rejected(self) -> None:
        problem = _problem()
        with self.assertRaisesRegex(ValueError, "state outside the grid"):
            stone_spx_cutting_plane(
                problem,
                incompatible_moves=(((0, 0, 0, 7), (1, 0, 0, 1)),),
            )
