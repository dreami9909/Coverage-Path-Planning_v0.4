"""2026-09-13 정합성 검토에서 확정된 결함의 회귀 검사.

네 건 모두 **독립 수치검증**(완전열거 · 유한차분)으로 잡았다. 읽기만으로는
안 보였던 것들이라, 같은 방식의 검사를 여기 고정한다.

1. 정사각 격자 ``cell_of`` — ``int()`` 절단으로 서쪽·남쪽 바깥 한 셀 폭이
   가장자리 셀로 들어갔다 (동쪽·북쪽은 정상 → belief 비대칭 왜곡).
2. SPX 절단평면 — 국소개선이 켜지면 master 자기 점에 접선이 안 놓여 하한이
   정체했다 (passes=2 에서 200 회 반복 뒤 gap 6%).
3. MAPPO critic — clip 이 물린 표본의 도함수를 0 이 아니라 clipped_error 로
   밀었다 (유한차분 대비 5~14 배).
4. MAPPO 은닉층 역전파 — 출력 가중치를 갱신한 **뒤** 그 가중치로 연쇄법칙을
   적용했다 (큰 스텝에서 0.4 배 어긋남).
"""

from __future__ import annotations

import itertools
import unittest

import numpy as np
from scipy import sparse
from dataclasses import replace

from cpp_search.core.models import MissionConfig, Point2D
from cpp_search.learning.mappo import MAPPORollout, NumpyMAPPO, _softmax
from cpp_search.planning.stone_spx import _SquareGrid
from cpp_search.theory import transitions as transition_ops
from cpp_search.theory.path_constrained import (
    PathConstrainedProblem,
    SearcherModel,
    joint_survival,
    path_detection_probability,
)
from cpp_search.theory.stone_path import (
    StoneTargetModel,
    stone_sp1_cutting_plane,
    stone_spx_cutting_plane,
)


class SquareGridCellOfTests(unittest.TestCase):
    def test_points_outside_the_square_are_outside_on_every_side(self) -> None:
        mission = MissionConfig(center=Point2D(0.0, 0.0), search_radius_m=5_436.3, uav_count=6)
        grid = _SquareGrid.inscribed(mission, 4, 4)
        margin = grid.half_side_m + 100.0
        for point in (
            Point2D(-margin, 0.0),
            Point2D(margin, 0.0),
            Point2D(0.0, -margin),
            Point2D(0.0, margin),
            Point2D(-margin, -margin),
        ):
            self.assertIsNone(grid.cell_of(point), f"{point} must be outside the grid")

    def test_cell_assignment_is_mirror_symmetric(self) -> None:
        mission = MissionConfig(center=Point2D(0.0, 0.0), search_radius_m=5_000.0, uav_count=6)
        grid = _SquareGrid.inscribed(mission, 4, 4)
        rng = np.random.default_rng(0)
        for _ in range(500):
            x, y = rng.uniform(-5_000.0, 5_000.0, size=2)
            east = grid.cell_of(Point2D(x, y))
            west = grid.cell_of(Point2D(-x, y))
            self.assertEqual(east is None, west is None, f"asymmetric at x={x:.1f}")
            if east is not None:
                # 열 인덱스가 거울상이어야 한다.
                self.assertEqual(east % grid.width, grid.width - 1 - west % grid.width)


def _small_spx_problem():
    S, T = 5, 3
    adjacency = np.zeros((S, S), dtype=bool)
    for a, b in ((0, 1), (1, 2), (2, 3), (0, 2)):
        adjacency[a, b] = adjacency[b, a] = True
    for i in range(S):
        adjacency[i, i] = True
    transit = np.ones((S, S))
    for a in range(4):
        for b in range(4):
            if a != b:
                transit[a, b] = 0.6 if abs(a - b) == 1 else 0.35
    searchers = tuple(
        SearcherModel(
            start,
            np.concatenate([np.full((T, 4), rate), np.zeros((T, 1))], axis=1),
            adjacency,
            transit,
        )
        for start, rate in zip((0, 3, 1), (0.9, 0.5, 0.7))
    )

    def target(seed: int) -> StoneTargetModel:
        rng = np.random.default_rng(seed)
        initial = np.append(rng.dirichlet(np.ones(4) * 2), 0.0)
        transitions = np.zeros((T - 1, S, S))
        for t in range(T - 1):
            for i in range(4):
                row = rng.dirichlet(np.ones(4))
                transitions[t, i, :4] = row * 0.97
                transitions[t, i, 4] = 0.03
            transitions[t, 4, 4] = 1.0
        return StoneTargetModel(initial, transitions, 1.0)

    targets = (target(11), target(12))
    problem = PathConstrainedProblem(
        targets[0].initial_mass, targets[0].transitions, searchers
    )
    return problem, targets, searchers, adjacency, (0, 3, 1)


def _brute_force_minimax(problem, targets, searchers, adjacency, starts, conflicts):
    S, T = problem.state_count, problem.time_count

    def feasible(paths) -> bool:
        for start, path in zip(starts, paths):
            source = start
            for cell in path:
                if not adjacency[source, cell]:
                    return False
                source = cell
        for t in range(T):
            dest = [path[t] for path in paths]
            if len(set(dest)) != len(dest):
                return False
            for a in range(len(paths)):
                for b in range(a + 1, len(paths)):
                    if (min(dest[a], dest[b]), max(dest[a], dest[b])) in conflicts:
                        return False
                    sa = starts[a] if t == 0 else paths[a][t - 1]
                    sb = starts[b] if t == 0 else paths[b][t - 1]
                    if sa == dest[b] and sb == dest[a] and sa != sb:
                        return False
        return True

    def worst(paths) -> float:
        return max(
            1.0
            - path_detection_probability(
                PathConstrainedProblem(tg.initial_mass, tg.transitions, searchers), paths
            )
            for tg in targets
        )

    best = 2.0
    for bundle in itertools.product(itertools.product(range(S), repeat=T), repeat=len(starts)):
        if feasible(bundle):
            best = min(best, worst(bundle))
    return best


class CuttingPlaneLocalImprovementTests(unittest.TestCase):
    def test_certificate_closes_with_local_improvement_enabled(self) -> None:
        problem, targets, searchers, adjacency, starts = _small_spx_problem()
        conflicts = ((0, 1),)
        optimum = _brute_force_minimax(problem, targets, searchers, adjacency, starts, set(conflicts))
        for passes in (0, 1, 2, 3):
            with self.subTest(passes=passes):
                result = stone_spx_cutting_plane(
                    problem,
                    targets=targets,
                    occupancy_limits=1,
                    destination_conflicts=conflicts,
                    forbid_opposing_edge_swaps=True,
                    relative_tolerance=1e-9,
                    max_iterations=60,
                    local_improvement_passes=passes,
                )
                self.assertTrue(result.converged, f"passes={passes} did not converge")
                self.assertAlmostEqual(result.upper_bound_nondetection, optimum, places=7)
                self.assertLessEqual(result.lower_bound_nondetection, optimum + 1e-9)
                self.assertLess(result.iterations, 30)


class InfeasibleWarmStartTests(unittest.TestCase):
    def test_an_infeasible_warm_start_is_discarded_not_fatal(self) -> None:
        """분리충돌을 어기는 warm 후보가 섞여 있어도 solve 는 진행돼야 한다."""

        problem, targets, searchers, adjacency, starts = _small_spx_problem()
        conflicts = ((0, 1),)
        # 셀 0 과 1 을 동시에 점유하는 후보 (하드 제약 위반)
        bad = ((0, 0, 0), (1, 1, 1), (2, 2, 2))
        result = stone_spx_cutting_plane(
            problem,
            targets=targets,
            occupancy_limits=1,
            destination_conflicts=conflicts,
            forbid_opposing_edge_swaps=True,
            relative_tolerance=1e-9,
            max_iterations=60,
            initial_path_candidates=(bad,),
        )
        self.assertEqual(result.discarded_warm_starts, 1)
        self.assertTrue(result.converged)
        optimum = _brute_force_minimax(problem, targets, searchers, adjacency, starts, set(conflicts))
        self.assertAlmostEqual(result.upper_bound_nondetection, optimum, places=7)


class WarmStartRecoveryTests(unittest.TestCase):
    """warm start 가 하나도 없어도 SPX 는 최적해와 인증을 내야 한다.

    H1/H2 는 예약이 도달 가능한 첫 수를 전부 막으면 InfeasiblePathError 를
    낸다 (6대가 축소 격자 중앙에 몰려 출발할 때 실제로 났다). 그때 후보를
    포기하고 남은 것으로 진행하는데, **전부 포기된 경우**가 최악이다.
    그 상태에서도 master 가 스스로 해를 찾는지 검사한다.
    """

    def test_a_clustered_launch_survives_every_heuristic_failing(self) -> None:
        """6대가 한 셀에서 출발하면 H1/H2 가 **둘 다** 실행 불가를 낸다.

        이송 후 TP 상공에서 탐색을 시작하는 조건이 정확히 그렇다. 예전에는
        후보 목록이 비어 ``max()`` 가 터지면서 solve 가 시작도 못 했다.
        warm start 는 가속 장치이지 필수가 아니다.
        """

        from cpp_search.planning.stone_spx import solve_stone_grid_paths
        from cpp_search.config import load_chapter_config
        from cpp_search.options import RunOptions
        from cpp_search.chapters.ch2_stone_spx import (
            build_instance, case_mission, case_specs, case_target_specs,
        )

        config = load_chapter_config("2")
        # 출발 고리비는 **연구 대상**이라 config 값이 움직인다. 이 회귀는 한
        # 점 출발에서만 재현되므로 여기서 직접 못박는다 — config 를 따라가면
        # 고리비를 바꿀 때마다 관계 없는 테스트가 깨진다.
        config.data["stone_spx"]["initial_position_ring_ratio"] = 0.0
        mission, sensor = config.mission(), config.sensor()
        case = case_specs(config)[0]
        targets = case_target_specs(config, case)
        base = dict(
            next(
                item
                for item in config.get("stone_spx.grid_conditions", [])
                if item.get("grid_kind") == "square"
            )
        )
        options = RunOptions(
            sample_count=10, particle_count=400, episode_count=1,
            target_profile="both", seed=1, map_error=False,
            communication_loss_probability=0.0, communication_latency_slices=0,
            mission_time_s=600.0,
        )
        grid = {
            **base, "grid_width": 4, "grid_height": 4, "time_slice_count": 3,
            "particle_count": 400, "max_iterations": 8, "master_time_limit_s": 20.0,
            "relative_tolerance": 1e-2, "square_fit": "circumscribed",
        }
        instance = build_instance(
            config, options, case_mission(mission, case), sensor,
            grid=grid, seed=20260201, terrain_weighting=True, targets=targets,
        )
        # 출발이 한 점이어야 이 회귀를 재현한다.
        self.assertEqual(len(set(instance.starts)), 1, instance.starts)
        solution = solve_stone_grid_paths(instance, method="stone-spx")
        self.assertEqual(len(solution.paths), mission.uav_count)
        self.assertTrue(np.isfinite(solution.upper_bound_nondetection))

    def test_solving_without_any_warm_start_still_certifies(self) -> None:
        problem, targets, searchers, adjacency, starts = _small_spx_problem()
        conflicts = ((0, 1),)
        optimum = _brute_force_minimax(
            problem, targets, searchers, adjacency, starts, set(conflicts)
        )
        result = stone_spx_cutting_plane(
            problem,
            targets=targets,
            occupancy_limits=1,
            destination_conflicts=conflicts,
            forbid_opposing_edge_swaps=True,
            relative_tolerance=1e-9,
            max_iterations=60,
            initial_paths=None,
            initial_path_candidates=(),
        )
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.upper_bound_nondetection, optimum, places=7)
        self.assertEqual(len(result.paths), len(problem.searchers))


class BoundHistoryTests(unittest.TestCase):
    """수렴 궤적이 유효한 인증 증거여야 한다.

    Ch2 의 주장은 "solver 가 수렴한다"이고, 그 근거는 반복별 상한·하한
    궤적뿐이다. 궤적이 단조가 아니면(하한이 내려가거나 상한이 올라가면)
    절단이 전역 유효하지 않다는 뜻이라 인증 자체가 무효가 된다.
    """

    def test_history_is_monotone_and_ends_at_the_reported_gap(self) -> None:
        problem, targets, searchers, adjacency, starts = _small_spx_problem()
        result = stone_spx_cutting_plane(
            problem,
            targets=targets,
            occupancy_limits=1,
            destination_conflicts=((0, 1),),
            forbid_opposing_edge_swaps=True,
            relative_tolerance=1e-9,
            max_iterations=60,
        )
        history = result.bound_history
        self.assertEqual(len(history), result.iterations)
        self.assertEqual([row[0] for row in history], list(range(1, result.iterations + 1)))

        lowers = [row[1] for row in history]
        uppers = [row[2] for row in history]
        # 하한은 절단이 쌓일수록 올라가기만, 상한은 좋은 정수해를 찾을수록
        # 내려가기만 해야 한다.
        for previous, current in zip(lowers, lowers[1:]):
            self.assertGreaterEqual(current, previous - 1e-12)
        for previous, current in zip(uppers, uppers[1:]):
            self.assertLessEqual(current, previous + 1e-12)
        # 하한이 상한을 절대 넘어서면 안 된다 (넘으면 최적해를 잘라낸 것).
        for _, lower, upper, gap in history:
            self.assertLessEqual(lower, upper + 1e-9)
            self.assertGreaterEqual(gap, 0.0)

        # 마지막 행이 보고된 값과 같아야 한다.
        self.assertAlmostEqual(history[-1][1], result.lower_bound_nondetection, places=12)
        self.assertAlmostEqual(history[-1][2], result.upper_bound_nondetection, places=12)
        self.assertAlmostEqual(history[-1][3], result.relative_optimality_gap, places=12)


class SquareGridFitTests(unittest.TestCase):
    """정사각 격자는 AOI 원을 **전부** 덮어야 한다.

    내접(v0.3 동작)은 원의 63.7% 만 덮어, 480 s 뒤 TEL belief 의 20.07% 가
    ``outside`` 흡수상태로 빠졌다. minimax 의 최악 표적이 TEL 이므로 그
    손실은 목적함수를 조용히 바꾼다. 셀을 아무리 늘려도 사라지지 않는다
    (한 변이 n 과 무관하게 고정이라서). 기본값이 되돌아가면 여기서 깨진다.
    """

    def setUp(self) -> None:
        self.mission = MissionConfig(
            center=Point2D(0.0, 0.0), search_radius_m=4_784.0, uav_count=6
        )

    def test_default_fit_covers_the_whole_aoi_circle(self) -> None:
        grid = _SquareGrid.fitted(self.mission, 5, 5, "circumscribed")
        self.assertAlmostEqual(grid.half_side_m, self.mission.search_radius_m, places=9)
        # 원 위의 어느 점도 격자 밖으로 나가지 않는다.
        for step in range(72):
            angle = step * (2.0 * np.pi / 72)
            x = self.mission.search_radius_m * np.cos(angle)
            y = self.mission.search_radius_m * np.sin(angle)
            self.assertIsNotNone(
                grid.cell_of(Point2D(float(x), float(y))),
                f"AOI 경계 {np.degrees(angle):.0f}° 가 격자 밖이다",
            )

    def test_inscribed_leaves_the_aoi_rim_uncovered(self) -> None:
        """내접이 왜 기각됐는지를 숫자로 남긴다."""

        grid = _SquareGrid.fitted(self.mission, 5, 5, "inscribed")
        outside = 0
        for step in range(72):
            angle = step * (2.0 * np.pi / 72)
            x = self.mission.search_radius_m * np.cos(angle)
            y = self.mission.search_radius_m * np.sin(angle)
            if grid.cell_of(Point2D(float(x), float(y))) is None:
                outside += 1
        self.assertGreater(outside, 0)

    def test_the_two_fits_never_compare_as_the_same_instance(self) -> None:
        """fingerprint 가 같으면 서로 다른 조건의 결과가 비교 가능해 보인다."""

        a = _SquareGrid.fitted(self.mission, 5, 5, "circumscribed")
        b = _SquareGrid.fitted(self.mission, 5, 5, "inscribed")
        self.assertNotEqual(a.fingerprint, b.fingerprint)

    def test_an_unknown_fit_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _SquareGrid.fitted(self.mission, 5, 5, "snug")


class SweptHazardTests(unittest.TestCase):
    """이동 중 소인을 지나간 셀에 나눠 줘도 SPX 인증이 성립해야 한다.

    이동 중에도 seeker gimbal 은 켜져 있으므로 소인량은 도착셀 하나가 아니라
    경로가 지나간 셀들에 쌓인다. 이는 Stone 의 ``alpha[l,k](j', j, t)`` 를
    소인 셀 j 로 한 겹 더 일반화한 것이고, X 에 대해 여전히 선형이라 master
    LP 와 절단평면이 그대로 성립해야 한다 — **완전열거로 확인한다.**
    """

    def _problem_with_swept(self):
        problem, targets, searchers, adjacency, starts = _small_spx_problem()
        rng = np.random.default_rng(2026)
        swept = {}
        for source in range(problem.state_count):
            for destination in range(problem.state_count):
                if not adjacency[source, destination]:
                    continue
                cells = {destination: float(rng.uniform(0.2, 0.8))}
                if source != destination:
                    # 지나온 셀에도 일부가 떨어진다.
                    cells[source] = float(rng.uniform(0.1, 0.5))
                swept[(source, destination)] = cells
        swept_searchers = tuple(
            replace(searcher, transit_fraction=None, swept_hazard=swept)
            for searcher in searchers
        )
        swept_problem = PathConstrainedProblem(
            targets[0].initial_mass, targets[0].transitions, swept_searchers
        )
        return swept_problem, targets, swept_searchers, adjacency, starts

    def test_certified_optimum_matches_brute_force(self) -> None:
        problem, targets, searchers, adjacency, starts = self._problem_with_swept()
        conflicts = ((0, 1),)
        optimum = _brute_force_minimax(
            problem, targets, searchers, adjacency, starts, set(conflicts)
        )
        result = stone_spx_cutting_plane(
            problem,
            targets=targets,
            occupancy_limits=1,
            destination_conflicts=conflicts,
            forbid_opposing_edge_swaps=True,
            relative_tolerance=1e-9,
            max_iterations=80,
        )
        self.assertTrue(result.converged, f"gap {result.relative_optimality_gap:.3e}")
        self.assertAlmostEqual(result.upper_bound_nondetection, optimum, places=7)
        # 하한이 최적해를 잘라내지 않았는가 — 인증의 유효성 그 자체.
        self.assertLessEqual(result.lower_bound_nondetection, optimum + 1e-9)

    def test_hazard_actually_lands_on_the_passed_cells(self) -> None:
        """분배가 실제로 일어나는지. 안 하면 위 테스트가 우연히 통과할 수 있다."""

        problem, _, searchers, _, _ = self._problem_with_swept()
        paths = ((1, 2, 1),) * len(problem.searchers)
        survival = joint_survival(problem, paths, 0)
        moved = [
            searcher.swept_cells(0, searcher.start_state, 1)
            for searcher in problem.searchers
        ]
        self.assertTrue(
            any(len(cells) > 1 for cells in moved), "이동 arc 가 한 셀만 훑는다"
        )
        expected = np.zeros(problem.state_count)
        for cells in moved:
            for cell, value in cells.items():
                expected[cell] += value
        np.testing.assert_allclose(survival, np.exp(-expected), rtol=1e-12)

    def test_sp1_refuses_swept_hazard(self) -> None:
        problem, targets, *_ = self._problem_with_swept()
        with self.assertRaises(ValueError):
            stone_sp1_cutting_plane(problem)


class SparseTransitionTests(unittest.TestCase):
    """희소 전이는 **근사가 아니다.** 조밀과 같은 수를 내야 한다.

    셀=탐지폭 격자(2,500셀 x 59슬라이스)는 조밀 배열로 표적 하나당 2.9 GB 라
    빌드 자체가 불가능하다. 0 을 적지 않으면 들어가는데, 0 은 곱해도 더해도
    0 이므로 결과가 달라질 이유가 없다. 그 사실을 여기 고정한다 — 값이
    갈리기 시작하면 큰 격자의 결과를 전혀 믿을 수 없게 된다.
    """

    def _sparsify(self, problem, targets):
        steps = tuple(
            sparse.csr_matrix(step)
            for step in transition_ops.as_sequence(problem.transitions)
        )
        sparse_problem = PathConstrainedProblem(
            problem.initial_mass, steps, problem.searchers
        )
        sparse_targets = tuple(
            StoneTargetModel(
                target.initial_mass,
                tuple(
                    sparse.csr_matrix(step)
                    for step in transition_ops.as_sequence(target.transitions)
                ),
                target.hazard_multiplier,
            )
            for target in targets
        )
        return sparse_problem, sparse_targets

    def test_solver_returns_the_same_answer(self) -> None:
        problem, targets, *_ = _small_spx_problem()
        sparse_problem, sparse_targets = self._sparsify(problem, targets)
        kwargs = dict(
            occupancy_limits=1,
            destination_conflicts=((0, 1),),
            forbid_opposing_edge_swaps=True,
            relative_tolerance=1e-9,
            max_iterations=60,
        )
        dense = stone_spx_cutting_plane(problem, targets=targets, **kwargs)
        thin = stone_spx_cutting_plane(
            sparse_problem, targets=sparse_targets, **kwargs
        )
        self.assertEqual(dense.paths, thin.paths)
        self.assertAlmostEqual(
            dense.upper_bound_nondetection, thin.upper_bound_nondetection, places=12
        )
        self.assertAlmostEqual(
            dense.lower_bound_nondetection, thin.lower_bound_nondetection, places=12
        )

    def test_forward_evaluation_matches_to_machine_precision(self) -> None:
        problem, targets, *_ = _small_spx_problem()
        sparse_problem, _ = self._sparsify(problem, targets)
        paths = ((1, 2, 1), (3, 2, 3), (1, 0, 1))
        self.assertAlmostEqual(
            path_detection_probability(problem, paths),
            path_detection_probability(sparse_problem, paths),
            places=14,
        )

    def test_the_whole_pipeline_runs_and_agrees(self) -> None:
        """단위 검사만으로는 부족했다 — 실제 파이프라인에서 터졌다.

        조밀 배열을 가정한 ``np.asarray(target.transitions)`` 가 네 군데
        남아 있었고, 큰 격자 스윕이 solve 단계에서 전부 죽었다 (빌드는
        성공해서 3시간을 쓰고 나서야 드러났다). 인스턴스 생성부터 인증까지
        **한 번 통과시켜야** 같은 종류의 누락을 잡는다.
        """

        from cpp_search.config import load_chapter_config
        from cpp_search.options import RunOptions
        from cpp_search.chapters.ch2_stone_spx import (
            build_instance,
            case_mission,
            case_specs,
            case_target_specs,
        )
        from cpp_search.planning.stone_spx import solve_stone_grid_paths

        config = load_chapter_config("2")
        mission, sensor = config.mission(), config.sensor()
        case = case_specs(config)[0]
        case_mission_config = case_mission(mission, case)
        targets = case_target_specs(config, case)
        base = dict(
            next(
                item
                for item in config.get("stone_spx.grid_conditions", [])
                if item.get("grid_kind") == "square"
            )
        )
        options = RunOptions(
            sample_count=10, particle_count=400, episode_count=1,
            target_profile="both", seed=1, map_error=False,
            communication_loss_probability=0.0, communication_latency_slices=0,
            mission_time_s=600.0,
        )
        answers = {}
        for flag in (False, True):
            grid = {
                **base,
                "grid_width": 4, "grid_height": 4, "time_slice_count": 3,
                "particle_count": 400, "sparse_transitions": flag,
                "max_iterations": 12, "master_time_limit_s": 20.0,
                "relative_tolerance": 1e-2, "square_fit": "circumscribed",
            }
            instance = build_instance(
                config, options, case_mission_config, sensor,
                grid=grid, seed=20260201, terrain_weighting=True, targets=targets,
            )
            answers[flag] = solve_stone_grid_paths(instance, method="stone-spx")
        dense, thin = answers[False], answers[True]
        # 경로까지 같기를 요구하지 않는다. 6대가 동일 기체라 역할을 맞바꾼
        # **동값 대칭해**가 여럿이고, MILP 가 타이브레이크를 다르게 할 수
        # 있다. 값이 같은지가 검사할 불변량이다.
        self.assertAlmostEqual(
            dense.upper_bound_nondetection, thin.upper_bound_nondetection, places=12
        )
        # 하한까지 비트 단위로 같기를 요구하면 안 된다. 두 경로가 만드는
        # 전이행렬은 1 ULP (2.2e-16) 만큼 다르다 — 근사가 아니라 부동소수점
        # **합산 순서** 차이이고 비영원소 개수는 완전히 같다. 그런데 절단평면
        # 은 "gap < 허용치면 정지"라는 **이산 분기**를 갖고 있어서, 그 1 비트가
        # 경계를 넘기면 반복이 한 번 더 돌고 하한이 1e-5 만큼 조여진다.
        # 불변량은 (1) 인증한 계획의 값이 같을 것(위), (2) 두 하한 모두 유효할
        # 것, (3) 두 쪽 다 선언한 허용치 안에서 닫힐 것 이다.
        for label, result in (("dense", dense), ("sparse", thin)):
            self.assertLessEqual(
                result.lower_bound_nondetection,
                result.upper_bound_nondetection + 1e-12,
                msg=f"{label}: 하한이 상한을 넘었다",
            )
            self.assertLessEqual(
                result.relative_optimality_gap, 1e-2 + 1e-12, msg=label
            )
        self.assertAlmostEqual(
            min(dense.target_detection_probabilities),
            min(thin.target_detection_probabilities),
            places=12,
        )

    def test_a_row_that_lost_all_particles_stays_put(self) -> None:
        """입자가 없던 셀은 조밀 경로와 같이 제자리 흡수여야 한다."""

        empty = sparse.csr_matrix(np.eye(4))
        transition_ops.validate((empty,), states=4, steps=1)


class SolverInstrumentationTests(unittest.TestCase):
    """예산을 어디에 넣을지, 그리고 SPX 가 휴리스틱을 이겼는지 기록한다.

    이게 없어서 두 가지를 추측으로 말했다. 하나는 "시간이 병목인가 반복이
    병목인가"를 궤적의 정지 구간으로 **추론**한 것이고 (정황이지 인과가
    아니다), 다른 하나는 최종 계획을 "SPX 가 최적화한 것"이라고 부른 것이다
    (warm start 휴리스틱 것일 수도 있었다).
    """

    def _solve(self, **overrides):
        problem, targets, *_ = _small_spx_problem()
        kwargs = dict(
            occupancy_limits=1,
            destination_conflicts=((0, 1),),
            forbid_opposing_edge_swaps=True,
            relative_tolerance=1e-9,
            max_iterations=40,
        )
        kwargs.update(overrides)
        return stone_spx_cutting_plane(problem, targets=targets, **kwargs)

    def test_every_solve_records_a_master_status(self) -> None:
        result = self._solve()
        self.assertTrue(result.master_status_history)
        masters = [row for row in result.master_status_history if row[1] == "master"]
        self.assertEqual(len(masters), result.iterations)
        for iteration, phase, status, seconds in result.master_status_history:
            self.assertIn(phase, {"master", "relaxation"})
            self.assertIn(status, {"optimal", "limit", "failed"})
            self.assertGreaterEqual(seconds, 0.0)
        # master 반복은 1..N, 완화는 음수로 구분된다.
        self.assertEqual([row[0] for row in masters], list(range(1, result.iterations + 1)))

    def test_relaxation_rounds_are_recorded_separately(self) -> None:
        result = self._solve(continuous_relaxation_iterations=2)
        rounds = [row for row in result.master_status_history if row[1] == "relaxation"]
        self.assertEqual(len(rounds), 2)
        self.assertEqual([row[0] for row in rounds], [-1, -2])

    def test_incumbent_source_names_where_the_plan_came_from(self) -> None:
        result = self._solve()
        self.assertTrue(
            result.incumbent_source == "warm-start"
            or result.incumbent_source.startswith("master-iteration-"),
            result.incumbent_source,
        )

    def test_a_solve_without_warm_start_has_no_baseline(self) -> None:
        """warm start 가 없으면 기준선도 없어야 한다 — 0 이나 임의값이면 안 된다."""

        result = self._solve(initial_paths=None, initial_path_candidates=())
        self.assertTrue(np.isnan(result.warm_start_nondetection))
        self.assertTrue(result.incumbent_source.startswith("master-iteration-"))


class MappoGradientTests(unittest.TestCase):
    """수동 역전파를 유한차분과 비교한다. 큰 스텝 + clip 활성 조건까지."""

    obs_dim, gs, A, K, Tn, H = 6, 7, 3, 4, 5, 8
    gamma, lam, clip, ent = 0.95, 0.9, 0.2, 0.01

    def _rollout(self, policy, rng):
        local = rng.normal(size=(self.Tn, self.A, self.obs_dim))
        states = rng.normal(size=(self.Tn, self.gs))
        actions = rng.integers(0, self.K, size=(self.Tn, self.A))
        rewards = rng.normal(size=(self.Tn, self.A)) * 3.0
        dones = np.zeros(self.Tn)
        dones[-1] = 1.0
        probabilities = policy.action_probabilities(local.reshape(-1, self.obs_dim))
        old = np.log(
            probabilities.reshape(self.Tn, self.A, self.K)[
                np.arange(self.Tn)[:, None], np.arange(self.A)[None, :], actions
            ]
            + 1e-12
        ) + rng.normal(scale=0.15, size=(self.Tn, self.A))
        return MAPPORollout(local, states, actions, old, rewards, dones, rng.normal(size=self.gs))

    @staticmethod
    def _critic(params, states):
        hidden = np.tanh(states @ params["critic_hidden_weight"] + params["critic_hidden_bias"])
        return hidden @ params["critic_weight"] + params["critic_bias"]

    def _numeric(self, objective, params, name):
        gradient = np.zeros_like(params[name])
        eps = 1e-6
        iterator = np.nditer(gradient, flags=["multi_index"])
        for _ in iterator:
            index = iterator.multi_index
            plus = {k: v.copy() for k, v in params.items()}
            plus[name][index] += eps
            minus = {k: v.copy() for k, v in params.items()}
            minus[name][index] -= eps
            gradient[index] = (objective(plus) - objective(minus)) / (2 * eps)
        return gradient

    def _params(self, policy, names):
        return {n: getattr(policy, n).copy() for n in names}

    def test_actor_gradient_matches_finite_differences_at_a_large_step(self) -> None:
        rng = np.random.default_rng(3)
        policy = NumpyMAPPO((self.obs_dim,), self.gs, self.A, self.K, hidden_size=self.H, seed=1)
        roll = self._rollout(policy, rng)
        critic_names = ["critic_hidden_weight", "critic_hidden_bias", "critic_weight", "critic_bias"]
        critic_params = self._params(policy, critic_names)
        values = self._critic(critic_params, roll.global_states)
        next_values = np.vstack([values[1:], self._critic(critic_params, roll.final_global_state[None, :])])
        delta = roll.rewards + self.gamma * (1 - roll.dones)[:, None] * next_values - values
        advantage = np.zeros_like(delta)
        running = 0.0
        for t in range(self.Tn - 1, -1, -1):
            running = delta[t] + self.gamma * self.lam * (1 - roll.dones[t]) * running
            advantage[t] = running
        normalized = (advantage - advantage.mean()) / max(advantage.std(), 1e-8)
        names = ["actor_hidden_weight", "actor_hidden_bias", "actor_weight", "actor_bias"]
        before = self._params(policy, names)

        def objective(p):
            hidden = np.tanh(roll.local_observations @ p["actor_hidden_weight"] + p["actor_hidden_bias"])
            prob = _softmax(hidden @ p["actor_weight"] + p["actor_bias"])
            selected = prob[np.arange(self.Tn)[:, None], np.arange(self.A)[None, :], roll.actions]
            ratio = np.exp(np.log(selected + 1e-12) - roll.old_log_probabilities)
            surrogate = np.minimum(ratio * normalized, np.clip(ratio, 1 - self.clip, 1 + self.clip) * normalized)
            entropy = -np.sum(prob * np.log(prob + 1e-12), axis=2)
            return surrogate.mean() + self.ent * entropy.mean()

        step = 0.3
        policy.update(
            roll, actor_learning_rate=step, critic_learning_rate=1e-9, gamma=self.gamma,
            gae_lambda=self.lam, clip_ratio=self.clip, entropy_coefficient=self.ent,
            epochs=1, optimizer="sgd",
        )
        for name in names:
            analytic = (getattr(policy, name) - before[name]) / step
            numeric = self._numeric(objective, before, name)
            relative = np.abs(analytic - numeric).max() / max(np.abs(numeric).max(), 1e-12)
            self.assertLess(relative, 1e-6, f"{name}: relative error {relative:.2e}")

    def test_critic_gradient_is_zero_where_the_value_clip_binds(self) -> None:
        """epoch 2 에서 |V - V_old| > clip 인 표본의 도함수는 0 이어야 한다."""

        rng = np.random.default_rng(3)
        policy = NumpyMAPPO((self.obs_dim,), self.gs, self.A, self.K, hidden_size=self.H, seed=1)
        roll = self._rollout(policy, rng)
        names = ["critic_hidden_weight", "critic_hidden_bias", "critic_weight", "critic_bias"]
        start = self._params(policy, names)
        values0 = self._critic(start, roll.global_states)
        next0 = np.vstack([values0[1:], self._critic(start, roll.final_global_state[None, :])])
        delta = roll.rewards + self.gamma * (1 - roll.dones)[:, None] * next0 - values0
        advantage = np.zeros_like(delta)
        running = 0.0
        for t in range(self.Tn - 1, -1, -1):
            running = delta[t] + self.gamma * self.lam * (1 - roll.dones[t]) * running
            advantage[t] = running
        returns = advantage + values0

        big = 0.5
        # 1 epoch 짜리 큰 스텝: V 가 V_old ± clip 을 넘어가게 만든다.
        policy.update(
            roll, actor_learning_rate=1e-9, critic_learning_rate=big, gamma=self.gamma,
            gae_lambda=self.lam, clip_ratio=self.clip, entropy_coefficient=0.0,
            epochs=1, optimizer="sgd",
        )
        middle = self._params(policy, names)
        binding = np.abs(self._critic(middle, roll.global_states) - values0) > self.clip
        self.assertGreater(binding.mean(), 0.5, "the test needs the clip to bind")

        def objective(p):
            value = self._critic(p, roll.global_states)
            clipped = values0 + np.clip(value - values0, -self.clip, self.clip)
            return -0.5 * np.mean(
                np.sum(np.maximum((returns - value) ** 2, (returns - clipped) ** 2), axis=1)
            )

        # 같은 상태에서 epochs=2 로 다시 돌려 2번째 epoch 의 스텝만 본다.
        policy2 = NumpyMAPPO((self.obs_dim,), self.gs, self.A, self.K, hidden_size=self.H, seed=1)
        policy2.update(
            roll, actor_learning_rate=1e-9, critic_learning_rate=big, gamma=self.gamma,
            gae_lambda=self.lam, clip_ratio=self.clip, entropy_coefficient=0.0,
            epochs=2, optimizer="sgd",
        )
        for name in names:
            analytic = (getattr(policy2, name) - middle[name]) / big
            numeric = self._numeric(objective, middle, name)
            relative = np.abs(analytic - numeric).max() / max(np.abs(numeric).max(), 1e-12)
            self.assertLess(relative, 1e-6, f"{name}: relative error {relative:.2e}")


if __name__ == "__main__":
    unittest.main()


class FingerprintCoverageTests(unittest.TestCase):
    """지문은 **문제를 바꾸는 모든 것**을 담아야 한다.

    ``common_input_fingerprint`` 는 "두 계획법이 같은 문제를 받았다"를
    증명하는 유일한 장치다. 그런데 예전 정의는 ``detection_rate`` 와
    ``adjacency`` 만 담아서, 출발셀만 다른 두 인스턴스가 같은 지문을 냈다 —
    고리비 실험에서 Tank 8x8 의 ring 0.21 과 0.45 가 실제로 같은 지문
    ``ec11f3d7c80daef0...`` 를 찍었다. 지문이 같다는 말이 곧 결과를 비교해도
    된다는 뜻이므로, 이건 조용히 잘못된 비교를 승인해 주는 결함이었다.

    희소 전이는 또 다른 구멍이었다. object 배열의 ``tobytes()`` 는 값이
    아니라 포인터라, 셀=탐지폭 격자에서는 **같은 입력이 실행마다 다른 지문**을
    냈을 것이다.
    """

    def _problem(self, *, start_states, swept=None):
        states = 3
        steps = 2
        detection = np.full((steps, states), 0.4)
        adjacency = np.ones((states, states), dtype=bool)
        searchers = tuple(
            SearcherModel(
                start_state=start,
                detection_rate=detection,
                adjacency=adjacency,
                swept_hazard=swept,
            )
            for start in start_states
        )
        transitions = tuple(np.eye(states) for _ in range(steps - 1))
        problem = PathConstrainedProblem(
            np.full(states, 1.0 / states), transitions, searchers
        )
        targets = (
            StoneTargetModel(np.full(states, 1.0 / states), transitions, 1.0),
        )
        return problem, targets

    def _fingerprint(self, problem, targets):
        from cpp_search.planning.stone_spx import _common_input_fingerprint

        mission = MissionConfig(
            center=Point2D(0.0, 0.0), search_radius_m=5_000.0, uav_count=2
        )
        grid = _SquareGrid.circumscribed(mission, 2, 2)
        return _common_input_fingerprint(problem, targets, seed=7, grid=grid)

    def test_start_cells_change_the_fingerprint(self) -> None:
        here, targets = self._problem(start_states=(0, 1))
        there, _ = self._problem(start_states=(0, 2))
        self.assertNotEqual(
            self._fingerprint(here, targets), self._fingerprint(there, targets)
        )

    def test_the_same_instance_still_agrees_with_itself(self) -> None:
        problem, targets = self._problem(start_states=(0, 1))
        twin, twin_targets = self._problem(start_states=(0, 1))
        self.assertEqual(
            self._fingerprint(problem, targets),
            self._fingerprint(twin, twin_targets),
        )

    def test_swept_hazard_changes_the_fingerprint(self) -> None:
        plain, targets = self._problem(start_states=(0, 1))
        swept, _ = self._problem(
            start_states=(0, 1), swept={(0, 1): {0: 0.3, 1: 0.2}}
        )
        self.assertNotEqual(
            self._fingerprint(plain, targets), self._fingerprint(swept, targets)
        )

    def test_sparse_transitions_hash_by_value_not_by_pointer(self) -> None:
        problem, targets = self._problem(start_states=(0, 1))
        steps = tuple(
            sparse.csr_matrix(step)
            for step in transition_ops.as_sequence(problem.transitions)
        )
        first = PathConstrainedProblem(
            problem.initial_mass, steps, problem.searchers
        )
        # 같은 값의 **다른 객체**. 포인터를 해시하면 여기서 갈린다.
        second = PathConstrainedProblem(
            problem.initial_mass,
            tuple(sparse.csr_matrix(step.toarray()) for step in steps),
            problem.searchers,
        )
        self.assertEqual(
            self._fingerprint(first, targets), self._fingerprint(second, targets)
        )


class SweepRateConventionTests(unittest.TestCase):
    """소인량은 W x (비행속도) x 시간이다 — 경로 모양과 무관하다.

    단위시간에 지면에 쌓이는 hazard 는 ``∫∫ gamma dx dz = W * v`` 이고,
    측방 적분 W 는 진행 **방향**에 대한 것이라 경로가 휘어도 변하지 않는다.
    weave 를 해도 비행속도는 그대로이므로 초당 소인량은 직선비행과 같다.

    전에는 체류 항만 중심선속도 v_c 를 써서 weave 비율(1.1398)만큼,
    즉 12.3 % 를 잃었다. 이동 항은 실제 거리를 썼으므로 두 항의 규약이
    달랐고, **제자리 대기가 이동보다 11 % 불리**해지는 편향이 생겼다.
    """

    def _grid_and_speeds(self):
        from cpp_search.planning.stone_spx import _SquareGrid

        mission = MissionConfig(
            center=Point2D(0.0, 0.0), search_radius_m=2_000.0, uav_count=2
        )
        return _SquareGrid.circumscribed(mission, 4, 4), 44.0, 400.0

    def test_dwelling_in_place_credits_the_flown_distance(self) -> None:
        from cpp_search.planning.stone_spx import _swept_hazard

        grid, speed, width = self._grid_and_speeds()
        slice_s = 60.0
        adjacency = np.eye(grid.cell_count, dtype=bool)
        swept = _swept_hazard(
            grid, adjacency, sweep_width_m=width, transit_speed_mps=speed,
            search_speed_mps=speed, slice_s=slice_s,
            cell_scale=np.ones(grid.cell_count),
        )
        area = sum(swept[(0, 0)].values()) * grid.cell_area_m2
        self.assertAlmostEqual(area, width * speed * slice_s, places=6)

    def test_every_arc_credits_the_same_total_sweep(self) -> None:
        """이동하든 머무르든 한 슬라이스의 총 소인량은 같아야 한다.

        비행속도가 같고 시간이 같으므로 총량도 같다. 달라지는 것은 그 양이
        **어느 셀에 쌓이느냐** 뿐이다. 두 항의 규약이 어긋나면 여기서 갈린다.
        """
        from cpp_search.planning.stone_spx import _swept_hazard, _adjacency

        grid, speed, width = self._grid_and_speeds()
        slice_s = 60.0
        adjacency = _adjacency(grid, reach_m=speed * slice_s)
        swept = _swept_hazard(
            grid, adjacency, sweep_width_m=width, transit_speed_mps=speed,
            search_speed_mps=speed, slice_s=slice_s,
            cell_scale=np.ones(grid.cell_count),
        )
        expected = width * speed * slice_s
        for (source, destination), cells in swept.items():
            distance = grid.center_of(source).distance_to(grid.center_of(destination))
            if distance > speed * slice_s:
                continue  # 슬라이스 안에 도착 못 하는 arc 는 체류분이 없다
            area = sum(cells.values()) * grid.cell_area_m2
            self.assertAlmostEqual(
                area, expected, delta=expected * 1e-9,
                msg=f"arc {source}->{destination} 소인량 {area:,.0f} != {expected:,.0f}",
            )


class CommonScoringTests(unittest.TestCase):
    """공통 채점 함수는 절단평면이 최적화한 것과 **같은** 목적함수를 재야 한다.

    ``score_paths`` 는 "모든 계획법이 통과하는 하나의 자"로 선언되어 있고,
    SPX 와 학습 계획법의 비교가 전부 이 함수 위에서 이루어진다. 그런데 표적
    hazard 배율을 ``detection_rate`` 에만 걸고 ``swept_hazard`` 에 걸지 않으면,
    경로 배분 소인이 켜진 순간 배율이 사라진다 — ``swept_cells`` 가
    ``swept_hazard`` 를 그대로 돌려주고 ``detection_rate`` 를 쓰지 않기 때문이다.

    증상은 조용하다. 값이 그럴듯하게 나오고, 단지 **다른 문제**의 답일 뿐이다.
    실제로 인증된 계획을 이 함수로 다시 재면 솔버가 보고한 상계보다 좋은 값이
    나왔다. 여기서 두 값이 일치하는지 고정한다.
    """

    def test_the_scorer_reproduces_the_certified_upper_bound(self) -> None:
        from cpp_search.planning.stone_spx import solve_stone_grid_paths
        from cpp_search.config import load_chapter_config
        from cpp_search.options import RunOptions
        from cpp_search.chapters.ch2_stone_spx import (
            build_instance, case_mission, case_specs, case_target_specs,
        )

        config = load_chapter_config("2")
        mission, sensor = config.mission(), config.sensor()
        case = case_specs(config)[0]
        targets = case_target_specs(config, case)
        # 배율이 1 이면 이 회귀가 재현되지 않는다. 1 이 아님을 먼저 못박는다.
        self.assertTrue(
            any(abs(float(t.hazard_multiplier) - 1.0) > 1e-9 for t in targets),
            "표적 hazard 배율이 전부 1 이라 이 검사가 무의미하다",
        )
        base = dict(
            next(
                item
                for item in config.get("stone_spx.grid_conditions", [])
                if item.get("grid_kind") == "square"
            )
        )
        options = RunOptions(
            sample_count=10, particle_count=400, episode_count=1,
            target_profile="both", seed=1, map_error=False,
            communication_loss_probability=0.0, communication_latency_slices=0,
            mission_time_s=600.0,
        )
        grid = {
            **base, "grid_width": 4, "grid_height": 4, "time_slice_count": 3,
            "particle_count": 400, "max_iterations": 6, "master_time_limit_s": 20.0,
            "relative_tolerance": 1e-2, "square_fit": "circumscribed",
        }
        instance = build_instance(
            config, options, case_mission(mission, case), sensor,
            grid=grid, seed=20260201, terrain_weighting=True, targets=targets,
        )
        solution = solve_stone_grid_paths(instance, method="stone-spx")
        # 상계는 정의상 "반환한 계획을 정확히 평가한 값"이다.
        self.assertAlmostEqual(
            min(instance.score_paths(solution.paths)),
            1.0 - solution.upper_bound_nondetection,
            places=10,
        )

    def test_the_multiplier_reaches_the_swept_hazard(self) -> None:
        """배율을 바꾸면 채점값이 따라 움직여야 한다."""
        from cpp_search.planning.stone_spx import _target_path_detection_probabilities

        states, steps = 3, 2
        detection = np.full((steps, states), 0.5)
        adjacency = np.ones((states, states), dtype=bool)
        swept = {(i, k): {k: 0.4, i: 0.2} for i in range(states) for k in range(states)}
        searcher = SearcherModel(
            start_state=0, detection_rate=detection, adjacency=adjacency,
            swept_hazard=swept,
        )
        transitions = (np.eye(states),)
        problem = PathConstrainedProblem(
            np.full(states, 1.0 / states), transitions, (searcher,)
        )
        paths = ((1, 2),)
        weak = StoneTargetModel(np.full(states, 1.0 / states), transitions, 0.5)
        strong = StoneTargetModel(np.full(states, 1.0 / states), transitions, 2.0)
        low, high = _target_path_detection_probabilities(
            problem, (weak, strong), paths
        )
        self.assertLess(low, high)


class RelaxationBudgetTests(unittest.TestCase):
    """연속완화 단계도 시간 한도 안에서 끝나야 하고, 하계는 유효해야 한다.

    두 가지를 한꺼번에 고정한다.

    1) **한도**: 예전에는 이 단계에만 시간 한도가 없었다. master 를 300 초로
       묶어도 전체 실행시간의 상한이 성립하지 않았고, 셀 = 탐지폭 격자에서
       24 시간을 넘겨도 반환되지 않았다.

    2) **하계의 유효성**: 완화의 목적값은 그대로 ``certified_lower`` 가 된다.
       그런데 한도에 걸린 LP 가 반환하는 값은 *실행가능점의* 목적값이므로
       최소화 문제에서는 **상계**다. 그것을 하계 자리에 넣으면 L <= OPT 가
       깨지고 인증 전체가 무의미해진다. 그래서 최적으로 풀린 경우에만
       하계를 갱신한다. 접평면은 평가점이 어디든 유효하므로 그대로 쓴다.
    """

    def _problem(self):
        states, steps = 4, 3
        rng = np.random.default_rng(11)
        detection = rng.random((steps, states)) * 0.8 + 0.2
        adjacency = np.ones((states, states), dtype=bool)
        searchers = tuple(
            SearcherModel(start_state=s, detection_rate=detection, adjacency=adjacency)
            for s in range(2)
        )
        transitions = tuple(np.eye(states) for _ in range(steps - 1))
        problem = PathConstrainedProblem(
            np.full(states, 1.0 / states), transitions, searchers
        )
        targets = (
            StoneTargetModel(np.full(states, 1.0 / states), transitions, 1.0),
            StoneTargetModel(np.full(states, 1.0 / states), transitions, 0.7),
        )
        return problem, targets

    def test_the_relaxation_phase_receives_the_time_limit(self) -> None:
        import inspect
        from cpp_search.theory import stone_path

        source = inspect.getsource(stone_path.stone_spx_cutting_plane)
        head = source[: source.index("persistent = (")]
        self.assertIn("relaxation_options", head)
        self.assertIn('relaxation_options["time_limit"]', head)

    def test_the_certificate_holds_with_the_relaxation_enabled(self) -> None:
        problem, targets = self._problem()
        solution = stone_spx_cutting_plane(
            problem,
            targets=targets,
            occupancy_limits=1,
            relative_tolerance=1e-9,
            max_iterations=60,
            continuous_relaxation_iterations=3,
            master_time_limit_s=20.0,
        )
        self.assertLessEqual(
            solution.lower_bound_nondetection,
            solution.upper_bound_nondetection + 1e-9,
            "하계가 상계를 넘었다 — 완화가 낸 값이 하계로 잘못 쓰였다",
        )
        # 상계는 정의상 반환한 계획의 정확 평가값이다.
        for target, value in zip(targets, solution.target_nondetection_probabilities):
            self.assertLessEqual(value, solution.upper_bound_nondetection + 1e-9)
