"""MAPPO 로 Stone SPX 를 해석한다 — **같은 문제, 다른 해법.**

왜 이 모듈이 있나
-----------------
Stone 절단평면은 최적성 인증을 준다. 대신 격자가 커지면 정수 master 의 전역
하한 증명이 무너진다 (v0.3 Ch6 고해상도 감사: 240셀·5슬라이스·6기에서 13분
27초를 써도 상대 PND gap 15.18%, 사전선언 기준 1% 실패). 즉 **인증은 되지만
확장이 안 된다.**

MAPPO 는 그 반대다. 인증은 못 주지만 상태공간이 커져도 정책 평가 비용이
선형이다. 그래서 이 모듈은 SPX 를 "MAPPO 로 근사해석"한다.

    같은 StoneGridInstance
        ├── solve_stone_grid_paths      절단평면 -> 경로 + 인증폭
        └── solve_with_mappo (여기)     학습 정책 -> 경로 (인증 없음)
                    │
                    └── 두 경로를 instance.score_paths 하나로 채점

**비교가 성립하는 근거는 인스턴스를 공유한다는 것 하나다.** 격자·슬라이스·
belief·hazard·출발셀·예약반경이 전부 같고, 그 사실이 두 결과의 같은
``common_input_fingerprint`` 로 증명된다. 다른 것은 셀 경로를 고르는 방법뿐이다.

환경 정의 (``StoneSPXEnvironment``)
-----------------------------------
* **에이전트** = 탐색자. 한 스텝 = SPX 의 한 시간 슬라이스. 에피소드 길이 =
  ``time_slice_count``. 즉 MAPPO 의 에피소드와 SPX 의 계획지평이 같다.

* **행동** = 인접 셀 선택. 극좌표 운용격자는 셀당 후보가 수십 개까지 가므로
  고정 행동수 ``K`` 로 **가지친다**. 가지치기 순위는 belief 에서 오는 정적
  점수 ``mean_t m_t(c) * hazard(c)`` 이고, 제자리(self)는 항상 0번에 넣는다.
  이것은 MAPPO 를 SPX 실행가능집합의 **부분집합**으로 제한한다 — MAPPO 가
  제약을 깨서 이기는 일은 구조적으로 불가능하고, 그 대가로 잃은 후보 비율을
  ``action_candidate_coverage`` 로 보고한다.

* **점유·분리 복구**: 동시 행동선택이라 두 기체가 같은 셀을 고를 수 있다.
  낮은 번호부터 순서대로 확정하고, 이미 예약된 셀(분리반경 포함)을 고른
  기체는 자기 후보 중 다음 순위로 밀린다. 그래서 **환경이 내보내는 경로는
  항상 SPX 와 같은 실행가능집합 안에 있다.** 복구 횟수는 감춰지지 않고
  ``repaired_moves`` 로 나온다 (정책이 얼마나 충돌을 못 피했는지의 척도).

* **보상**: minimax 목적은 분해되지 않으므로 두 항으로 성형한다.

      slice 보상 = sum_i w_i * (그 슬라이스에 표적 i 에서 깎은 질량)
                   w_i ∝ exp(-PD_i^누적 / tau)      (뒤처진 표적에 가중)
      종단 보상 = terminal_weight * min_i PD_i       (= SPX 목적함수)

  팀 보상은 전원에게 같이 주고, 복구된 행동에만 개별 벌점을 준다.

* **미탐지 재귀**: ``theory/path_constrained.nondetection_trace`` 와 **같은
  식**을 온라인으로 굴린다. 두 값이 어긋나면 보상이 채점과 다른 것을 재게
  되므로, ``tests/test_spx_env.py`` 가 동일성을 검사한다.

의존
----
* 위: ``planning/stone_spx``(인스턴스·채점·경로변환), ``theory/path_constrained``,
  ``learning/mappo``.
* 아래: ``chapters/ch3_mappo_stone``, ``chapters/ch4_synthetic_terrain``.

참고: Su and Qian (2023) — MAPPO 학습규칙. 원 논문의 격자 도메인이 아니라
Stone 인스턴스에 적용한 것이므로 "논문 재현"이 아니다.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from cpp_search.theory import transitions as transition_ops

from cpp_search.planning.stone_spx import StoneGridInstance


#: 후보 셀 하나를 설명하는 관측 채널 수.
CANDIDATE_FEATURE_COUNT = 7

#: 후보와 무관한 전역 스칼라 관측 수.
SCALAR_FEATURE_COUNT = 5


@dataclass(frozen=True, slots=True)
class SPXEnvironmentConfig:
    """환경 쪽 조건. 학습 하이퍼파라미터는 :class:`StoneMAPPOSettings` 에 있다."""

    #: 에피소드 길이. ``StoneSPXEnvironment`` 가 인스턴스에서 읽어 채운다.
    horizon_steps: int = 0
    #: 고정 행동 수 K. 셀 차수가 이보다 크면 belief 점수로 가지친다.
    action_count: int = 9
    #: 뒤처진 표적에 얼마나 몰아줄지. 작을수록 최악표적 하나만 본다.
    worst_target_temperature: float = 0.05
    #: 종단 보상 계수. minimax 목적함수 자체를 보상에 넣는 항.
    terminal_weight: float = 10.0
    #: slice 보상 배율. 질량 증분이 0.01 수준이라 그대로 쓰면 신호가 약하다.
    slice_weight: float = 20.0
    #: 점유·분리 복구가 일어난 기체에 주는 개별 벌점.
    repair_penalty: float = 0.05

    def __post_init__(self) -> None:
        if self.action_count < 2:
            raise ValueError("action_count must allow at least two candidates")
        if self.worst_target_temperature <= 0.0:
            raise ValueError("worst_target_temperature must be positive")
        if self.terminal_weight < 0.0 or self.slice_weight < 0.0:
            raise ValueError("reward weights cannot be negative")
        if self.repair_penalty < 0.0:
            raise ValueError("repair_penalty cannot be negative")


@dataclass(frozen=True, slots=True)
class SPXSearchMetrics:
    """한 에피소드가 만든 경로와, 그 경로에 대해 말할 수 있는 것."""

    paths: tuple[tuple[int, ...], ...]
    target_detection_probabilities: tuple[float, ...]
    worst_target_detection_probability: float
    repaired_moves: int
    occupancy_violations: int
    separation_violations: int
    completed_slices: int


class StoneSPXEnvironment:
    """SPX 인스턴스 하나를 다중에이전트 강화학습 환경으로 노출한다."""

    def __init__(
        self,
        instance: StoneGridInstance,
        config: SPXEnvironmentConfig | None = None,
    ) -> None:
        self.instance = instance
        declared = config or SPXEnvironmentConfig()
        self.config = replace(declared, horizon_steps=instance.time_slice_count)
        problem = instance.problem
        self.searcher_count = len(problem.searchers)
        self.state_count = problem.state_count
        self.time_count = instance.time_slice_count
        self.target_count = len(instance.target_models)

        # 표적별 hazard 배율을 미리 곱해 둔다. 온라인 미탐지 재귀가
        # nondetection_trace 와 같은 값을 내려면 여기서 같은 배율을 써야 한다.
        self._target_initial = np.stack(
            [
                np.asarray(target.initial_mass, dtype=float)
                for target in instance.target_models
            ]
        )
        self._target_transitions = [
            transition_ops.as_sequence(target.transitions)
            for target in instance.target_models
        ]
        self._target_multiplier = np.asarray(
            [
                float(np.asarray(target.hazard_multiplier, dtype=float))
                for target in instance.target_models
            ],
            dtype=float,
        )
        if any(
            np.asarray(target.hazard_multiplier, dtype=float).ndim != 0
            for target in instance.target_models
        ):
            raise ValueError(
                "MAPPO interpretation of SPX requires scalar target hazard multipliers"
            )

        self._candidates, self._candidate_counts = self._build_candidates()
        # 폴백용: 상태별 전체 인접셀을 belief 정적 점수 순으로.
        adjacency = np.asarray(instance.problem.searchers[0].adjacency, dtype=bool)
        priority = self._static_priority()
        self._successor_order = tuple(
            tuple(
                sorted(
                    (int(k) for k in np.flatnonzero(adjacency[state])),
                    key=lambda k: (-priority[k], k),
                )
            )
            for state in range(instance.problem.state_count)
        )
        self._reservations = instance.reservation_neighborhoods
        self._occupancy_limit = max(1, int(instance.config.occupancy_limit))
        self._search_free = self._search_free_marginals()

        self.positions = np.asarray(instance.starts, dtype=int)
        self.step_count = 0
        self._mass = self._target_initial.copy()
        self._paths: list[list[int]] = [[] for _ in range(self.searcher_count)]
        self._detected = np.zeros(self.target_count, dtype=float)
        self._repaired_moves = 0
        self._occupancy_violations = 0
        self._separation_violations = 0

    # ------------------------------------------------------------------ setup

    def _build_candidates(self) -> tuple[np.ndarray, np.ndarray]:
        """상태별 고정 크기 후보 셀 표. 제자리를 0번에 둔다.

        belief 정적 점수로 가지치는 이유는 학습 중에 행동의 **의미가 바뀌지
        않아야** 하기 때문이다. 잔존질량으로 매 스텝 다시 정렬하면 같은 행동
        인덱스가 슬라이스마다 다른 셀을 뜻해서 정책이 배울 대상이 없다.
        """

        instance = self.instance
        problem = instance.problem
        cell_count = instance.cell_count
        priority = self._static_priority()

        width = self.config.action_count
        table = np.zeros((problem.state_count, width), dtype=int)
        counts = np.zeros(problem.state_count, dtype=int)
        # 모든 탐색자가 같은 adjacency 를 쓴다 (인스턴스 구축이 그렇게 만든다).
        adjacency = np.asarray(problem.searchers[0].adjacency, dtype=bool)
        for state in range(problem.state_count):
            successors = [int(k) for k in np.flatnonzero(adjacency[state])]
            if not successors:
                table[state, :] = state
                counts[state] = 1
                continue
            ordered = sorted(
                (k for k in successors if k != state),
                key=lambda k: (-priority[k] if k < cell_count else 0.0, k),
            )
            if state in successors:
                ordered.insert(0, state)
            counts[state] = len(ordered)
            kept = ordered[:width]
            while len(kept) < width:
                kept.append(kept[-1])
            table[state, :] = kept
        return table, counts

    def _static_priority(self) -> np.ndarray:
        """셀별 정적 점수 = 지평 평균 표적질량 × 평균 hazard. 후보 순위 전용.

        표적 0 의 전이가 아니라 **대표 전이**(표적 평균)를 쓴다.
        ``problem.transitions`` 가 그 평균이고, SPX 의 warm start 도 같은
        것을 본다. 표적 하나만 보면 두 표적이 갈리는 방향에서 후보 순위가
        한쪽으로 기울고, 그 기울기가 행동 인덱스의 의미에 박힌다.
        """

        problem = self.instance.problem
        marginals = np.zeros(problem.state_count, dtype=float)
        mass = self._target_initial.mean(axis=0)
        marginals += mass
        representative = transition_ops.as_sequence(problem.transitions)
        for time_index in range(self.time_count - 1):
            mass = mass @ representative[time_index]
            marginals += mass
        hazard = np.asarray(problem.searchers[0].detection_rate, dtype=float).mean(axis=0)
        return marginals * (hazard + 1e-12)

    def _search_free_marginals(self) -> np.ndarray:
        """탐색 영향 없는 표적 주변분포. 관측 정규화 상수로만 쓴다."""

        marginals = np.zeros(
            (self.target_count, self.time_count, self.state_count), dtype=float
        )
        for index in range(self.target_count):
            marginals[index, 0] = self._target_initial[index]
            for time_index in range(1, self.time_count):
                marginals[index, time_index] = (
                    marginals[index, time_index - 1]
                    @ self._target_transitions[index][time_index - 1]
                )
        return marginals

    # ------------------------------------------------------------ properties

    @property
    def local_observation_shape(self) -> tuple[int, ...]:
        return (
            self.config.action_count * CANDIDATE_FEATURE_COUNT + SCALAR_FEATURE_COUNT,
        )

    @property
    def global_state_size(self) -> int:
        return 2 * self.state_count + 2 * self.searcher_count + self.target_count + 1

    @property
    def action_count(self) -> int:
        return self.config.action_count

    def candidates_for(self, state: int) -> tuple[int, ...]:
        """그 상태에서 정책이 고를 수 있는 셀 목록. 행동 인덱스 순서다."""

        return tuple(int(cell) for cell in self._candidates[int(state)])

    @property
    def action_candidate_coverage(self) -> float:
        """가지치기가 남긴 후보 비율. 1.0 이면 SPX 실행가능집합 전체."""

        reachable = self._candidate_counts[self._candidate_counts > 0]
        if reachable.size == 0:
            return 1.0
        kept = np.minimum(reachable, self.config.action_count)
        return float(np.mean(kept / reachable))

    # ----------------------------------------------------------------- rollout

    def reset(self, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        del seed  # 환경은 결정론적이다. 확률성은 정책의 행동표본추출에서만 온다.
        self.positions = np.asarray(self.instance.starts, dtype=int)
        self.step_count = 0
        self._mass = self._target_initial.copy()
        self._paths = [[] for _ in range(self.searcher_count)]
        self._detected = np.zeros(self.target_count, dtype=float)
        self._repaired_moves = 0
        self._occupancy_violations = 0
        self._separation_violations = 0
        return self.local_observations(), self.global_state()

    def step(
        self,
        actions,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, SPXSearchMetrics]:
        selected = np.asarray(tuple(actions), dtype=int)
        if selected.shape != (self.searcher_count,):
            raise ValueError("one action is required per searcher")
        if np.any((selected < 0) | (selected >= self.config.action_count)):
            raise ValueError("action index is outside the SPX candidate space")

        time_index = self.step_count
        destinations, repaired = self._resolve_destinations(selected)
        rewards = np.zeros(self.searcher_count, dtype=float)
        rewards -= self.config.repair_penalty * repaired
        self._repaired_moves += int(repaired.sum())

        hazard = self._slice_hazard(time_index, destinations)
        weights = self._worst_target_weights()
        slice_gain = 0.0
        for index in range(self.target_count):
            survival = np.exp(-hazard * self._target_multiplier[index])
            before = float(self._mass[index].sum())
            self._mass[index] = self._mass[index] * survival
            after = float(self._mass[index].sum())
            self._detected[index] += before - after
            slice_gain += weights[index] * (before - after)
            if time_index < self.time_count - 1:
                self._mass[index] = (
                    self._mass[index] @ self._target_transitions[index][time_index]
                )
        rewards += self.config.slice_weight * slice_gain

        for searcher in range(self.searcher_count):
            self._paths[searcher].append(int(destinations[searcher]))
        self.positions = destinations
        self.step_count += 1
        done = self.step_count >= self.time_count
        if done:
            rewards += self.config.terminal_weight * float(self._detected.min())
        return (
            self.local_observations(),
            self.global_state(),
            rewards,
            done,
            self.metrics(),
        )

    def _resolve_destinations(self, selected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """점유한도와 분리반경을 지켜 목적 셀을 확정한다.

        낮은 번호 기체가 먼저 예약한다. 예약이 결정론적이므로 같은 정책 +
        같은 행동열은 항상 같은 경로를 낸다.
        """

        destinations = np.zeros(self.searcher_count, dtype=int)
        repaired = np.zeros(self.searcher_count, dtype=float)
        occupancy: dict[int, int] = {}
        blocked: set[int] = set()
        for searcher in range(self.searcher_count):
            current = int(self.positions[searcher])
            candidates = self._candidates[current]
            order = [int(candidates[selected[searcher]])]
            order.extend(int(cell) for cell in candidates)
            order.append(current)
            # 고정 K 후보가 전부 예약에 막히면 **전체 인접셀**로 물러난다.
            # SPX 실행가능집합은 인접셀 전부이므로 이 폴백도 그 안이다.
            # 운용 격자(288셀·800 m 분리·8슬라이스)에서 K=9 후보만으로는
            # seed 당 위반이 1,500 건을 넘었다 — 그러면 "같은 실행가능집합"
            # 이라는 비교 전제가 깨진다.
            order.extend(
                int(cell)
                for cell in self._successor_order[current]
                if cell not in order
            )
            chosen: int | None = None
            for rank, cell in enumerate(order):
                if cell in blocked:
                    continue
                if occupancy.get(cell, 0) >= self._occupancy_limit:
                    continue
                chosen = cell
                if rank > 0:
                    repaired[searcher] = 1.0
                break
            if chosen is None:
                # 예약이 도달가능한 모든 셀을 막았다. 조용히 규칙을 깨는 대신
                # 원래 선택을 쓰고 그 사실을 위반으로 센다.
                chosen = int(candidates[selected[searcher]])
                if occupancy.get(chosen, 0) >= self._occupancy_limit:
                    self._occupancy_violations += 1
                else:
                    self._separation_violations += 1
            destinations[searcher] = chosen
            occupancy[chosen] = occupancy.get(chosen, 0) + 1
            if self._reservations is not None:
                blocked.update(self._reservations[chosen])
            else:
                blocked.add(chosen)
        return destinations, repaired

    def _slice_hazard(self, time_index: int, destinations: np.ndarray) -> np.ndarray:
        """셀별 hazard 합. 같은 셀의 여러 기체는 **더한다**(보고서 4.2절)."""

        hazard = np.zeros(self.state_count, dtype=float)
        for searcher, model in enumerate(self.instance.problem.searchers):
            source = int(self.positions[searcher])
            destination = int(destinations[searcher])
            # 이동 중 소인도 지나간 셀에 쌓인다. ``swept_cells`` 는 그
            # 모형이 없으면 도착셀 하나만 돌려주므로 극좌표 격자도 그대로다.
            for cell, value in model.swept_cells(
                time_index, source, destination
            ).items():
                hazard[cell] += value
        return hazard

    def _worst_target_weights(self) -> np.ndarray:
        """누적 탐지가 뒤처진 표적에 몰아주는 softmin 가중."""

        logits = -self._detected / self.config.worst_target_temperature
        logits -= logits.max()
        weights = np.exp(logits)
        return weights / weights.sum()

    # ------------------------------------------------------------ observation

    def local_observations(self) -> np.ndarray:
        config = self.config
        width = config.action_count
        observations = np.zeros(
            (self.searcher_count, width * CANDIDATE_FEATURE_COUNT + SCALAR_FEATURE_COUNT),
            dtype=float,
        )
        time_index = min(self.step_count, self.time_count - 1)
        worst = int(np.argmin(self._detected))
        worst_mass = self._mass[worst]
        mean_mass = self._mass.mean(axis=0)
        scale = max(float(self._search_free[worst, time_index].max()), 1e-12)
        mean_scale = max(float(self._search_free.mean(axis=0)[time_index].max()), 1e-12)
        for searcher, model in enumerate(self.instance.problem.searchers):
            current = int(self.positions[searcher])
            candidates = self._candidates[current]
            degree = int(self._candidate_counts[current])
            neighbours = [
                int(self.positions[other])
                for other in range(self.searcher_count)
                if other != searcher
            ]
            for action in range(width):
                cell = int(candidates[action])
                base = action * CANDIDATE_FEATURE_COUNT
                observations[searcher, base + 0] = min(worst_mass[cell] / scale, 1.0)
                observations[searcher, base + 1] = min(mean_mass[cell] / mean_scale, 1.0)
                # SPX 와 **같은** hazard 를 봐야 비교가 성립한다. 이동 중
                # 소인을 지나간 셀에 나눠 주는 모형에서는 이 이동으로 쌓이는
                # hazard 의 총합이 곧 한 수의 가치다.
                swept = model.swept_cells(time_index, current, cell)
                observations[searcher, base + 2] = 1.0 - np.exp(
                    -sum(swept.values())
                )
                # 그 중 도착셀에 남는 몫. 이동이 길수록 낮아지므로 예전
                # transit_fraction 과 같은 역할을 한다.
                total = sum(swept.values())
                observations[searcher, base + 3] = (
                    float(swept.get(cell, 0.0) / total) if total > 0.0 else 0.0
                )
                crowd = self._reservations[cell] if self._reservations is not None else {cell}
                observations[searcher, base + 4] = (
                    sum(1 for other in neighbours if other in crowd)
                    / max(len(neighbours), 1)
                )
                observations[searcher, base + 5] = float(cell == current)
                observations[searcher, base + 6] = float(action >= degree)
            tail = width * CANDIDATE_FEATURE_COUNT
            observations[searcher, tail + 0] = time_index / max(self.time_count - 1, 1)
            observations[searcher, tail + 1] = float(worst_mass.sum())
            observations[searcher, tail + 2] = float(mean_mass.sum())
            observations[searcher, tail + 3] = min(worst_mass[current] / scale, 1.0)
            observations[searcher, tail + 4] = float(self.instance.scales[searcher])
        return observations

    def global_state(self) -> np.ndarray:
        time_index = min(self.step_count, self.time_count - 1)
        worst = int(np.argmin(self._detected))
        scale = max(float(self._search_free[worst, time_index].max()), 1e-12)
        mean_field = self._mass.mean(axis=0)
        mean_scale = max(float(mean_field.max()), 1e-12)
        pieces = [
            np.minimum(self._mass[worst] / scale, 1.0),
            mean_field / mean_scale,
            np.asarray(
                [
                    min(self._mass[worst][int(cell)] / scale, 1.0)
                    for cell in self.positions
                ],
                dtype=float,
            ),
            self.positions.astype(float) / max(self.state_count - 1, 1),
            self._detected.copy(),
            np.asarray([time_index / max(self.time_count - 1, 1)], dtype=float),
        ]
        return np.concatenate(pieces)

    # ---------------------------------------------------------------- metrics

    def metrics(self) -> SPXSearchMetrics:
        completed = min(len(path) for path in self._paths) if self._paths else 0
        if completed == self.time_count:
            paths = tuple(tuple(path) for path in self._paths)
            target_pds = self.instance.score_paths(paths)
        else:
            paths = tuple(tuple(path) for path in self._paths)
            target_pds = tuple(float(value) for value in self._detected)
        return SPXSearchMetrics(
            paths=paths,
            target_detection_probabilities=tuple(float(v) for v in target_pds),
            worst_target_detection_probability=(
                min(target_pds) if target_pds else 0.0
            ),
            repaired_moves=self._repaired_moves,
            occupancy_violations=self._occupancy_violations,
            separation_violations=self._separation_violations,
            completed_slices=completed,
        )

    def online_target_detection_probabilities(self) -> tuple[float, ...]:
        """환경이 온라인으로 누적한 표적별 탐지확률.

        ``instance.score_paths`` 와 같아야 한다 — 다르면 보상이 채점과 다른
        것을 재고 있다는 뜻이다. ``tests/test_spx_env.py`` 가 검사한다.
        """

        return tuple(float(value) for value in self._detected)
