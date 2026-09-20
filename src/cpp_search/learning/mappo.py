"""MAPPO — 중앙집중 학습 / 분산 실행(CTDE) 다중에이전트 정책.

논문 도메인 격자 환경과, 의존성 없이 굴러가는 선형 근사 MAPPO 구현.

환경 (``MAPPOSearchEnvironment``)
--------------------------------
* 행동공간: 9방향 격자 이동 (``GRID_ACTIONS``, 정지 포함).
* 관측: 3채널 지역 패치 (표적확률 / 탐색 신선도 / 팀 위치).
* 센서 보고 -> 로그오즈 갱신 (``belief.py``와 같은 식)

      탐지  : l += ln(PD / PF)
      미탐지: l += ln((1-PD) / (1-PF))

* 통신 융합 (``_fuse_communicating_agents``): 통신 반경 안의 이웃 중
  **|로그오즈|가 가장 큰 값**을 채택(가장 확신 있는 관측을 신뢰).
  ``communication_loss_probability``로 링크를 베르누이 드롭한다.
* 보상 = 추적 보상 + 탐사 보상 - 근접충돌 벌점.

정책 (``NumpyMAPPO``)
---------------------
* actor: 지역 관측 -> softmax 행동확률 (파라미터 공유, 분산 실행)
* critic: 전역 상태 -> 에이전트별 가치 (중앙집중 학습)

* GAE (일반화 이점 추정)

      delta_t = r_t + gamma * V(s_{t+1}) * (1-done) - V(s_t)
      A_t     = delta_t + gamma * lambda * (1-done) * A_{t+1}
      R_t     = A_t + V(s_t)

* PPO clipped surrogate

      r_t(theta) = exp( ln pi_theta(a|o) - ln pi_old(a|o) )
      L = E[ min( r_t A_t, clip(r_t, 1-eps, 1+eps) A_t ) ] + c_H * H(pi)

  이점은 배치 평균·표준편차로 정규화한다. ``approximate_kl``과
  ``clip_fraction``이 갱신이 과했는지 알려주는 진단값이다.

* ``save`` / ``load`` — Ch5-b-2가 전이 도메인에서 **미리 학습한** 정책으로
  비교하기 위해 필요하다. 미학습 정책 비교는 알고리즘 차이가 아니라
  초기화 잡음을 재는 것이다.

참고: Su and Qian (2023).

의존
----
* 위: numpy만.
* 아래: ``teamwork/asoc``(환경 재사용), ``teamwork/transfer``,
  ``chapters/chapter5b1``, ``chapters/chapter5b2``.
"""

from __future__ import annotations

from dataclasses import dataclass
from json import dumps, loads
from math import exp, log
from pathlib import Path
from typing import Sequence

import numpy as np

from cpp_search.learning.nets import ConvNet


GRID_ACTIONS = np.asarray(
    [
        (0, 0),
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ],
    dtype=int,
)


@dataclass(frozen=True, slots=True)
class MAPPOSearchConfig:
    width: int = 20
    height: int = 20
    uav_count: int = 5
    target_count: int = 7
    horizon_steps: int = 200
    sensor_radius_cells: float = 3.0
    communication_radius_cells: float = 6.0
    local_observation_size: int = 7
    information_decay: float = 0.1
    detection_probability: float = 0.9
    false_alarm_probability: float = 0.1
    uav_speed_cells_per_step: float = 1.0
    target_speed_cells_per_step: float = 0.5
    tracking_reward_weight: float = 2.0
    exploration_reward_weight: float = 1.0
    collision_reward_weight: float = 0.5
    communication_loss_probability: float = 0.0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("grid dimensions must be positive")
        if self.uav_count <= 0 or self.target_count <= 0:
            raise ValueError("uav_count and target_count must be positive")
        if self.horizon_steps <= 0:
            raise ValueError("horizon_steps must be positive")
        if self.local_observation_size <= 0 or self.local_observation_size % 2 == 0:
            raise ValueError("local_observation_size must be positive and odd")
        if not 0.0 < self.detection_probability < 1.0:
            raise ValueError("detection_probability must be in (0, 1)")
        if not 0.0 < self.false_alarm_probability < 1.0:
            raise ValueError("false_alarm_probability must be in (0, 1)")
        if self.sensor_radius_cells <= 0.0 or self.communication_radius_cells <= 0.0:
            raise ValueError("sensor and communication radii must be positive")
        if not 0.0 <= self.information_decay < 1.0:
            raise ValueError("information_decay must be in [0, 1)")
        if not 0.0 <= self.communication_loss_probability <= 1.0:
            raise ValueError("communication_loss_probability must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class SearchTrackingMetrics:
    average_target_observation_rate: float
    average_exploration_rate: float
    unique_area_coverage_ratio: float
    collision_proximity_events: int


class MAPPOSearchEnvironment:
    """Su-Qian-style grid environment for the Chapter 5-b-1 reproduction."""

    def __init__(self, config: MAPPOSearchConfig | None = None) -> None:
        self.config = config or MAPPOSearchConfig()
        self.rng = np.random.default_rng(0)
        self.step_count = 0
        self.uav_positions = np.zeros((self.config.uav_count, 2), dtype=int)
        #: 격자 밖으로 나가려다 제자리가 된 행동 수. 0 이 아니면 정책이
        #: 실행 불가능한 행동을 고르고 있다는 뜻이라 진단으로 남긴다.
        self.boundary_rejected_moves = 0
        self.target_positions = np.zeros((self.config.target_count, 2), dtype=float)
        self.target_headings = np.zeros(self.config.target_count, dtype=float)
        shape = (self.config.uav_count, self.config.height, self.config.width)
        self.agent_log_odds = np.zeros(shape, dtype=float)
        self.agent_last_observed = np.zeros(shape, dtype=int)
        self.unique_covered = np.zeros((self.config.height, self.config.width), dtype=bool)
        self.target_observed_steps = np.zeros(self.config.target_count, dtype=int)
        self.collision_proximity_events = 0
        #: 반경 -> (dx, dy). 원판 모양은 반경에만 달렸으니 한 번만 만든다.
        self._footprint_offsets: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    @property
    def local_observation_shape(self) -> tuple[int, int, int]:
        size = self.config.local_observation_size
        return 3, size, size

    @property
    def global_state_size(self) -> int:
        return (
            2 * self.config.width * self.config.height
            + 2 * self.config.uav_count
        )

    def reset(self, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.agent_log_odds.fill(0.0)
        self.agent_last_observed.fill(0)
        self.unique_covered.fill(False)
        self.target_observed_steps.fill(0)
        self.collision_proximity_events = 0
        self.boundary_rejected_moves = 0

        cell_count = self.config.width * self.config.height
        starts = self.rng.choice(
            cell_count,
            size=self.config.uav_count,
            replace=self.config.uav_count > cell_count,
        )
        self.uav_positions[:, 0] = starts % self.config.width
        self.uav_positions[:, 1] = starts // self.config.width
        self.target_positions[:, 0] = self.rng.uniform(
            0.0,
            self.config.width - 1.0,
            self.config.target_count,
        )
        self.target_positions[:, 1] = self.rng.uniform(
            0.0,
            self.config.height - 1.0,
            self.config.target_count,
        )
        self.target_headings = self.rng.uniform(
            -np.pi,
            np.pi,
            self.config.target_count,
        )
        return self.local_observations(), self.global_state()

    def step(
        self,
        actions: Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, SearchTrackingMetrics]:
        selected = np.asarray(tuple(actions), dtype=int)
        if selected.shape != (self.config.uav_count,):
            raise ValueError("one action is required per UAV")
        if np.any((selected < 0) | (selected >= len(GRID_ACTIONS))):
            raise ValueError("action index is outside the nine-action space")

        self.step_count += 1
        self.agent_log_odds *= 1.0 - self.config.information_decay
        # 원문은 격자 밖으로 나가는 행동을 **후보에서 제외**한다. 좌표를
        # 잘라내면 "밖으로 가려는 행동"이 조용히 대기/경계이동으로 바뀌어,
        # 정책이 고른 행동과 실제로 일어난 행동이 달라진다 (검토서 F7).
        # 여기서는 그런 행동을 **제자리**로 처리하고 그 사실을 센다.
        movement = GRID_ACTIONS[selected]
        proposed = self.uav_positions + movement
        outside = (
            (proposed[:, 0] < 0)
            | (proposed[:, 0] > self.config.width - 1)
            | (proposed[:, 1] < 0)
            | (proposed[:, 1] > self.config.height - 1)
        )
        self.boundary_rejected_moves += int(outside.sum())
        self.uav_positions = np.where(outside[:, None], self.uav_positions, proposed)
        self._move_targets()

        rewards = np.zeros(self.config.uav_count, dtype=float)
        positive_increment = log(
            self.config.detection_probability / self.config.false_alarm_probability
        )
        negative_increment = log(
            (1.0 - self.config.detection_probability)
            / (1.0 - self.config.false_alarm_probability)
        )
        observed_by_team = np.zeros(self.config.target_count, dtype=bool)
        newly_observed_fraction = np.zeros(self.config.uav_count, dtype=float)

        target_cells = np.rint(self.target_positions).astype(int)
        # 표적 점유를 격자 하나로 만들어 둔다. 예전에는 센서 발자국의 셀마다
        # ``np.any(target_cells == (x, y))`` 를 돌려서 스텝당 130여 번,
        # 롤아웃 하나에 15만 번 넘게 불렀다 — 롤아웃 시간의 81% 가 env.step
        # 이었고 그 대부분이 이 루프였다 (신경망 forward 는 18%).
        occupancy = np.zeros((self.config.height, self.config.width), dtype=bool)
        inside_grid = (
            (target_cells[:, 0] >= 0)
            & (target_cells[:, 0] < self.config.width)
            & (target_cells[:, 1] >= 0)
            & (target_cells[:, 1] < self.config.height)
        )
        if inside_grid.any():
            visible = target_cells[inside_grid]
            occupancy[visible[:, 1], visible[:, 0]] = True

        for vehicle_id, position in enumerate(self.uav_positions):
            xs, ys = self._footprint(position, self.config.sensor_radius_cells)
            last_observed = self.agent_last_observed[vehicle_id]
            newly_observed_fraction[vehicle_id] = int(
                (last_observed[ys, xs] == 0).sum()
            ) / (self.config.width * self.config.height)

            # 난수 소비 순서를 보존한다. 예전 코드는 셀마다 정확히 한 번
            # ``rng.random()`` 을 불렀고, ``rng.random(n)`` 은 같은 비트
            # 스트림에서 n개를 순서대로 꺼내므로 결과가 비트 단위로 같다.
            draws = self.rng.random(xs.size)
            occupied = occupancy[ys, xs]
            report = draws < np.where(
                occupied,
                self.config.detection_probability,
                self.config.false_alarm_probability,
            )
            # 발자국의 셀은 서로 겹치지 않으므로 fancy index 누산이 안전하다.
            self.agent_log_odds[vehicle_id, ys, xs] += np.where(
                report, positive_increment, negative_increment
            )
            last_observed[ys, xs] = self.step_count
            self.unique_covered[ys, xs] = True

            distances = np.linalg.norm(self.target_positions - position, axis=1)
            in_range = distances <= self.config.sensor_radius_cells
            observed_draw = self.rng.random(self.config.target_count) < (
                self.config.detection_probability * in_range
            )
            observed_by_team |= observed_draw
            if np.any(in_range):
                nearest = float(np.min(distances[in_range]))
                rewards[vehicle_id] += self.config.tracking_reward_weight * (
                    1.0
                    + (self.config.sensor_radius_cells - nearest)
                    / self.config.sensor_radius_cells
                )
            rewards[vehicle_id] += (
                self.config.exploration_reward_weight
                * newly_observed_fraction[vehicle_id]
            )

        self.target_observed_steps += observed_by_team.astype(int)
        rewards += self._collision_penalties()
        self._fuse_communicating_agents()
        done = self.step_count >= self.config.horizon_steps
        metrics = self.metrics()
        return (
            self.local_observations(),
            self.global_state(),
            rewards,
            done,
            metrics,
        )

    def local_observations(self) -> np.ndarray:
        size = self.config.local_observation_size
        radius = size // 2
        observations = np.zeros(
            (self.config.uav_count, 3, size, size),
            dtype=float,
        )
        # 원문은 관측창 **밖**의 freshness 를 1 로 둔다 — 오래 안 본 곳이라는
        # 뜻이다. 0 으로 두면 "방금 봤다"로 읽혀 격자 경계 근처에서 정책이
        # 밖을 향하지 않게 된다 (검토서 F7).
        observations[:, 1, :, :] = 1.0
        height, width = self.config.height, self.config.width
        # 격자를 관측창 반경만큼 **덧대고 잘라낸다**. 예전에는 기체마다
        # 7x7 을 파이썬 이중 루프로 돌아 호출당 245 회 반복했고, 이 함수가
        # 롤아웃에서 가장 무거운 한 곳이었다. 덧대는 값이 곧 창 밖의 값이라
        # 경계 처리도 그대로 옮겨진다 — freshness 만 1, 나머지는 0.
        occupancy = _sigmoid(self.agent_log_odds)
        freshness = self.agent_last_observed / max(self.step_count, 1)
        # 동료 채널: 기체 수를 센 격자에서 자기 자신만 뺀다. 두 기체가 같은
        # 셀에 있으면 자기를 빼도 1 이 남아야 하므로 bool 이 아니라 개수다.
        occupant_count = np.zeros((height, width), dtype=int)
        np.add.at(
            occupant_count,
            (self.uav_positions[:, 1], self.uav_positions[:, 0]),
            1,
        )

        pad = radius
        padded = np.zeros((self.config.uav_count, 3, height + 2 * pad, width + 2 * pad))
        padded[:, 1, :, :] = 1.0
        padded[:, 0, pad : pad + height, pad : pad + width] = occupancy
        padded[:, 1, pad : pad + height, pad : pad + width] = freshness
        for vehicle_id in range(self.config.uav_count):
            own = np.zeros((height, width), dtype=int)
            own[
                self.uav_positions[vehicle_id, 1],
                self.uav_positions[vehicle_id, 0],
            ] = 1
            padded[vehicle_id, 2, pad : pad + height, pad : pad + width] = (
                occupant_count - own
            ) > 0

        for vehicle_id, (center_x, center_y) in enumerate(self.uav_positions):
            observations[vehicle_id] = padded[
                vehicle_id,
                :,
                center_y : center_y + size,
                center_x : center_x + size,
            ]
        return observations

    def global_state(self) -> np.ndarray:
        confidence_index = np.argmax(np.abs(self.agent_log_odds), axis=0)
        y_index, x_index = np.indices((self.config.height, self.config.width))
        fused_log_odds = self.agent_log_odds[confidence_index, y_index, x_index]
        occupancy = _sigmoid(fused_log_odds)
        freshness = np.max(self.agent_last_observed, axis=0) / max(self.step_count, 1)
        normalized_positions = self.uav_positions.astype(float).copy()
        normalized_positions[:, 0] /= max(self.config.width - 1, 1)
        normalized_positions[:, 1] /= max(self.config.height - 1, 1)
        return np.concatenate(
            [
                occupancy.ravel(),
                freshness.ravel(),
                normalized_positions.ravel(),
            ]
        )

    def metrics(self) -> SearchTrackingMetrics:
        denominator = max(self.step_count, 1)
        global_last_observed = np.max(self.agent_last_observed, axis=0)
        return SearchTrackingMetrics(
            average_target_observation_rate=float(
                np.mean(self.target_observed_steps / denominator)
            ),
            average_exploration_rate=float(
                np.mean(global_last_observed / denominator)
            ),
            unique_area_coverage_ratio=float(np.mean(self.unique_covered)),
            collision_proximity_events=self.collision_proximity_events,
        )

    def _footprint(
        self,
        position: np.ndarray,
        radius: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """반경 안 셀의 ``(xs, ys)``. 순서는 행 우선 — 난수 순서가 걸려 있다.

        원판 모양은 반경에만 달렸으므로 한 번 만들어 두고 중심만 옮긴다.
        격자 밖은 걸러내지만 남은 셀의 행 우선 순서는 유지되므로 예전
        ``_cells_in_radius`` 와 같은 순서다 — 그래서 결과가 바뀌지 않는다.
        """

        offsets = self._footprint_offsets.get(radius)
        if offsets is None:
            reach = int(np.ceil(radius))
            span = np.arange(-reach, reach + 1)
            delta_y, delta_x = np.meshgrid(span, span, indexing="ij")
            inside = delta_x**2 + delta_y**2 <= radius * radius
            offsets = (delta_x[inside], delta_y[inside])
            self._footprint_offsets[radius] = offsets
        delta_x, delta_y = offsets
        xs = delta_x + int(position[0])
        ys = delta_y + int(position[1])
        keep = (
            (xs >= 0) & (xs < self.config.width)
            & (ys >= 0) & (ys < self.config.height)
        )
        return xs[keep], ys[keep]

    def _cells_in_radius(
        self,
        position: np.ndarray,
        radius: float,
    ) -> list[tuple[int, int]]:
        center_x, center_y = int(position[0]), int(position[1])
        integer_radius = int(np.ceil(radius))
        return [
            (x, y)
            for y in range(
                max(0, center_y - integer_radius),
                min(self.config.height, center_y + integer_radius + 1),
            )
            for x in range(
                max(0, center_x - integer_radius),
                min(self.config.width, center_x + integer_radius + 1),
            )
            if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius * radius
        ]

    def _move_targets(self) -> None:
        self.target_headings += self.rng.normal(
            0.0,
            0.25,
            self.config.target_count,
        )
        speed = self.config.target_speed_cells_per_step
        self.target_positions[:, 0] += speed * np.cos(self.target_headings)
        self.target_positions[:, 1] += speed * np.sin(self.target_headings)
        for axis, limit in ((0, self.config.width - 1.0), (1, self.config.height - 1.0)):
            low = self.target_positions[:, axis] < 0.0
            high = self.target_positions[:, axis] > limit
            if np.any(low | high):
                self.target_positions[:, axis] = np.clip(
                    self.target_positions[:, axis],
                    0.0,
                    limit,
                )
                if axis == 0:
                    self.target_headings[low | high] = (
                        np.pi - self.target_headings[low | high]
                    )
                else:
                    self.target_headings[low | high] *= -1.0

    def _collision_penalties(self) -> np.ndarray:
        threshold = 2.0 * self.config.sensor_radius_cells
        # 기체 쌍 거리를 한 번에. 위삼각만 세므로 쌍을 두 번 세지 않는다.
        delta = (
            self.uav_positions[:, None, :].astype(float)
            - self.uav_positions[None, :, :]
        )
        distance = np.sqrt((delta**2).sum(axis=2))
        close = np.triu(distance <= threshold, k=1)
        self.collision_proximity_events += int(close.sum())
        penalty = np.where(
            close,
            -self.config.collision_reward_weight
            * np.exp((threshold - distance) / max(threshold, 1e-12)),
            0.0,
        )
        # 쌍 (l, r) 의 벌점은 두 기체 모두에게 더해진다.
        return penalty.sum(axis=0) + penalty.sum(axis=1)

    def _fuse_communicating_agents(self) -> None:
        before_log_odds = self.agent_log_odds.copy()
        before_last = self.agent_last_observed.copy()
        for vehicle_id in range(self.config.uav_count):
            distances = np.linalg.norm(
                self.uav_positions - self.uav_positions[vehicle_id],
                axis=1,
            )
            in_range = distances <= self.config.communication_radius_cells
            if self.config.communication_loss_probability > 0.0:
                delivered = (
                    self.rng.random(self.config.uav_count)
                    >= self.config.communication_loss_probability
                )
                delivered[vehicle_id] = True
                in_range &= delivered
            neighbors = np.flatnonzero(in_range)
            local_values = before_log_odds[neighbors]
            strongest = np.argmax(np.abs(local_values), axis=0)
            y_index, x_index = np.indices((self.config.height, self.config.width))
            self.agent_log_odds[vehicle_id] = local_values[
                strongest,
                y_index,
                x_index,
            ]
            self.agent_last_observed[vehicle_id] = np.max(
                before_last[neighbors],
                axis=0,
            )


@dataclass(frozen=True, slots=True)
class MAPPORollout:
    local_observations: np.ndarray
    global_states: np.ndarray
    actions: np.ndarray
    old_log_probabilities: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    final_global_state: np.ndarray


@dataclass(frozen=True, slots=True)
class MAPPOUpdateStats:
    actor_objective: float
    critic_loss: float
    approximate_kl: float
    clip_fraction: float
    sample_count: int


class NumpyMAPPO:
    """Shared decentralized actor and centralized per-agent critic.

    The linear function approximators keep the implementation dependency-free;
    the training update is MAPPO: GAE, centralized values, and PPO clipping.

    재현 범위의 한계 (Ch5-b-1 이 주장할 수 있는 것)
    ------------------------------------------------
    행위자와 비평자가 **은닉층 없는 선형 softmax**다. 따라서 이 구현이
    재현하는 것은 MAPPO의 **학습 규칙**(GAE, 중앙화 가치함수, PPO 클리핑)
    이지 원 논문의 **신경망 용량**이 아니다. 선형 정책은 관측의 비선형
    상호작용을 표현할 수 없으므로, 절대 성능값을 심층 MAPPO 구현과
    나란히 놓고 "재현했다"고 말할 수 없다.

    그래서 Ch5-b-1이 내보내는 주장은 두 단계로 나뉜다.

    * ``learning_effect`` (학습 정책 - 미학습 정책, held-out seed에서 측정)
      — 이 구현으로 정당하게 주장할 수 있는 값이다.
    * ``reproduction_error`` (digitize된 논문값과의 차이)
      — 정책 용량에 막혀 있다. 값이 커도 알고리즘 구현 오류라고 단정할 수
      없고, 작아도 논문 재현이라고 단정할 수 없다.

    ``chapters/chapter4b1.py``의 ``policy_capacity`` 블록이 이 한계를
    결과 JSON에 같이 적는다. 은닉층을 넣으려면 ``update``의 수동 그래디언트
    (연쇄법칙)도 같이 고쳐야 한다.
    """

    def __init__(
        self,
        local_observation_shape: Sequence[int],
        global_state_size: int,
        agent_count: int,
        action_count: int = 9,
        *,
        hidden_size: int = 0,
        architecture: str = "linear",
        cnn: dict | None = None,
        critic_grid_shape: Sequence[int] | None = None,
        seed: int = 20_260_830,
    ) -> None:
        self.local_observation_shape = tuple(int(v) for v in local_observation_shape)
        self.local_observation_size = int(np.prod(self.local_observation_shape))
        self.global_state_size = int(global_state_size)
        self.agent_count = int(agent_count)
        self.action_count = int(action_count)
        if min(
            self.local_observation_size,
            self.global_state_size,
            self.agent_count,
            self.action_count,
        ) <= 0:
            raise ValueError("MAPPO dimensions must be positive")
        self.rng = np.random.default_rng(seed)
        #: 은닉층 폭. 0 이면 예전의 선형 softmax 다.
        #:
        #: 원문(Su & Qian 2023)의 기본 MAPPO 에도 은닉층이 있다. 은닉층 없는
        #: 선형 정책은 관측 채널 사이의 비선형 상호작용 — 예를 들어 "표적
        #: 확률이 높으면서 동시에 오래 안 본 곳" — 을 표현하지 못한다. 세
        #: 채널을 각각 가중합할 뿐이라 곱 항이 없다. 그래서 절대 성능값을
        #: 심층 구현과 나란히 놓을 수 없었다 (검토서 F7).
        #:
        #: 활성함수는 tanh 다. 관측이 [0,1] 로 정규화돼 있어 대칭 활성이
        #: 안정적이고, 아래 수동 역전파에서 도함수가 ``1 - h^2`` 로 간단하다.
        self.hidden_size = int(hidden_size)
        if self.hidden_size < 0:
            raise ValueError("hidden_size must not be negative")
        if architecture not in {"linear", "cnn"}:
            raise ValueError("architecture must be 'linear' or 'cnn'")
        self.architecture = architecture
        self.cnn_settings = dict(cnn or {})
        self.critic_grid_shape = (
            tuple(int(v) for v in critic_grid_shape) if critic_grid_shape else None
        )
        if architecture == "cnn":
            self._build_convolutional(seed)
            return

        def scaled(fan_in: int, shape: tuple[int, ...]) -> np.ndarray:
            # 입력 차원이 커지면 초기 로짓이 커져 softmax 가 포화한다.
            # 1/sqrt(fan_in) 로 맞춘다.
            return self.rng.normal(0.0, 1.0 / np.sqrt(max(fan_in, 1)), shape)

        if self.hidden_size:
            self.actor_hidden_weight = scaled(
                self.local_observation_size,
                (self.local_observation_size, self.hidden_size),
            )
            self.actor_hidden_bias = np.zeros(self.hidden_size, dtype=float)
            self.critic_hidden_weight = scaled(
                self.global_state_size,
                (self.global_state_size, self.hidden_size),
            )
            self.critic_hidden_bias = np.zeros(self.hidden_size, dtype=float)
            actor_in = critic_in = self.hidden_size
        else:
            self.actor_hidden_weight = None
            self.actor_hidden_bias = None
            self.critic_hidden_weight = None
            self.critic_hidden_bias = None
            actor_in, critic_in = self.local_observation_size, self.global_state_size

        self.actor_weight = scaled(actor_in, (actor_in, self.action_count))
        self.actor_bias = np.zeros(self.action_count, dtype=float)
        self.critic_weight = scaled(critic_in, (critic_in, self.agent_count))
        self.critic_bias = np.zeros(self.agent_count, dtype=float)
        # Adam 상태. 논문(Su & Qian 2023, Table 3)이 선언한 최적화기가 Adam
        # 이고, 거기 적힌 lr = 5e-4 는 **Adam 기준 값**이다. 아래 _adam_step
        # 주석에 왜 이것이 재현의 전제인지 적어둔다.
        self._adam_state: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._adam_steps = 0
        self.actor_net = None
        self.critic_net = None

    def _build_convolutional(self, seed: int) -> None:
        """원문 Figure 2 의 기본(비재귀) 구조를 세운다.

        행위자 입력은 국소관측 격자 ``(3, L, L)`` 그대로다. 비평자 입력은
        전역상태 벡터인데, 그 앞부분이 ``(2, H, W)`` 격자(점유·신선도)이고
        뒤가 기체 위치 벡터다. 논문은 비평자도 CNN 으로 시작한다고 적었으므로
        격자 부분만 합성곱에 넣고 위치는 평탄화 뒤에 이어 붙인다.
        """

        settings = self.cnn_settings
        fully_connected = tuple(
            int(v) for v in settings.get("fully_connected", (64, 64))
        )
        self.actor_net = ConvNet(
            self.local_observation_shape,
            self.action_count,
            conv_channels=int(settings.get("actor_channels", 32)),
            kernel_size=int(settings.get("kernel_size", 3)),
            stride=int(settings.get("actor_stride", 1)),
            fully_connected=fully_connected,
            tail_size=0,
            rng=self.rng,
            prefix="actor",
        )
        if self.critic_grid_shape is None:
            raise ValueError(
                "the convolutional critic needs critic_grid_shape "
                "(channels, height, width) describing the leading part of "
                "the global state"
            )
        channels, height, width = self.critic_grid_shape
        grid_size = channels * height * width
        if grid_size > self.global_state_size:
            raise ValueError("critic_grid_shape does not fit inside the global state")
        self.critic_tail_size = self.global_state_size - grid_size
        self.critic_net = ConvNet(
            self.critic_grid_shape,
            self.agent_count,
            conv_channels=int(settings.get("critic_channels", 16)),
            kernel_size=int(settings.get("kernel_size", 3)),
            # 전역격자는 20x20 이라 stride 1 이면 특징이 10^4 개가 되고
            # 뒤따르는 FC 가중치가 수십만 개로 불어난다. stride 2 로 줄인다.
            stride=int(settings.get("critic_stride", 2)),
            fully_connected=fully_connected,
            tail_size=self.critic_tail_size,
            rng=self.rng,
            prefix="critic",
        )
        # 선형 경로의 속성들은 존재하되 비어 있다 — save/load 와 진단 코드가
        # 이 이름들을 참조한다.
        self.actor_hidden_weight = None
        self.actor_hidden_bias = None
        self.critic_hidden_weight = None
        self.critic_hidden_bias = None
        self.actor_weight = None
        self.actor_bias = None
        self.critic_weight = None
        self.critic_bias = None
        self._adam_state = {}
        self._adam_steps = 0

    def _split_global_state(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(N, global_state_size)`` -> 격자 ``(N, C, H, W)`` 와 꼬리."""

        channels, height, width = self.critic_grid_shape
        grid_size = channels * height * width
        grid = states[:, :grid_size].reshape(-1, channels, height, width)
        return grid, states[:, grid_size:]

    def _adam_step(
        self,
        name: str,
        gradient: np.ndarray,
        *,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> np.ndarray:
        """Adam 상승 스텝. 반환값을 파라미터에 **더한다**.

        왜 SGD 로는 안 되는가
        ---------------------
        Adam 은 그래디언트를 그 RMS 로 나누므로 파라미터당 스텝 크기가
        그래디언트 크기와 **무관하게** 대략 ``lr`` 이 된다. 순수 SGD 의
        스텝은 ``lr * |g|`` 다. 이 환경에서 실측한 값으로 쓰면

            |W| RMS = 1.99e-2,  |g| RMS = 4.68e-3
            SGD (lr 5e-4)  스텝 2.34e-6  -> 가중치의 0.012%
            Adam(lr 5e-4)  스텝 5.0e-4   -> 가중치의 2.5%

        **214배 차이**다. SGD 로 논문의 lr 을 쓰면 정책이 거의 움직이지
        않고, 그러면 PPO 비율 ``r_t`` 가 ``1 +- 1e-5`` 를 벗어나지 못해
        신뢰영역이 **한 번도 작동하지 않는다** — 실제로 ``clip_fraction``
        이 500 회 갱신 내내 정확히 0.000 이었고 ``approximate_kl`` 이 lr 에
        정비례했다(순수 선형 영역이라는 증거).

        즉 ``clip_fraction == 0`` 은 "클리핑이 필요 없을 만큼 안정적"이 아니라
        "클리핑이 걸릴 만큼 움직이지 못한다"였다.
        """

        moment, velocity = self._adam_state.get(
            name, (np.zeros_like(gradient), np.zeros_like(gradient))
        )
        moment = beta1 * moment + (1.0 - beta1) * gradient
        velocity = beta2 * velocity + (1.0 - beta2) * gradient * gradient
        self._adam_state[name] = (moment, velocity)
        bias_correction1 = 1.0 - beta1**self._adam_steps
        bias_correction2 = 1.0 - beta2**self._adam_steps
        corrected_moment = moment / max(bias_correction1, 1e-12)
        corrected_velocity = velocity / max(bias_correction2, 1e-12)
        return learning_rate * corrected_moment / (np.sqrt(corrected_velocity) + epsilon)

    def _actor_forward(self, flat: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """``(logits, cache)``.

        선형/MLP 경로에서 두 번째 값은 은닉 활성이고, CNN 경로에서는 층별
        cache 리스트다. 어느 쪽이든 역전파에 필요한 것을 그대로 돌려준다.
        """

        if self.architecture == "cnn":
            # 앞쪽 축이 (샘플, 기체) 두 개일 수 있다. 합성곱은 4차원만 받으니
            # 한 번 합쳤다가 되돌린다.
            leading = flat.shape[:-1]
            grid = flat.reshape(-1, *self.local_observation_shape)
            logits, caches = self.actor_net.forward(grid)
            return logits.reshape(*leading, self.action_count), caches
        if self.actor_hidden_weight is None:
            return flat @ self.actor_weight + self.actor_bias, None
        hidden = np.tanh(flat @ self.actor_hidden_weight + self.actor_hidden_bias)
        return hidden @ self.actor_weight + self.actor_bias, hidden

    def _critic_forward(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        if self.architecture == "cnn":
            flat_states = np.atleast_2d(states)
            grid, tail = self._split_global_state(flat_states)
            values, caches = self.critic_net.forward(grid, tail)
            if np.ndim(states) == 1:
                return values[0], caches
            return values, caches
        if self.critic_hidden_weight is None:
            return states @ self.critic_weight + self.critic_bias, None
        hidden = np.tanh(states @ self.critic_hidden_weight + self.critic_hidden_bias)
        return hidden @ self.critic_weight + self.critic_bias, hidden

    def _apply_network(
        self, net, gradients: dict, learning_rate: float, optimizer: str
    ) -> None:
        """CNN 의 파라미터 전부를 한 번에 갱신한다.

        Adam 상태는 층 이름(``actor.conv.weight`` 등)으로 찾으므로 선형 경로와
        같은 ``_adam_step`` 을 그대로 쓴다.
        """

        for name, gradient in gradients.items():
            parameter = net.parameters[name]
            if optimizer == "adam":
                net.set(name, parameter + self._adam_step(
                    name, gradient, learning_rate=learning_rate
                ))
            else:
                net.set(name, parameter + learning_rate * gradient)

    def _apply(self, name: str, gradient, learning_rate: float, optimizer: str) -> None:
        """파라미터 하나를 갱신한다. Adam 상태를 이름으로 찾는다."""

        parameter = getattr(self, name)
        if optimizer == "adam":
            setattr(self, name, parameter + self._adam_step(
                name, gradient, learning_rate=learning_rate
            ))
        else:
            setattr(self, name, parameter + learning_rate * gradient)

    def action_probabilities(self, local_observations: np.ndarray) -> np.ndarray:
        local = np.asarray(local_observations, dtype=float)
        if local.shape[1:] != self.local_observation_shape:
            raise ValueError("local observations do not match actor input shape")
        flattened = local.reshape(local.shape[0], -1)
        return _softmax(self._actor_forward(flattened)[0])

    def act(
        self,
        local_observations: np.ndarray,
        global_state: np.ndarray,
        *,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        probabilities = self.action_probabilities(local_observations)
        if deterministic:
            actions = np.argmax(probabilities, axis=1)
        else:
            actions = np.asarray(
                [
                    self.rng.choice(self.action_count, p=probability)
                    for probability in probabilities
                ],
                dtype=int,
            )
        log_probabilities = np.log(
            probabilities[np.arange(self.agent_count), actions] + 1e-12
        )
        values = self.value(global_state)
        return actions, log_probabilities, values

    def value(self, global_state: np.ndarray) -> np.ndarray:
        state = np.asarray(global_state, dtype=float)
        if state.shape != (self.global_state_size,):
            raise ValueError("global_state does not match centralized critic input")
        return self._critic_forward(state)[0]

    def update(
        self,
        rollout: "MAPPORollout | Sequence[MAPPORollout]",
        *,
        actor_learning_rate: float = 0.01,
        critic_learning_rate: float = 0.01,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coefficient: float = 0.01,
        epochs: int = 4,
        optimizer: str = "adam",
    ) -> MAPPOUpdateStats:
        if actor_learning_rate <= 0.0 or critic_learning_rate <= 0.0:
            raise ValueError("learning rates must be positive")
        if optimizer not in {"adam", "sgd"}:
            raise ValueError("optimizer must be 'adam' or 'sgd'")
        if not 0.0 < gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
            raise ValueError("invalid return parameters")
        if not 0.0 < clip_ratio < 1.0 or epochs <= 0:
            raise ValueError("invalid PPO update parameters")

        # 배치 갱신 (Su & Qian Table 3 의 B=16): 여러 episode 를 한 번에 받는다.
        # 각 episode 는 자기 자신의 ``final_global_state`` 로 부트스트랩해야
        # 하므로 GAE 는 episode 별로 따로 계산한 뒤 이어붙인다. 이어붙인 뒤
        # 한 번에 훑으면 episode 경계에서 이득이 새어 나간다.
        episodes = [rollout] if isinstance(rollout, MAPPORollout) else list(rollout)
        if not episodes:
            raise ValueError("update requires at least one rollout")

        local_parts: list[np.ndarray] = []
        state_parts: list[np.ndarray] = []
        action_parts: list[np.ndarray] = []
        log_probability_parts: list[np.ndarray] = []
        advantage_parts: list[np.ndarray] = []
        return_parts: list[np.ndarray] = []
        old_value_parts: list[np.ndarray] = []

        for episode in episodes:
            local = np.asarray(episode.local_observations, dtype=float)
            states = np.asarray(episode.global_states, dtype=float)
            actions = np.asarray(episode.actions, dtype=int)
            old_log_probabilities = np.asarray(
                episode.old_log_probabilities,
                dtype=float,
            )
            rewards = np.asarray(episode.rewards, dtype=float)
            dones = np.asarray(episode.dones, dtype=float)
            episode_steps = local.shape[0]
            expected_agent_shape = (episode_steps, self.agent_count)
            if (
                actions.shape != expected_agent_shape
                or rewards.shape != expected_agent_shape
            ):
                raise ValueError("rollout action and reward arrays have invalid shape")
            if old_log_probabilities.shape != expected_agent_shape or dones.shape != (
                episode_steps,
            ):
                raise ValueError(
                    "rollout probability or done arrays have invalid shape"
                )

            episode_old_values = self._critic_forward(states)[0]
            final_value = self.value(episode.final_global_state)
            next_values = np.vstack([episode_old_values[1:], final_value[None, :]])
            nonterminal = (1.0 - dones)[:, None]
            # TD 오차 delta_t = r_t + gamma * V(s_{t+1}) * (1-done) - V(s_t)
            delta = rewards + gamma * nonterminal * next_values - episode_old_values
            episode_advantages = np.zeros_like(delta)
            running = np.zeros(self.agent_count, dtype=float)
            # GAE: A_t = delta_t + (gamma * lambda) * (1-done) * A_{t+1}
            # 뒤에서 앞으로 한 번만 훑으면 되는 지수가중 누적합이다.
            for time_index in range(episode_steps - 1, -1, -1):
                running = (
                    delta[time_index]
                    + gamma
                    * gae_lambda
                    * nonterminal[time_index]
                    * running
                )
                episode_advantages[time_index] = running

            local_parts.append(local)
            state_parts.append(states)
            action_parts.append(actions)
            log_probability_parts.append(old_log_probabilities)
            advantage_parts.append(episode_advantages)
            # 가치 목표값 R_t = A_t + V(s_t)
            return_parts.append(episode_advantages + episode_old_values)
            old_value_parts.append(episode_old_values)

        local = np.concatenate(local_parts, axis=0)
        states = np.concatenate(state_parts, axis=0)
        actions = np.concatenate(action_parts, axis=0)
        old_log_probabilities = np.concatenate(log_probability_parts, axis=0)
        advantages = np.concatenate(advantage_parts, axis=0)
        returns = np.concatenate(return_parts, axis=0)
        old_values = np.concatenate(old_value_parts, axis=0)
        time_count = local.shape[0]

        advantage_mean = float(advantages.mean())
        advantage_std = float(advantages.std())
        normalized_advantage = (advantages - advantage_mean) / max(
            advantage_std,
            1e-8,
        )

        flattened_local = local.reshape(time_count, self.agent_count, -1)
        sample_count = time_count * self.agent_count
        objective_values: list[float] = []
        critic_losses: list[float] = []
        kl_values: list[float] = []
        clipped_total = 0

        for _ in range(epochs):
            logits, actor_hidden = self._actor_forward(flattened_local)
            probabilities = _softmax(logits)
            selected_probability = probabilities[
                np.arange(time_count)[:, None],
                np.arange(self.agent_count)[None, :],
                actions,
            ]
            new_log_probability = np.log(selected_probability + 1e-12)
            ratio = np.exp(new_log_probability - old_log_probabilities)
            unclipped = ratio * normalized_advantage
            clipped_ratio = np.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
            clipped = clipped_ratio * normalized_advantage
            objective_values.append(float(np.mean(np.minimum(unclipped, clipped))))
            kl_values.append(float(np.mean(old_log_probabilities - new_log_probability)))
            is_clipped = (
                ((normalized_advantage >= 0.0) & (ratio > 1.0 + clip_ratio))
                | ((normalized_advantage < 0.0) & (ratio < 1.0 - clip_ratio))
            )
            clipped_total += int(is_clipped.sum())
            coefficient = np.where(
                is_clipped,
                0.0,
                normalized_advantage * ratio,
            ) / sample_count
            one_hot = np.eye(self.action_count)[actions]
            gradient_logits = coefficient[:, :, None] * (one_hot - probabilities)
            if entropy_coefficient > 0.0:
                entropy = -np.sum(
                    probabilities * np.log(probabilities + 1e-12),
                    axis=2,
                    keepdims=True,
                )
                gradient_logits += (
                    entropy_coefficient
                    * -probabilities
                    * (np.log(probabilities + 1e-12) + entropy)
                    / sample_count
                )
            self._adam_steps += 1
            if self.architecture == "cnn":
                # 층 순서를 따라 자동으로 내려간다 — 수동 연쇄법칙이 없다.
                self._apply_network(
                    self.actor_net,
                    self.actor_net.backward(
                        gradient_logits.reshape(-1, self.action_count), actor_hidden
                    ),
                    actor_learning_rate,
                    optimizer,
                )
                predicted_values, critic_cache = self._critic_forward(states)
                clipped_values = old_values + np.clip(
                    predicted_values - old_values, -clip_ratio, clip_ratio
                )
                plain_error = returns - predicted_values
                clipped_error = returns - clipped_values
                use_clipped = clipped_error**2 > plain_error**2
                # clip 분기가 이기는 것은 clip 이 **물렸을 때뿐**이다 (안 물리면
                # 두 오차가 같아 strict > 가 거짓). 물린 clip 의 도함수는 0 이므로
                # 그 표본의 그래디언트는 0 이다. 예전에는 clipped_error 를 그대로
                # 밀어 넣어 clip 을 넘어간 표본을 더 밀었다 — 유한차분 검증에서
                # 상대오차 5~14 배로 드러났다.
                value_error = np.where(use_clipped, 0.0, plain_error)
                critic_losses.append(
                    float(0.5 * np.mean(np.maximum(plain_error**2, clipped_error**2)))
                )
                self._apply_network(
                    self.critic_net,
                    self.critic_net.backward(value_error / time_count, critic_cache),
                    critic_learning_rate,
                    optimizer,
                )
                continue
            # 출력층은 은닉 활성(있으면)을 입력으로 받는다.
            actor_input = flattened_local if actor_hidden is None else actor_hidden
            # 은닉층 역전파는 **순전파에 쓰인** 출력 가중치를 써야 한다. 갱신
            # 뒤의 가중치를 쓰면 연쇄법칙이 다른 함수의 도함수가 된다. lr 이
            # 작으면 눈에 안 띄지만(1e-6 에서 상대오차 1e-7), 큰 스텝에서
            # 유한차분과 0.4 배 어긋났다. ``_apply`` 는 새 배열을 만들므로
            # 옛 참조를 잡아두면 된다.
            actor_output_weight = self.actor_weight
            self._apply(
                "actor_weight",
                np.einsum("tad,tak->dk", actor_input, gradient_logits),
                actor_learning_rate,
                optimizer,
            )
            self._apply(
                "actor_bias",
                gradient_logits.sum(axis=(0, 1)),
                actor_learning_rate,
                optimizer,
            )
            if actor_hidden is not None:
                # 연쇄법칙: dL/dh = g @ W2^T,  tanh 도함수 = 1 - h^2
                hidden_grad = (
                    gradient_logits @ actor_output_weight.T
                ) * (1.0 - actor_hidden**2)
                self._apply(
                    "actor_hidden_weight",
                    np.einsum("tad,tah->dh", flattened_local, hidden_grad),
                    actor_learning_rate,
                    optimizer,
                )
                self._apply(
                    "actor_hidden_bias",
                    hidden_grad.sum(axis=(0, 1)),
                    actor_learning_rate,
                    optimizer,
                )

            predicted_values, critic_hidden = self._critic_forward(states)
            # 원문 식 (32) 의 **clipped value loss**. actor 의 PPO clipping 과
            # 별개 항목이다 — 이전 구현은 일반 MSE 만 썼다 (검토서 F7).
            #
            #   L^V = max[ (V - R)^2 , (clip(V, V_old +- eps) - R)^2 ]
            #
            # 가치예측이 이전 값에서 크게 벗어나면 그 큰 쪽 오차를 쓰므로,
            # 한 번의 갱신으로 critic 이 튀는 것을 막는다.
            clipped_values = old_values + np.clip(
                predicted_values - old_values, -clip_ratio, clip_ratio
            )
            plain_error = returns - predicted_values
            clipped_error = returns - clipped_values
            use_clipped = clipped_error**2 > plain_error**2
            # clip 이 물린 표본의 도함수는 0 (위 CNN 경로의 주석 참조).
            value_error = np.where(use_clipped, 0.0, plain_error)
            critic_losses.append(
                float(0.5 * np.mean(np.maximum(plain_error**2, clipped_error**2)))
            )
            critic_input = states if critic_hidden is None else critic_hidden
            critic_output_weight = self.critic_weight  # 순전파 시점의 가중치
            self._apply(
                "critic_weight",
                critic_input.T @ value_error / time_count,
                critic_learning_rate,
                optimizer,
            )
            self._apply(
                "critic_bias", value_error.mean(axis=0), critic_learning_rate, optimizer
            )
            if critic_hidden is not None:
                critic_hidden_grad = (
                    value_error @ critic_output_weight.T
                ) * (1.0 - critic_hidden**2)
                self._apply(
                    "critic_hidden_weight",
                    states.T @ critic_hidden_grad / time_count,
                    critic_learning_rate,
                    optimizer,
                )
                self._apply(
                    "critic_hidden_bias",
                    critic_hidden_grad.mean(axis=0),
                    critic_learning_rate,
                    optimizer,
                )

        return MAPPOUpdateStats(
            actor_objective=float(np.mean(objective_values)),
            critic_loss=float(np.mean(critic_losses)),
            approximate_kl=float(np.mean(kl_values)),
            clip_fraction=clipped_total / (epochs * sample_count),
            sample_count=sample_count,
        )

    def save(self, path: Path) -> None:
        metadata = dumps(
            {
                "local_observation_shape": self.local_observation_shape,
                "global_state_size": self.global_state_size,
                "agent_count": self.agent_count,
                "action_count": self.action_count,
                "hidden_size": self.hidden_size,
                "architecture": self.architecture,
                "cnn": self.cnn_settings,
                "critic_grid_shape": self.critic_grid_shape,
                "algorithm": "MAPPO-CTDE-v3",
            }
        )
        if self.architecture == "cnn":
            arrays = {
                # npz 키에 '.' 을 쓰면 되돌릴 때 헷갈리므로 '__' 로 바꾼다.
                name.replace(".", "__"): value
                for net in (self.actor_net, self.critic_net)
                for name, value in net.parameters.items()
            }
            np.savez_compressed(path, metadata=np.asarray(metadata), **arrays)
            return
        arrays = {
            "actor_weight": self.actor_weight,
            "actor_bias": self.actor_bias,
            "critic_weight": self.critic_weight,
            "critic_bias": self.critic_bias,
        }
        if self.hidden_size:
            arrays.update(
                actor_hidden_weight=self.actor_hidden_weight,
                actor_hidden_bias=self.actor_hidden_bias,
                critic_hidden_weight=self.critic_hidden_weight,
                critic_hidden_bias=self.critic_hidden_bias,
            )
        np.savez_compressed(path, metadata=np.asarray(metadata), **arrays)

    @classmethod
    def load(cls, path: Path) -> "NumpyMAPPO":
        with np.load(path, allow_pickle=False) as data:
            metadata = loads(str(data["metadata"]))
            policy = cls(
                metadata["local_observation_shape"],
                metadata["global_state_size"],
                metadata["agent_count"],
                metadata["action_count"],
                # 예전 파일(v1)에는 이 키가 없다 — 그때는 선형이었다.
                hidden_size=int(metadata.get("hidden_size", 0)),
                architecture=str(metadata.get("architecture", "linear")),
                cnn=metadata.get("cnn"),
                critic_grid_shape=metadata.get("critic_grid_shape"),
            )
            if policy.architecture == "cnn":
                for net in (policy.actor_net, policy.critic_net):
                    for name in list(net.parameters):
                        net.set(name, data[name.replace(".", "__")].copy())
                return policy
            for name in (
                "actor_weight", "actor_bias", "critic_weight", "critic_bias",
                "actor_hidden_weight", "actor_hidden_bias",
                "critic_hidden_weight", "critic_hidden_bias",
            ):
                if name in data.files:
                    setattr(policy, name, data[name].copy())
        return policy


def collect_mappo_rollout(
    environment: MAPPOSearchEnvironment,
    policy: NumpyMAPPO,
    *,
    seed: int,
    horizon_steps: int | None = None,
    deterministic: bool = False,
) -> tuple[MAPPORollout, SearchTrackingMetrics]:
    """``deterministic=True`` 면 행동을 argmax 로 고른다.

    평가용이다. 같은 정책 + 같은 seed 면 **완전히 같은 궤적**이 나오므로,
    두 시점의 평가값 차이가 오롯이 정책 변화다. 표본추출로 행동을 고르면
    정책이 고정돼 있어도 값이 흔들리고, 그 흔들림은 seed 를 늘려도 비싸게만
    줄어든다 — 수렴 판정의 분해능이 거기서 막혔다.
    """

    local, global_state = environment.reset(seed)
    horizon = horizon_steps or environment.config.horizon_steps
    local_history: list[np.ndarray] = []
    global_history: list[np.ndarray] = []
    action_history: list[np.ndarray] = []
    probability_history: list[np.ndarray] = []
    reward_history: list[np.ndarray] = []
    done_history: list[bool] = []
    metrics = environment.metrics()

    for _ in range(horizon):
        actions, log_probabilities, _ = policy.act(
            local, global_state, deterministic=deterministic
        )
        local_history.append(local.copy())
        global_history.append(global_state.copy())
        action_history.append(actions.copy())
        probability_history.append(log_probabilities.copy())
        local, global_state, rewards, done, metrics = environment.step(actions)
        reward_history.append(rewards.copy())
        done_history.append(done)
        if done:
            break

    return (
        MAPPORollout(
            local_observations=np.asarray(local_history),
            global_states=np.asarray(global_history),
            actions=np.asarray(action_history),
            old_log_probabilities=np.asarray(probability_history),
            rewards=np.asarray(reward_history),
            dones=np.asarray(done_history),
            final_global_state=global_state.copy(),
        ),
        metrics,
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=-1, keepdims=True)
