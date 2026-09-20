"""Stone 인스턴스 위에서 MAPPO 를 학습시키고, 그 결과를 SPX 와 같은 자로 잰다.

이 모듈이 내보내는 주장은 **두 단계로 나뉜다**. 섞어서 말하면 안 된다.

1. ``learning_effect`` = 학습 정책 - 미학습 정책 (같은 평가 seed, 결정론적).
   이 구현으로 정당하게 주장할 수 있는 값이다. MAPPO 가 Stone 문제에서
   **무엇이든 배웠는가**에 답한다.

2. ``spx_gap`` = SPX incumbent - MAPPO 최선 (최악표적 PD 차이).
   0 이상이면 SPX 가 낫다는 뜻이고, 음수면 MAPPO 가 그 seed 에서 더 좋은
   실행가능해를 찾았다는 뜻이다. **어느 쪽이든 최적성 증명은 아니다** —
   SPX 의 인증폭(``relative_optimality_gap``)이 열려 있으면 두 값 모두 진짜
   최적해보다 낮을 수 있다.

정책 용량의 한계
----------------
``NumpyMAPPO`` 는 numpy 만으로 도는 구현이라 은닉층이 하나(tanh)다. 따라서
재현하는 것은 MAPPO 의 **학습 규칙**(GAE, 중앙화 비평자, PPO 클리핑)이고
심층 구현의 표현력이 아니다. ``policy_capacity`` 블록이 이 사실을 결과 JSON
에 같이 적는다 — 절대 성능값을 심층 MAPPO 논문값과 나란히 놓을 수 없다.

의존
----
* 위: ``learning/{spx_env, mappo}``, ``planning/stone_spx``.
* 아래: ``chapters/ch3_mappo_stone``, ``chapters/ch4_synthetic_terrain``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np

from cpp_search.planning.stone_spx import StoneGridInstance, StoneGridPathSolution
from cpp_search.learning.mappo import NumpyMAPPO, collect_mappo_rollout
from cpp_search.learning.spx_env import SPXEnvironmentConfig, StoneSPXEnvironment


@dataclass(frozen=True, slots=True)
class StoneMAPPOSettings:
    """학습 조건. 전부 config 에서 읽히고 코드 기본값은 smoke 용이다."""

    iterations: int = 40
    rollouts_per_iteration: int = 8
    hidden_size: int = 64
    actor_learning_rate: float = 5e-4
    critic_learning_rate: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coefficient: float = 0.01
    epochs: int = 4

    def __post_init__(self) -> None:
        if self.iterations <= 0 or self.rollouts_per_iteration <= 0:
            raise ValueError("MAPPO training needs at least one iteration and rollout")
        if self.hidden_size < 0:
            raise ValueError("hidden_size cannot be negative")


@dataclass(frozen=True, slots=True)
class StoneMAPPOResult:
    """학습 한 번의 전부. 챕터는 이걸 그대로 JSON 으로 찍는다."""

    best_paths: tuple[tuple[int, ...], ...]
    best_target_detection_probabilities: tuple[float, ...]
    best_worst_target_detection_probability: float
    untrained_worst_target_detection_probability: float
    learning_effect: float
    learning_curve: tuple[dict, ...]
    repaired_moves: int
    occupancy_violations: int
    separation_violations: int
    action_candidate_coverage: float
    runtime_s: float
    settings: dict
    policy_capacity: dict


def _evaluate(
    environment: StoneSPXEnvironment,
    policy: NumpyMAPPO,
    *,
    seed: int,
) -> tuple[tuple[tuple[int, ...], ...], tuple[float, ...], object]:
    """결정론적 평가 한 판. 같은 정책 + 같은 seed 면 같은 경로가 나온다."""

    _, metrics = collect_mappo_rollout(
        environment, policy, seed=seed, deterministic=True
    )
    return metrics.paths, metrics.target_detection_probabilities, metrics


def train_stone_mappo(
    instance: StoneGridInstance,
    *,
    settings: StoneMAPPOSettings | None = None,
    environment_config: SPXEnvironmentConfig | None = None,
    seed: int = 20_260_913,
) -> StoneMAPPOResult:
    """SPX 인스턴스 하나에 MAPPO 를 학습시키고 최선 실행가능 경로를 돌려준다.

    최선의 기준은 **SPX 와 같은 목적함수**다 — ``instance.score_paths`` 의
    최소값(최악 표적 PD). 학습 중 평가는 전부 결정론적이라, 곡선의 변화가
    오롯이 정책 변화다.
    """

    declared = settings or StoneMAPPOSettings()
    environment = StoneSPXEnvironment(instance, environment_config)
    policy = NumpyMAPPO(
        local_observation_shape=environment.local_observation_shape,
        global_state_size=environment.global_state_size,
        agent_count=environment.searcher_count,
        action_count=environment.action_count,
        hidden_size=declared.hidden_size,
        seed=seed,
    )
    started = perf_counter()
    evaluation_seed = seed ^ 0x4D41_5050

    untrained_paths, untrained_pds, _ = _evaluate(
        environment, policy, seed=evaluation_seed
    )
    untrained_worst = min(untrained_pds)
    best_paths = untrained_paths
    best_pds = untrained_pds
    best_worst = untrained_worst

    curve: list[dict] = []
    repaired = 0
    occupancy_violations = 0
    separation_violations = 0
    for iteration in range(declared.iterations):
        rollouts = []
        for index in range(declared.rollouts_per_iteration):
            rollout, metrics = collect_mappo_rollout(
                environment,
                policy,
                seed=seed + iteration * declared.rollouts_per_iteration + index,
            )
            rollouts.append(rollout)
            repaired += metrics.repaired_moves
            occupancy_violations += metrics.occupancy_violations
            separation_violations += metrics.separation_violations
        statistics = policy.update(
            rollouts,
            actor_learning_rate=declared.actor_learning_rate,
            critic_learning_rate=declared.critic_learning_rate,
            gamma=declared.gamma,
            gae_lambda=declared.gae_lambda,
            clip_ratio=declared.clip_ratio,
            entropy_coefficient=declared.entropy_coefficient,
            epochs=declared.epochs,
        )
        paths, target_pds, metrics = _evaluate(
            environment, policy, seed=evaluation_seed
        )
        worst = min(target_pds)
        if worst > best_worst:
            best_worst = worst
            best_paths = paths
            best_pds = target_pds
        curve.append(
            {
                "iteration": iteration + 1,
                "worst_target_detection_probability": float(worst),
                "target_detection_probabilities": [float(v) for v in target_pds],
                "best_worst_target_detection_probability": float(best_worst),
                "repaired_moves": int(metrics.repaired_moves),
                "actor_objective": float(statistics.actor_objective),
                "critic_loss": float(statistics.critic_loss),
                "approximate_kl": float(statistics.approximate_kl),
                "clip_fraction": float(statistics.clip_fraction),
            }
        )

    return StoneMAPPOResult(
        best_paths=best_paths,
        best_target_detection_probabilities=tuple(float(v) for v in best_pds),
        best_worst_target_detection_probability=float(best_worst),
        untrained_worst_target_detection_probability=float(untrained_worst),
        learning_effect=float(best_worst - untrained_worst),
        learning_curve=tuple(curve),
        repaired_moves=int(repaired),
        occupancy_violations=int(occupancy_violations),
        separation_violations=int(separation_violations),
        action_candidate_coverage=environment.action_candidate_coverage,
        runtime_s=perf_counter() - started,
        settings=asdict(declared),
        policy_capacity={
            "architecture": "linear-softmax with one tanh hidden layer",
            "hidden_size": declared.hidden_size,
            "reproduces": "MAPPO learning rule (GAE, centralized critic, PPO clipping)",
            "does_not_reproduce": "deep MAPPO network capacity from the source paper",
            "claim_limit": (
                "absolute performance is not comparable to a deep MAPPO "
                "implementation; only learning_effect and the same-instance "
                "SPX comparison are claimed"
            ),
        },
    )


def mappo_path_solution(
    instance: StoneGridInstance,
    result: StoneMAPPOResult,
) -> StoneGridPathSolution:
    """MAPPO 경로를 SPX 와 **같은 경로변환·같은 채점**에 넘길 수 있게 포장한다.

    하한은 ``nan`` 이다 — MAPPO 는 최적성 하한을 증명하지 않는다. 그 사실을
    숫자로 남겨야 결과 JSON 을 읽는 사람이 인증된 값과 섞지 않는다.
    """

    target_pds = instance.score_paths(result.best_paths)
    return StoneGridPathSolution(
        paths=result.best_paths,
        target_detection_probabilities=target_pds,
        lower_bound_nondetection=float("nan"),
        upper_bound_nondetection=1.0 - min(target_pds),
        relative_optimality_gap=float("nan"),
        iterations=len(result.learning_curve),
        converged=False,
        fallback_used=False,
        fallback_reason=None,
        runtime_s=result.runtime_s,
    )


def compare_with_spx(
    spx: StoneGridPathSolution,
    mappo: StoneGridPathSolution,
) -> dict:
    """두 계획법의 **같은 인스턴스** 비교. 최적성 주장은 하지 않는다."""

    spx_worst = min(spx.target_detection_probabilities)
    mappo_worst = min(mappo.target_detection_probabilities)
    return {
        "spx_worst_target_pd": float(spx_worst),
        "mappo_worst_target_pd": float(mappo_worst),
        "spx_minus_mappo": float(spx_worst - mappo_worst),
        "spx_relative_optimality_gap": (
            None
            if not np.isfinite(spx.relative_optimality_gap)
            else float(spx.relative_optimality_gap)
        ),
        "spx_certified": bool(spx.converged),
        "mappo_certified": False,
        "interpretation": (
            "같은 StoneGridInstance 위의 실행가능해 비교다. SPX 인증폭이 "
            "열려 있으면 두 값 모두 진짜 최적해보다 낮을 수 있다."
        ),
        "spx_runtime_s": float(spx.runtime_s),
        "mappo_runtime_s": float(mappo.runtime_s),
    }
