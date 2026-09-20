"""Chapter 3 — MAPPO 로 Stone 을 해석한다.

**묻는 것**: Stone SPX 가 최적성을 증명하지 못하는 격자에서, 학습된 정책은
그 문제를 대신 풀 수 있는가? 그리고 증명이 되는 격자에서는 그 정책이 **증명된
최적해에 얼마나 붙는가**?

왜 이 챕터가 필요한가
---------------------
Chapter 2 가 보여주는 것은 두 가지다. 작은 격자에서 SPX 는 최적해와 인증을
같이 준다. 그런데 격자를 운용 해상도로 올리면 정수 master 의 전역 하한 증명이
무너지고, 남는 것은 "실행가능한 계획 하나"다 (v0.3 Ch6 감사: 240셀·5슬라이스·
6기, 13분 27초, gap 15.18%, 기준 1% 실패).

그 지점에서 선택지는 둘이다 — 인증을 포기하고 더 오래 돌리거나, **인증을
포기하는 대신 확장되는 근사해**를 쓰거나. 이 챕터는 후자를 잰다.

무엇이 같고 무엇이 다른가
-------------------------
같은 것: ``StoneGridInstance`` 전부. 격자, 시간 슬라이스, 지형결합 belief,
탐색자 hazard, 출발셀, 예약반경, 점유한도. 같다는 증거는 두 결과에 박히는
같은 ``common_input_fingerprint`` 다. 채점도 같다 —
``instance.score_paths`` 하나만 통과한다.

다른 것: 셀 경로를 고르는 방법. 절단평면이냐, 학습된 정책이냐.

MAPPO 는 SPX 의 실행가능집합을 **넘지 않는다**. 행동 후보를 인접 셀로
제한하고, 점유·분리 위반은 환경이 복구한다 (``learning/spx_env`` 참조).
행동 후보를 고정 K 개로 가지치기 때문에 오히려 **부분집합**이고, 그 대가는
``action_candidate_coverage`` 로 보고된다.

무엇을 주장하고 무엇을 주장하지 않는가
--------------------------------------
* 주장한다 — ``learning_effect``(학습 정책 - 미학습 정책, 같은 평가 seed,
  결정론적)와 ``achievement_ratio``(MAPPO 최악표적 PD / SPX 최악표적 PD).
* 주장하지 않는다 — MAPPO 의 최적성. 하한을 증명하지 않으므로 ``nan`` 을
  그대로 남긴다. SPX 인증폭이 열린 격자에서는 **두 값 모두** 진짜 최적해보다
  낮을 수 있다.
* 주장하지 않는다 — 심층 MAPPO 구현과의 절대 성능 비교. 이 구현은 은닉층
  하나(tanh)라 학습 **규칙**을 재현하고 신경망 **용량**을 재현하지 않는다.
  ``policy_capacity`` 블록이 그 사실을 결과에 같이 적는다.

의존
----
* config: ``config/chapter3_mappo_stone.json`` + ``config/common_experiment.json``
* 위: ``learning/{spx_policy, spx_env}``, ``planning/stone_spx``,
  ``chapters/ch2_stone_spx``(인스턴스 구축을 공유한다)
* 아래: Chapter 4 가 두 계획법의 경로를 실제로 비행시켜 KPI 를 잰다.
"""

from __future__ import annotations

from dataclasses import asdict
from math import isfinite

import numpy as np

from cpp_search.core.models import MissionConfig
from cpp_search.config import ChapterConfig
from cpp_search.learning.spx_env import SPXEnvironmentConfig
from cpp_search.learning.spx_policy import (
    StoneMAPPOSettings,
    compare_with_spx,
    mappo_path_solution,
    train_stone_mappo,
)
from cpp_search.options import RunOptions
from cpp_search.planning.stone_spx import solve_stone_grid_paths
from cpp_search.reliability import ReliabilityCriteria, paired_metric_report
from cpp_search.chapters.ch2_stone_spx import build_instance, searcher_classes, target_specs


def training_settings(config: ChapterConfig, options: RunOptions) -> StoneMAPPOSettings:
    block = dict(config.get("mappo", {}) or {})
    return StoneMAPPOSettings(
        # ``options.episode_count`` 는 러너가 이미 ``명령행 > config >
        # 기본값`` 으로 해석해 둔 값이다(``mappo.iterations`` 를 읽는다).
        # 여기서 ``block["iterations"]`` 를 다시 우선하면 ``--episodes`` 가
        # 조용히 무시된다.
        iterations=int(options.episode_count),
        rollouts_per_iteration=int(block.get("rollouts_per_iteration", 8)),
        hidden_size=int(block.get("hidden_size", 64)),
        actor_learning_rate=float(block.get("actor_learning_rate", 5e-4)),
        critic_learning_rate=float(block.get("critic_learning_rate", 1e-3)),
        gamma=float(block.get("gamma", 0.99)),
        gae_lambda=float(block.get("gae_lambda", 0.95)),
        clip_ratio=float(block.get("clip_ratio", 0.2)),
        entropy_coefficient=float(block.get("entropy_coefficient", 0.01)),
        epochs=int(block.get("epochs", 4)),
    )


def environment_settings(config: ChapterConfig) -> SPXEnvironmentConfig:
    block = dict(config.get("mappo.environment", {}) or {})
    return SPXEnvironmentConfig(
        action_count=int(block.get("action_count", 9)),
        worst_target_temperature=float(
            block.get("worst_target_temperature", 0.05)
        ),
        terminal_weight=float(block.get("terminal_weight", 10.0)),
        slice_weight=float(block.get("slice_weight", 20.0)),
        repair_penalty=float(block.get("repair_penalty", 0.05)),
    )


def _achievement_ratio(mappo_pd: float, spx_pd: float) -> float | None:
    """MAPPO 최악표적 PD 를 SPX 대비 비율로. SPX 가 0 이면 정의되지 않는다."""

    if spx_pd <= 0.0:
        return None
    return float(mappo_pd / spx_pd)


def run(config: ChapterConfig, options: RunOptions) -> dict:
    mission: MissionConfig = config.mission()
    sensor = config.sensor()
    criteria = ReliabilityCriteria.from_mapping(config.common_get("reliability"))
    settings = dict(config.get("stone_spx", {}) or {})
    if not settings:
        raise KeyError("config must declare the stone_spx block")

    seeds = tuple(int(value) for value in settings.get("planning_seeds", []))
    limit = settings.get("planning_seed_limit")
    if limit is not None:
        seeds = seeds[: int(limit)]
    if not seeds:
        raise ValueError("chapter 3 requires at least one planning seed")
    grids = list(settings.get("grid_conditions", []) or [])
    if not grids:
        raise ValueError("chapter 3 requires at least one grid condition")

    mappo_settings = training_settings(config, options)
    environment_config = environment_settings(config)
    keep_curve_for = set(
        str(name) for name in (config.get("mappo.keep_learning_curve_for") or [])
    )

    grid_blocks: list[dict] = []
    for grid in grids:
        name = str(grid.get("name", grid.get("grid_kind", "grid")))
        spx_rows: list[dict] = []
        mappo_rows: list[dict] = []
        comparisons: list[dict] = []
        for seed in seeds:
            instance = build_instance(
                config,
                options,
                mission,
                sensor,
                grid=grid,
                seed=seed,
                terrain_weighting=True,
            )
            spx = solve_stone_grid_paths(instance, method="stone-spx")
            learned = train_stone_mappo(
                instance,
                settings=mappo_settings,
                environment_config=environment_config,
                seed=seed ^ 0x4D41_5050,
            )
            mappo = mappo_path_solution(instance, learned)

            spx_worst = float(min(spx.target_detection_probabilities))
            mappo_worst = float(min(mappo.target_detection_probabilities))
            fingerprint = instance.fingerprint
            spx_rows.append(
                {
                    "seed": seed,
                    "common_input_fingerprint": fingerprint,
                    "worst_target_detection_probability": spx_worst,
                    "relative_optimality_gap": float(spx.relative_optimality_gap),
                    "certified_within_tolerance": bool(spx.converged),
                    "runtime_s": float(spx.runtime_s),
                }
            )
            row = {
                "seed": seed,
                "common_input_fingerprint": fingerprint,
                "worst_target_detection_probability": mappo_worst,
                "untrained_worst_target_detection_probability": (
                    learned.untrained_worst_target_detection_probability
                ),
                "learning_effect": learned.learning_effect,
                "achievement_ratio": _achievement_ratio(mappo_worst, spx_worst),
                "repaired_moves": learned.repaired_moves,
                "occupancy_violations": learned.occupancy_violations,
                "separation_violations": learned.separation_violations,
                "action_candidate_coverage": learned.action_candidate_coverage,
                "runtime_s": float(learned.runtime_s),
                "assignments": [list(path) for path in learned.best_paths],
            }
            # 학습곡선은 seed 당 수백 행이라 전부 담으면 결과 JSON 이 수십 MB 로
            # 커진다. 선언한 격자에서만 남긴다.
            if name in keep_curve_for:
                row["learning_curve"] = list(learned.learning_curve)
            mappo_rows.append(row)
            comparison = compare_with_spx(spx, mappo)
            comparison["seed"] = seed
            comparison["same_instance"] = True
            comparisons.append(comparison)

        ratios = [
            row["achievement_ratio"]
            for row in mappo_rows
            if row["achievement_ratio"] is not None
        ]
        certified_seeds = [
            row["seed"] for row in spx_rows if row["certified_within_tolerance"]
        ]
        gaps = [
            row["relative_optimality_gap"]
            for row in spx_rows
            if isfinite(row["relative_optimality_gap"])
        ]
        grid_blocks.append(
            {
                "name": name,
                "declared_condition": grid,
                "spx": spx_rows,
                "mappo": mappo_rows,
                "per_seed_comparison": comparisons,
                # 짝지은 차이. 부호 방향을 문자열로 같이 내보내므로 그림이
                # 규약을 추측하지 않는다.
                "paired_spx_minus_mappo": paired_metric_report(
                    spx_rows,
                    mappo_rows,
                    "worst_target_detection_probability",
                    criteria=criteria,
                    bootstrap_seed=options.seed ^ 0x3A99_0C41,
                    left_label="Stone SPX cutting plane",
                    right_label="MAPPO learned policy",
                ),
                "achievement": {
                    "metric": "MAPPO worst-target PD divided by SPX worst-target PD",
                    "mean": float(np.mean(ratios)) if ratios else None,
                    "min": float(np.min(ratios)) if ratios else None,
                    "max": float(np.max(ratios)) if ratios else None,
                    "seed_count": len(ratios),
                    "spx_certified_seeds": certified_seeds,
                    "spx_worst_relative_gap": max(gaps) if gaps else None,
                    "reading": (
                        "SPX 인증이 열린 격자에서는 이 비율이 1 을 넘어도 "
                        "MAPPO 가 최적이라는 뜻이 아니다 — 그 격자에서는 "
                        "SPX 값 자체가 최적해가 아니다."
                    ),
                },
                "learning_effect": {
                    "metric": "trained minus untrained worst-target PD",
                    "mean": float(
                        np.mean([row["learning_effect"] for row in mappo_rows])
                    ),
                    "min": float(
                        np.min([row["learning_effect"] for row in mappo_rows])
                    ),
                    "max": float(
                        np.max([row["learning_effect"] for row in mappo_rows])
                    ),
                },
                "runtime": {
                    "spx_mean_s": float(
                        np.mean([row["runtime_s"] for row in spx_rows])
                    ),
                    "mappo_mean_s": float(
                        np.mean([row["runtime_s"] for row in mappo_rows])
                    ),
                },
            }
        )

    return {
        "chapter": "3",
        "provenance": config.provenance(),
        "condition": {
            "planning_seeds": list(seeds),
            "targets": [
                {
                    "profile": spec.profile.name,
                    "hazard_multiplier": spec.hazard_multiplier,
                }
                for spec in target_specs(config)
            ],
            "searcher_classes": [
                {
                    "name": item.name,
                    "count": item.count,
                    "hazard_scale": item.hazard_scale,
                }
                for item in searcher_classes(config, mission)
            ],
            "terrain_weighting": True,
            "shared_objective": (
                "both planners minimise the worst target's non-detection "
                "probability on the same StoneGridInstance"
            ),
            "shared_scoring_function": "StoneGridInstance.score_paths",
            "mappo_settings": asdict(mappo_settings),
            "mappo_environment": {
                "action_count": environment_config.action_count,
                "worst_target_temperature": (
                    environment_config.worst_target_temperature
                ),
                "terminal_weight": environment_config.terminal_weight,
                "slice_weight": environment_config.slice_weight,
                "repair_penalty": environment_config.repair_penalty,
                "feasible_set": (
                    "subset of the SPX feasible set: adjacent-cell actions, "
                    "environment-side occupancy and separation repair, fixed-K "
                    "candidate pruning"
                ),
            },
        },
        "grid_conditions": grid_blocks,
        "policy_capacity": {
            "architecture": "shared linear-softmax actor with one tanh hidden layer",
            "reproduces": "MAPPO learning rule (GAE, centralized critic, PPO clipping)",
            "does_not_reproduce": "deep MAPPO network capacity",
            "claim_limit": (
                "absolute performance is not comparable to a deep MAPPO "
                "implementation; only learning_effect and the same-instance "
                "SPX comparison are claimed"
            ),
        },
    }
