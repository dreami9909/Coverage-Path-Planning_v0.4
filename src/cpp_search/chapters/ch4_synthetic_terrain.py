"""Chapter 4 — 합성지형 대입 결과.

**묻는 것**: Chapter 2·3 이 잰 것은 계획모형 안에서의 탐지확률(model PD)이다.
그 계획을 **실제로 비행시켜** IMM5 진리 표적 앙상블에 대고 재면 무엇이 남는가?

model PD 와 flown PD 는 왜 다른가
---------------------------------
계획모형은 격자·시간슬라이스·Markov 전이로 이산화된 세계다. 실제 평가는 다르다.

* 진리 표적은 격자 셀이 아니라 연속 좌표 위를 IMM5 로 움직인다.
* 탐지는 셀 방문 hazard 가 아니라 **SR-Z50 EO/IR 공간 탐지모델**(LOS·지형
  차폐·표적 시그니처)이 판정한다.
* 계획모형은 셀 방문 한 번에 hazard 하나를 센다. 실제 비행은 그 셀 안에서
  평행소인으로 **연속으로** 훑는다.
* 경로는 시간 예산에 맞춰 잘리고, 이동 구간은 센서가 꺼진다.
* 계획이 쓴 지형과 진리가 쓴 지형은 ``map_error`` 가 켜지면 **독립 추출**이다.

이 요인들이 서로 반대 방향으로 작용하므로 **차이의 부호는 미리 알 수 없다.**
그래서 이 챕터는 부호를 가정하지 않고 ``model_minus_flown`` 을 그대로
보고한다 — 그 값 자체가 "계획모형이 실제를 어느 쪽으로 얼마나 잘못
추정하는가"의 측정치다.

측정된 부호는 **음수**다 (축소 4x4 격자, 합성지형: 약 −0.03 ~ −0.05).
즉 계획모형이 실제보다 **낮게** 잡는다. 셀 방문 하나를 hazard 하나로 세는
이산화가 셀 안의 연속 소인이 실제로 덮는 면적을 과소평가하기 때문이다.
이것은 계획을 보수적으로 만드는 방향이므로, 계획모형의 PD 를 성능 주장으로
쓸 때 과대주장이 되지는 않는다 — 대신 **계획모형이 고른 경로가 최적이라는
보장이 실비행 척도에서는 약해진다**. Chapter 2 의 최적성 인증은 계획모형
안에서의 인증이고, 실비행 KPI 의 인증이 아니다.

무엇을 비교하는가
-----------------
같은 합성지형 seed 에서 SPX 와 MAPPO 를 각각 계획하고, **같은 경로변환**
(``planning/routes.local_sweep``)을 거쳐 **같은 진리 앙상블**에 대고 잰다.
경로기하 변환이 공통이라 KPI 차이가 소인 패턴 차이로 새지 않는다.

지형 다양성은 두 계열로 준다.

* ``synthetic`` — 회랑(기동로) · 장애물(수역) · 은폐패치(수목)를 가진 지형.
* ``control``   — 방향성 회랑이 없고 은폐패치만 있는 대조군. 집중도는 맞추되
                  이동 방향 정보를 주지 않는다. 지형의 **무엇이** 도움이
                  되는지를 가른다.

의존
----
* config: ``config/chapter4_synthetic_terrain.json`` + ``config/common_experiment.json``
* 위: ``planning/stone_spx``, ``learning/spx_policy``,
  ``chapters/ch2_stone_spx``(인스턴스 구축 공유), ``research/truth``,
  ``cpp_search.core.simulation``
* 아래: 없음. 이 챕터가 v0.4 의 최종 결과표다.
"""

from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np

from cpp_search.core.evaluation import evaluate_routes
from cpp_search.core.models import MissionConfig
from cpp_search.core.sensor_observation import SENSOR_PERFORMANCE_SR_Z50, SpatialDetectionModel
from cpp_search.core.simulation import EvaluationConfig, evaluate_monte_carlo
from cpp_search.core.terrain import build_control_terrain, build_synthetic_terrain
from cpp_search.config import ChapterConfig
from cpp_search.kpi import mission_kpi, summarise_kpi
from cpp_search.learning.spx_policy import mappo_path_solution, train_stone_mappo
from cpp_search.options import RunOptions
from cpp_search.planning.stone_spx import (
    build_stone_grid_instance,
    route_plan_from_solution,
    solve_stone_grid_paths,
)
from cpp_search.reliability import ReliabilityCriteria, paired_metric_report
from cpp_search.truth import cue_prior, generate_truth, target_profile, truth_provenance
from cpp_search.chapters.ch2_stone_spx import (
    launch_positions,
    resolved_grid,
    route_config,
    searcher_classes,
    target_specs,
)
from cpp_search.chapters.ch3_mappo_stone import environment_settings, training_settings


TERRAIN_BUILDERS = {
    "synthetic": build_synthetic_terrain,
    "control": build_control_terrain,
}

PLANNERS = ("stone-spx", "mappo-stone")


def _combine_strata(blocks: list[dict], weights: list[float]) -> dict:
    """표적 계층별 KPI 를 선언된 가중으로 합친다."""

    total = sum(weights)
    if total <= 0.0:
        raise ValueError("target stratum weights must have positive mass")
    normalised = [weight / total for weight in weights]
    combined: dict = {}
    numeric_keys = [
        key
        for key, value in blocks[0].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    for key in numeric_keys:
        combined[key] = sum(
            weight * float(block[key]) for weight, block in zip(normalised, blocks)
        )
    # 조건부 평균탐지시간은 "탐지된 표적만"의 평균이라 계층 가중이 아니라
    # **탐지 질량**(w_i × 탐지율_i)으로 가중해야 한다. 탐지가 0 인 계층은 그
    # 평균이 정의되지 않으므로(NaN) 빼고, 전 계층이 0 이면 NaN 을 남긴다 —
    # 0 으로 채우면 "즉시 탐지"로 읽힌다.
    detection_mass = [
        weight * float(block["detection_probability_within_limit"])
        for weight, block in zip(normalised, blocks)
    ]
    conditional = [
        float(block["conditional_mean_detection_time_s"]) for block in blocks
    ]
    usable = [
        (mass, value)
        for mass, value in zip(detection_mass, conditional)
        if mass > 0.0 and value == value
    ]
    combined["conditional_mean_detection_time_s"] = (
        sum(mass * value for mass, value in usable) / sum(mass for mass, _ in usable)
        if usable
        else float("nan")
    )
    combined["sample_count"] = sum(int(block["sample_count"]) for block in blocks)
    if "team" in blocks[0]:
        # 팀 블록은 경로 기하라서 표적 계층과 무관하다. 첫 블록이 곧 전부다.
        combined["team"] = dict(blocks[0]["team"])
    return combined


def run(config: ChapterConfig, options: RunOptions) -> dict:
    mission: MissionConfig = config.mission()
    sensor = config.sensor()
    criteria = ReliabilityCriteria.from_mapping(config.common_get("reliability"))
    settings = dict(config.get("stone_spx", {}) or {})
    if not settings:
        raise KeyError("config must declare the stone_spx block")

    seeds = tuple(int(value) for value in settings.get("planning_seeds", []))
    limit = settings.get("evaluation_seed_limit")
    if limit is not None:
        seeds = seeds[: int(limit)]
    if not seeds:
        raise ValueError("chapter 4 requires at least one evaluation seed")
    grid = dict(settings.get("evaluation_grid", {}) or {})
    if not grid:
        raise ValueError("chapter 4 requires stone_spx.evaluation_grid")
    families = tuple(
        str(name) for name in (settings.get("terrain_families") or ["synthetic"])
    )
    unknown = set(families) - set(TERRAIN_BUILDERS)
    if unknown:
        raise ValueError(f"unknown terrain families: {sorted(unknown)}")

    strata = tuple(
        target_profile(config, str(name))
        for name in (config.get("target_strata") or ["Tank-nominal"])
    )
    weight_map = dict(config.get("target_stratum_weights") or {})
    weights = [
        float(weight_map.get(profile.name, 1.0 / len(strata))) for profile in strata
    ]
    prior = cue_prior(config, mission.search_radius_m)[0]
    positions = launch_positions(
        mission,
        ring_ratio=float(config.get("stone_spx.initial_position_ring_ratio", 0.25)),
    )
    mappo_settings = training_settings(config, options)
    environment_config = environment_settings(config)
    plan_config = route_config(
        resolved_grid(mission, sensor, grid, options.mission_time_s),
        options=options,
        terrain_weighting=True,
    )

    family_blocks: list[dict] = []
    for family in families:
        build_terrain = TERRAIN_BUILDERS[family]
        records: dict[str, list[dict]] = {planner: [] for planner in PLANNERS}
        per_seed: list[dict] = []
        for seed in seeds:
            planner_terrain = build_terrain(mission.search_radius_m, seed=seed)
            # 진리 지형은 map_error 가 켜지면 독립 추출이다. 계획이 쓴 지도와
            # 세계가 같다고 가정하면 계획기 성능이 과대평가된다.
            truth_terrain = (
                build_terrain(mission.search_radius_m, seed=seed ^ 0x00A5_1E55)
                if options.map_error
                else planner_terrain
            )
            evaluation = EvaluationConfig(
                sample_count=options.sample_count,
                seed=seed ^ 0x4556_414C,
                detection_time_limit_s=options.mission_time_s,
            )
            truths = [
                generate_truth(
                    config,
                    mission,
                    replace(evaluation, seed=evaluation.seed ^ (index * 0x9E37)),
                    profile,
                    horizon_s=options.mission_time_s,
                    terrain=truth_terrain,
                    prior=prior,
                )
                for index, profile in enumerate(strata)
            ]

            instance = build_stone_grid_instance(
                mission,
                sensor,
                prior=prior,
                terrain=planner_terrain,
                targets=target_specs(config),
                searcher_classes=searcher_classes(config, mission),
                initial_positions=positions,
                planning_time_s=options.mission_time_s,
                initial_belief_delay_s=float(
                    config.get("stone_spx.initial_belief_delay_s", 0.0)
                ),
                seed=seed,
                config=plan_config,
            )
            spx = solve_stone_grid_paths(instance, method="stone-spx")
            learned = train_stone_mappo(
                instance,
                settings=mappo_settings,
                environment_config=environment_config,
                seed=seed ^ 0x4D41_5050,
            )
            solutions = {
                "stone-spx": spx,
                "mappo-stone": mappo_path_solution(instance, learned),
            }

            seed_row: dict = {"seed": seed, "planners": {}}
            fingerprints: set[str] = set()
            for planner in PLANNERS:
                solution = solutions[planner]
                plan = route_plan_from_solution(instance, solution, method=planner)
                diagnostics = plan.diagnostics
                fingerprints.add(diagnostics.common_input_fingerprint)
                geometry = evaluate_routes(
                    plan.routes, mission, sensor, time_limit_s=options.mission_time_s
                )
                blocks = []
                for index, (profile, trajectories) in enumerate(zip(strata, truths)):
                    performance = evaluate_monte_carlo(
                        plan.routes,
                        mission,
                        sensor,
                        trajectories,
                        evaluation,
                        detection_model=SpatialDetectionModel(
                            condition="sensor",
                            sensor_performance=SENSOR_PERFORMANCE_SR_Z50,
                            target_signature=profile.signature,
                        ),
                        sensor_detection_seed=seed ^ 0x5345_4E53 ^ (index * 0x9E37),
                    )
                    block = mission_kpi(
                        performance,
                        uav_count=mission.uav_count,
                        geometry=geometry,
                        extra_team={
                            "duplicate_assignment_ratio": (
                                diagnostics.duplicate_assignment_ratio
                            ),
                            "opposing_edge_swaps": diagnostics.opposing_edge_swaps,
                        },
                    )
                    block.update(seed=seed, target_profile=profile.name)
                    blocks.append(block)
                combined = _combine_strata(blocks, weights)
                model_worst = float(min(solution.target_detection_probabilities))
                combined.update(
                    seed=seed,
                    stratum_kpi={
                        block["target_profile"]: block for block in blocks
                    },
                    model_worst_target_detection_probability=model_worst,
                    model_minus_flown=(
                        model_worst
                        - float(combined["detection_probability_within_limit"])
                    ),
                )
                records[planner].append(combined)
                seed_row["planners"][planner] = {
                    "flown_detection_probability": combined[
                        "detection_probability_within_limit"
                    ],
                    "model_worst_target_detection_probability": model_worst,
                    "model_minus_flown": combined["model_minus_flown"],
                    "conditional_mean_detection_time_s": combined[
                        "conditional_mean_detection_time_s"
                    ],
                    "restricted_mean_detection_time_s": combined[
                        "restricted_mean_detection_time_s"
                    ],
                    "unique_area_coverage_ratio": combined[
                        "unique_area_coverage_ratio"
                    ],
                    "stratum_detection_probability": {
                        name: block["detection_probability_within_limit"]
                        for name, block in combined["stratum_kpi"].items()
                    },
                    "assignments": [list(path) for path in solution.paths],
                    "diagnostics": asdict(diagnostics),
                }
            # 두 계획법이 같은 인스턴스를 받았다는 것이 비교의 전제다.
            # 어긋나면 비교값을 내보내지 않고 실패한다.
            if len(fingerprints) != 1:
                raise AssertionError(
                    f"planners received different instances for seed {seed}"
                )
            seed_row["common_input_fingerprint"] = next(iter(fingerprints))
            per_seed.append(seed_row)

        family_blocks.append(
            {
                "family": family,
                "terrain_builder": build_terrain.__name__,
                "per_seed": per_seed,
                "kpi": {
                    planner: {
                        "combined": summarise_kpi(
                            records[planner],
                            criteria=criteria,
                            bootstrap_seed=options.seed + index * 0x9E37,
                        ),
                        "target_strata": {
                            profile.name: summarise_kpi(
                                [
                                    record["stratum_kpi"][profile.name]
                                    for record in records[planner]
                                ],
                                criteria=criteria,
                                bootstrap_seed=(
                                    options.seed + index * 0x9E37 + offset
                                ),
                            )
                            for offset, profile in enumerate(strata)
                        },
                        "model_minus_flown_mean": float(
                            np.mean(
                                [
                                    record["model_minus_flown"]
                                    for record in records[planner]
                                ]
                            )
                        ),
                    }
                    for index, planner in enumerate(PLANNERS)
                },
                "paired_spx_minus_mappo": {
                    metric: paired_metric_report(
                        records["stone-spx"],
                        records["mappo-stone"],
                        metric,
                        criteria=criteria,
                        bootstrap_seed=options.seed ^ 0x51A0_7C33,
                        left_label="Stone SPX flown routes",
                        right_label="MAPPO flown routes",
                    )
                    for metric in (
                        # 헤드라인: 비행 전체 탐지율과 탐지된 것들의 평균 탐지시간.
                        "detection_probability_within_limit",
                        "conditional_mean_detection_time_s",
                        # 진단: 면적탐색률, 절단 평균시간.
                        "unique_area_coverage_ratio",
                        "restricted_mean_detection_time_s",
                    )
                },
            }
        )

    return {
        "chapter": "4",
        "provenance": config.provenance(),
        "condition": {
            "mission": {
                "uav_count": mission.uav_count,
                "search_radius_m": mission.search_radius_m,
                "mission_time_s": options.mission_time_s,
            },
            "evaluation_seeds": list(seeds),
            "sample_count_per_seed": options.sample_count,
            "terrain_families": list(families),
            "map_error": options.map_error,
            "planners": list(PLANNERS),
            "shared_route_adapter": "planning/routes.local_sweep",
            "detection_model": (
                "SR-Z50 EO/IR spatial detection model with terrain occlusion"
            ),
            "headline_metrics": {
                "detection_rate": "detection_probability_within_limit — 비행 전체(mission_time_s)에서 탐지된 표적 비율. 별도 절단 없음",
                "mean_detection_time_s": "conditional_mean_detection_time_s — 탐지된 표적만의 평균 탐지시간",
                "note": "탐지율과 평균탐지시간은 짝으로 읽는다. 하나만 보면 '빨리 조금'과 '늦게 많이'를 구별 못 한다.",
            },
            "target_strata": [profile.name for profile in strata],
            "target_stratum_weights": {
                profile.name: weight for profile, weight in zip(strata, weights)
            },
            "truth": {
                profile.name: truth_provenance(config, profile)
                for profile in strata
            },
            "evaluation_grid": grid,
        },
        "terrain_families": family_blocks,
        "reliability_criteria": criteria.as_dict(),
    }
