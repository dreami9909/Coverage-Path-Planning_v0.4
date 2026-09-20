"""Chapter 2 의 측정들. **논문 수치는 전부 여기서 나온다.**

이 파일이 생긴 이유: 인증 수치를 임시 스크립트 18 개로 냈는데 그것들이
레포 밖(``/tmp``)에 있었다. 세션이 끝나면 재현할 방법이 없었고, 실제로
스크립트마다 입자 수와 완화 횟수가 조금씩 달라서 두 실행의 경계를 합쳤다가
무효가 된 적이 있다 (fingerprint 가 달랐다).

그래서 여기서는 **조건을 한 곳에서 만들고**(``_measure``), 모든 행에
``fingerprint`` 를 강제한다. fingerprint 가 같을 때만 서로 다른 실행의
상한·하한을 합쳐 말할 수 있다.

측정 넷:

* ``certificate_by_grid``   — 격자를 올리며 PD 와 gap
* ``budget_allocation``     — **같은 총예산**을 다르게 쪼갠 대조
* ``seed_replication``      — 인증 격자를 seed 여러 개로 (인증 주장의 근거)
* ``particle_sensitivity``  — 입자 수가 답을 흔드는가 (solve 없이)
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Iterable, Sequence

import numpy as np

from cpp_search.core.models import MissionConfig, SensorSpec
from cpp_search.config import ChapterConfig
from cpp_search.options import RunOptions
from cpp_search.planning.stone_spx import StoneGridInstance, solve_stone_grid_paths
from cpp_search.theory.stone_path import _nondetection_value_gradient, _target_path_hazard
from cpp_search.chapters.ch2_stone_spx import (
    build_instance,
    case_mission,
    case_specs,
    case_target_specs,
)

__all__ = [
    "budget_allocation",
    "measure_one",
    "certificate_by_grid",
    "particle_sensitivity",
    "seed_replication",
]

#: 셀당 입자. 입자 8 배 증가에도 PD 변화가 0.0007~0.0026 으로
#: 격자 한 단계 효과(0.05~0.08)의 1/30 이하였다 — 50 이면 충분하다.
PARTICLES_PER_CELL = 50

#: 우리가 쓰는 기본 예산. master 60 초는 이 규모에서 병목이었다 —
#: Tank 8x8 이 60 초에서 it8 이후 52 회를 헛돌았고, 300 초로 바꾸니
#: 9 회 553 초에 닫혔다 (갭도 닫히고 6.5 배 빨라짐).
DEFAULT_BUDGET = {"master_time_limit_s": 300.0, "max_iterations": 20}


def _grid_condition(
    config: ChapterConfig,
    *,
    width: int,
    particles: int,
    budget: dict,
    sparse: bool | str = "auto",
) -> dict:
    """선언된 정사각 격자 조건 위에 이번 측정의 값만 덮어쓴다."""

    base = dict(
        next(
            item
            for item in config.get("stone_spx.grid_conditions", [])
            if item.get("grid_kind") == "square"
        )
    )
    base.update(
        grid_width=width,
        grid_height=width,
        # 셀 한 변을 한 슬라이스에 지나가도록 T 를 유도한다. 셀 수와
        # 슬라이스 수를 따로 고르면 반드시 한쪽이 어긋난다.
        time_slice_count="auto",
        particle_count=particles,
        sparse_transitions=sparse,
        relative_tolerance=1e-2,
        mip_relative_gap=0.0,
        persistent_master=True,
        continuous_relaxation_iterations=3,
        local_improvement_passes=1,
        aggregate_identical_searchers=True,
        square_fit="circumscribed",
        **budget,
    )
    return base


def _row(
    *,
    case: str,
    width: int,
    seed: int,
    budget: dict,
    instance: StoneGridInstance,
    solution,
    runtime_s: float,
) -> dict:
    """측정 한 건의 기록. **fingerprint 없이는 만들지 않는다.**"""

    detection = solution.target_detection_probabilities
    statuses = solution.master_status_history
    limits = sum(1 for _, phase, status, _ in statuses if phase == "master" and status == "limit")
    warm = float(solution.warm_start_nondetection)
    return {
        "case": case,
        "grid": f"{width}x{width}",
        "cells": int(instance.cell_count),
        "cell_m": float(2.0 * instance.mission.search_radius_m / width),
        "time_slices": int(instance.time_slice_count),
        "particles": int(instance.config.particle_count),
        "seed": int(seed),
        "fingerprint": instance.fingerprint,
        "budget": dict(budget),
        "worst_target_detection_probability": float(min(detection)),
        "relative_optimality_gap": float(solution.relative_optimality_gap),
        "lower_bound_nondetection": float(solution.lower_bound_nondetection),
        "upper_bound_nondetection": float(solution.upper_bound_nondetection),
        "certified": bool(solution.converged),
        "iterations": int(solution.iterations),
        "runtime_s": float(runtime_s),
        # 계획이 master 에서 왔는가 휴리스틱에서 왔는가. "warm-start" 면
        # SPX 가 최적화했다고 말할 수 없다.
        "incumbent_source": solution.incumbent_source,
        "warm_start_nondetection": warm,
        "master_improvement": (
            float(warm - solution.upper_bound_nondetection)
            if np.isfinite(warm)
            else None
        ),
        # 시간이 병목인지 절단이 병목인지 — 예산을 어디에 넣을지의 근거.
        "master_time_limit_hits": limits,
        # 계획 자체. 이게 없어서 그림 한 장 그리려고 38 분짜리 재계산을 했다.
        # 결과 JSON 만으로 경로를 그릴 수 있어야 한다.
        "assignments": [list(path) for path in solution.paths],
        "start_cells": list(instance.starts),
        "master_status_history": [
            {"iteration": int(i), "phase": p, "status": s, "seconds": float(sec)}
            for i, p, s, sec in statuses
        ],
        "bound_history": [
            {
                "iteration": int(i),
                "lower_bound_nondetection": float(lo),
                "upper_bound_nondetection": float(up),
                "relative_optimality_gap": float(gap),
            }
            for i, lo, up, gap in solution.bound_history
        ],
    }


def _measure(
    config: ChapterConfig,
    mission: MissionConfig,
    sensor: SensorSpec,
    *,
    case: dict,
    width: int,
    seed: int,
    budget: dict,
    search_time_s: float,
    particles_per_cell: int = PARTICLES_PER_CELL,
    sparse: bool | str = "auto",
    terrain_weighting: bool = True,
    grid_overrides: dict | None = None,
) -> dict:
    """측정 한 건. 인스턴스 생성부터 인증까지 **한 곳에서** 조건을 만든다."""

    case_mission_config = case_mission(mission, case)
    targets = case_target_specs(config, case)
    particles = width * width * particles_per_cell
    options = RunOptions(
        sample_count=10,
        particle_count=particles,
        episode_count=1,
        target_profile="both",
        seed=1,
        map_error=False,
        communication_loss_probability=0.0,
        communication_latency_slices=0,
        mission_time_s=search_time_s,
    )
    grid = _grid_condition(
        config, width=width, particles=particles, budget=budget, sparse=sparse
    )
    if grid_overrides:
        grid.update(grid_overrides)
    started = perf_counter()
    instance = build_instance(
        config,
        options,
        case_mission_config,
        sensor,
        grid=grid,
        seed=seed,
        terrain_weighting=terrain_weighting,
        targets=targets,
    )
    build_s = perf_counter() - started
    solution = solve_stone_grid_paths(instance, method="stone-spx")
    row = _row(
        case=str(case["name"]),
        width=width,
        seed=seed,
        budget=budget,
        instance=instance,
        solution=solution,
        runtime_s=perf_counter() - started,
    )
    row["build_s"] = float(build_s)
    row["terrain_weighting"] = bool(terrain_weighting)
    return row


def measure_one(
    config: ChapterConfig,
    mission: MissionConfig,
    sensor: SensorSpec,
    **kwargs,
) -> dict:
    """측정 한 건의 공개 진입점. ``ch2_stone_spx.run`` 이 이걸 쓴다.

    챕터 실행과 연구 측정이 **같은 코드**를 지나가야 조건이 갈리지 않는다.
    임시 스크립트마다 입자 수가 달라 두 실행의 경계를 합쳤다가 무효가 된
    적이 있다 — 그 실수를 구조로 막는다.
    """

    return _measure(config, mission, sensor, **kwargs)


def _search_time_s(config: ChapterConfig) -> float:
    return float(config.common_get("mission.mission_time_s", 600.0))


def _cases(config: ChapterConfig, names: Iterable[str] | None) -> list[dict]:
    declared = {str(item["name"]): item for item in case_specs(config)}
    if names is None:
        return list(declared.values())
    return [declared[str(name)] for name in names]


def certificate_by_grid(
    config: ChapterConfig,
    *,
    widths: Sequence[int],
    seeds: Sequence[int],
    budget: dict | None = None,
    cases: Iterable[str] | None = None,
    on_row=None,
) -> dict[str, Any]:
    """격자를 올리며 PD 와 gap. 촘촘할수록 계획은 좋아지고 인증은 무너진다."""

    mission, sensor = config.mission(), config.sensor()
    budget = dict(budget or DEFAULT_BUDGET)
    search_time = _search_time_s(config)
    rows: list[dict] = []
    for case in _cases(config, cases):
        for width in widths:
            for seed in seeds:
                row = _measure(
                    config, mission, sensor, case=case, width=width,
                    seed=seed, budget=budget, search_time_s=search_time,
                )
                rows.append(row)
                if on_row is not None:
                    on_row(row)
    return {
        "study": "certificate-by-grid",
        "question": "격자 해상도가 PD 와 최적성 인증에 어떻게 작용하는가",
        "budget": budget,
        "search_time_s": search_time,
        "rows": rows,
    }


def budget_allocation(
    config: ChapterConfig,
    *,
    width: int,
    seeds: Sequence[int],
    total_budget_s: float,
    splits: Sequence[int],
    cases: Iterable[str] | None = None,
    on_row=None,
) -> dict[str, Any]:
    """**총예산을 고정하고** master 시간과 반복의 배분만 바꾼다.

    총예산까지 같이 바뀌면 "배분 때문"이라고 말할 수 없다. 앞선 측정이
    60 초x60 회(3,600 s)와 300 초x20 회(6,000 s)라 교란돼 있었다.
    """

    mission, sensor = config.mission(), config.sensor()
    search_time = _search_time_s(config)
    rows: list[dict] = []
    for case in _cases(config, cases):
        for iterations in splits:
            budget = {
                "master_time_limit_s": float(total_budget_s) / iterations,
                "max_iterations": int(iterations),
            }
            for seed in seeds:
                row = _measure(
                    config, mission, sensor, case=case, width=width,
                    seed=seed, budget=budget, search_time_s=search_time,
                )
                row["total_budget_s"] = float(total_budget_s)
                rows.append(row)
                if on_row is not None:
                    on_row(row)
    return {
        "study": "budget-allocation",
        "question": "같은 총예산을 어떻게 쪼개느냐가 인증을 가르는가",
        "grid": f"{width}x{width}",
        "total_budget_s": float(total_budget_s),
        "splits_iterations": list(splits),
        "rows": rows,
    }


def seed_replication(
    config: ChapterConfig,
    *,
    jobs: Sequence[tuple[str, int]],
    seeds: Sequence[int],
    budget: dict | None = None,
    on_row=None,
) -> dict[str, Any]:
    """인증 격자를 seed 여러 개로. **인증 주장의 근거는 이것뿐이다.**

    선언 기준이 "계획 seed 전부가 통과해야 통과"인데, seed 한 개로 낸
    0.97% 는 기준 1% 에서 0.03%p 거리였다 — 하나만 달라도 뒤집힌다.
    """

    mission, sensor = config.mission(), config.sensor()
    budget = dict(budget or DEFAULT_BUDGET)
    search_time = _search_time_s(config)
    declared = {str(item["name"]): item for item in case_specs(config)}
    rows: list[dict] = []
    for case_name, width in jobs:
        case = declared[str(case_name)]
        for seed in seeds:
            row = _measure(
                config, mission, sensor, case=case, width=width,
                seed=seed, budget=budget, search_time_s=search_time,
            )
            rows.append(row)
            if on_row is not None:
                on_row(row)
    gate = float(config.get("stone_spx.certificate.required_relative_gap", 0.01))
    per_job: dict[str, Any] = {}
    for case_name, width in jobs:
        key = f"{case_name}/{width}x{width}"
        matched = [
            row for row in rows
            if row["case"] == case_name and row["grid"] == f"{width}x{width}"
        ]
        gaps = [row["relative_optimality_gap"] for row in matched]
        detections = [row["worst_target_detection_probability"] for row in matched]
        per_job[key] = {
            "seed_count": len(matched),
            "worst_relative_gap": max(gaps) if gaps else None,
            "all_seeds_within_gate": bool(gaps) and max(gaps) <= gate,
            "detection_mean": float(np.mean(detections)) if detections else None,
            "detection_spread": float(max(detections) - min(detections)) if detections else None,
        }
    return {
        "study": "seed-replication",
        "question": "인증이 인스턴스를 바꿔도 재현되는가",
        "declared_gate": gate,
        "budget": budget,
        "summary": per_job,
        "rows": rows,
    }


def particle_sensitivity(
    config: ChapterConfig,
    *,
    width: int,
    particles_per_cell: Sequence[int],
    seed: int,
    cases: Iterable[str] | None = None,
    on_row=None,
) -> dict[str, Any]:
    """입자 수가 답을 흔드는가. **solve 없이** 같은 경로를 각 사슬로 평가한다.

    경로는 인접성만으로 정해지고 인접성은 입자와 무관하므로 재사용된다.
    격자를 올릴 때 PD 가 오르는 것이 추정잡음일 가능성을 배제하기 위한 검사.
    """

    mission, sensor = config.mission(), config.sensor()
    search_time = _search_time_s(config)
    rows: list[dict] = []
    for case in _cases(config, cases):
        case_mission_config = case_mission(mission, case)
        targets = case_target_specs(config, case)
        paths = None
        for ppc in particles_per_cell:
            particles = width * width * ppc
            options = RunOptions(
                sample_count=10, particle_count=particles, episode_count=1,
                target_profile="both", seed=1, map_error=False,
                communication_loss_probability=0.0, communication_latency_slices=0,
                mission_time_s=search_time,
            )
            grid = _grid_condition(
                config, width=width, particles=particles,
                budget=DEFAULT_BUDGET, sparse=False,
            )
            instance = build_instance(
                config, options, case_mission_config, sensor,
                grid=grid, seed=seed, terrain_weighting=True, targets=targets,
            )
            if paths is None:
                paths = solve_stone_grid_paths(instance, method="team-h1").paths
            values = []
            for target in instance.target_models:
                multiplier = np.full(
                    (
                        instance.problem.searcher_count,
                        instance.time_slice_count,
                        instance.problem.state_count,
                        instance.problem.state_count,
                    ),
                    float(target.hazard_multiplier),
                )
                hazard = _target_path_hazard(instance.problem, paths, multiplier)
                values.append(_nondetection_value_gradient(target, hazard)[0])
            row = {
                "case": str(case["name"]),
                "grid": f"{width}x{width}",
                "particles_per_cell": int(ppc),
                "particles": int(particles),
                "fingerprint": instance.fingerprint,
                "fixed_path_detection_probability": float(1.0 - max(values)),
            }
            rows.append(row)
            if on_row is not None:
                on_row(row)
    return {
        "study": "particle-sensitivity",
        "question": "격자를 올릴 때의 PD 상승이 입자 추정잡음인가",
        "method": "같은 경로를 서로 다른 입자 수의 Markov 사슬로 평가 (solve 없음)",
        "rows": rows,
    }
