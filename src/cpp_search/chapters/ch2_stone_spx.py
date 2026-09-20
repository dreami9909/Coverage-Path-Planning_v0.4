"""Chapter 2 — Stone(2016) 기반 다중에이전트 경로 최적화 + 지형 가중.

**묻는 것 두 개.**

1. 지형 가중을 계획에 넣으면 최적 경로와 그 성능이 실제로 달라지는가?
2. 그 최적해에 **최적성 인증**을 붙일 수 있는 격자는 어디까지인가?

무엇을 푸는가
-------------
Stone, Royset & Washburn (2016) 4장의 SPX — 이질 탐색자 · 다중 표적 ·
셀 점유한도 · 양립불가 이동을 갖는 경로제약 탐색 (식 4.48-4.56). 목적은
**최악 표적의 미탐지확률 최소화**(minimax)다. 표적 하나를 잘 잡고 다른 하나를
놓치는 계획이 평균으로는 좋아 보이는 것을 막는다.

비선형 볼록 미탐지함수는 Sect. 4.3.1 절단평면으로 푼다. 모든 master 는
MILP 이고, 접선 절단이 미탐지확률의 **유효 하한**을, master 가 내는 정수경로가
**상한**을 준다. 따라서 이 챕터가 보고하는 것은 "최적해"가 아니라
**최적해와 얼마나 떨어져 있는지 증명된 값**이다.

지형 가중은 **표적 belief 의 두 지점**에 들어간다 (``planning/terrain_belief`` 참조).

    prior         -> 격자 사전질량      (통행성·은폐 -> 표적이 어디 있을 법한가)
    transition    -> 입자필터 예측커널  (이동 편향·off-road·정지확률)

셀별 hazard 에는 들어가지 않는다. 합성지형의 ``observability_weight`` 는
선언된 모델 범위상 항상 1.0 이다 — 지형은 탐색자가 **무엇을 믿는가**를 바꾸고
탐색자가 **얼마나 잘 보는가**는 바꾸지 않는다. ``terrain_weighting=False`` 는
두 지점을 **전부** 끊는다. 하나만 끊으면 "지형을 뺐다"가 어느 주장인지 알 수
없다.

그래서 대조군(지형 가중 off)의 belief 는 cue ring 사전분포 그대로 — **회전
대칭**이다. 정수 master 가 그 대칭을 깨지 못해 대조군 쪽 인증이 열린다
(같은 격자에서 가중 on 은 1 초, off 는 한도까지 돈다). 이 어려움의 원인은
hazard 가 아니라 belief 의 대칭성이다.

인증 관문 (``certificate_gate``)
--------------------------------
사전에 선언한 상대 PND gap 기준을 **계획 seed 전부**가 통과해야 통과다.
통과하지 못한 격자의 계획을 "최적"이라고 부르지 않는다. v0.3 Ch6 고해상도
감사에서 240셀·5슬라이스·6기 조건은 13분 27초를 써도 gap 15.18% 였고
1% 기준을 실패했다 — 그 사실이 Chapter 3(MAPPO 해석)의 존재 이유다.

의존
----
* config: ``config/chapter2_stone_spx_terrain.json`` + ``config/common_experiment.json``
* 위: ``planning/stone_spx``, ``theory/stone_path``, ``cpp_search.core.terrain``
* 아래: Chapter 3 이 이 챕터가 만든 인스턴스와 같은 조건을 다시 쓴다.
"""

from __future__ import annotations

from math import cos, isfinite, sin, tau

import numpy as np

from dataclasses import replace

from cpp_search.core.models import MissionConfig, Point2D, SensorSpec
from cpp_search.core.motion import (
    CONTINUOUS_MOVE_PROFILE,
    MANEUVER_HEAVY_PROFILE,
    RELOCATION_HEAVY_PROFILE,
    STOP_HEAVY_PROFILE,
)
from cpp_search.core.terrain import build_synthetic_terrain
from cpp_search.config import ChapterConfig
from cpp_search.options import RunOptions
from cpp_search.planning.stone_spx import (
    StoneGridInstance,
    StoneSPXRouteConfig,
    StoneSPXSearcherClass,
    StoneSPXTargetSpec,
    build_stone_grid_instance,
    solve_stone_grid_paths,
)
from cpp_search.reliability import ReliabilityCriteria, paired_metric_report
from cpp_search.truth import cue_prior, target_profile


def searcher_classes(
    config: ChapterConfig, mission: MissionConfig
) -> tuple[StoneSPXSearcherClass, ...]:
    """이질 탐색자 등급. 선언이 없으면 동질 한 등급으로 떨어진다."""

    declared = list(config.get("stone_spx.searcher_classes", []) or [])
    if not declared:
        return (StoneSPXSearcherClass("common-sensor", mission.uav_count, 1.0),)
    classes = tuple(
        StoneSPXSearcherClass(
            str(item["name"]), int(item["count"]), float(item["hazard_scale"])
        )
        for item in declared
    )
    if sum(item.count for item in classes) != mission.uav_count:
        raise ValueError(
            "searcher_classes counts must sum to mission.uav_count "
            f"({mission.uav_count})"
        )
    return classes


def signature_hazard_multipliers(config: ChapterConfig) -> dict[str, float]:
    """표적별 hazard 배율 = W_target / W_geometric = 중심선 PD0.

    Chapter 1 의 소인폭 표(``sweep_width_table``)와 **같은 적분**에서 나온다.
    계획모형(SPX)과 실비행 평가가 같은 표적별 공간 탐지모델을 보게 하는 장치다.
    Stone 의 alpha_{l,c',c,t,k} 가 표적 k 에 의존하는 항이 정확히 이것이다.
    """

    from cpp_search.chapters.ch1_evaluation import sweep_width_table

    sensor = config.sensor()
    geometric = sensor.effective_sweep_width_m
    return {
        row["target_profile"]: float(row["sweep_width_m"]) / geometric
        for row in sweep_width_table(sensor, integration_steps=512)
        if row["channel_mode"] == "fused"
    }


#: 선언된 IMM5 모드비중 프로파일. 케이스 안의 "서로 다른 개체"는 이 중에서 고른다.
IMM5_BEHAVIORS = {
    profile.name: profile
    for profile in (
        STOP_HEAVY_PROFILE,
        RELOCATION_HEAVY_PROFILE,
        CONTINUOUS_MOVE_PROFILE,
        MANEUVER_HEAVY_PROFILE,
    )
}


def case_specs(config: ChapterConfig) -> tuple[dict, ...]:
    """케이스 목록. 케이스 하나가 minimax 하나다.

    케이스는 표적 **종류**(시그니처)를 고정하고, 그 표적이 보일 법한 IMM5
    거동 여러 개를 담는다. 계획은 그 중 **최악 거동**으로 평가된다 — "TEL 이
    나갔다는 건 알지만 어떻게 기동할지 모른다"가 정확히 이 구조다.

    표적 이동은 ``imm5_behavior`` 만 결정하고 시그니처는 탐지 난이도에만
    영향을 준다(Tank 0.874 vs TEL 0.867). 그래서 케이스마다 **다른 거동
    집합**을 줘야 두 케이스가 갈린다. 같은 집합을 주면 쌍둥이가 된다.
    """

    declared = list(config.get("stone_spx.cases", []) or [])
    if not declared:
        raise KeyError("config must declare stone_spx.cases")
    return tuple(declared)


def case_target_specs(
    config: ChapterConfig, case: dict
) -> tuple[StoneSPXTargetSpec, ...]:
    """한 케이스의 표적들. 시그니처는 공통, 거동만 다르다."""

    base = target_profile(config, str(case["signature_profile"]))
    from_signature = bool(config.get("stone_spx.target_hazard_from_signature", False))
    # 배율은 **시그니처**에서 나오므로 거동을 바꿔도 같은 값이다. 이름을
    # 바꾸기 **전에** 찾아야 한다.
    multiplier = (
        signature_hazard_multipliers(config)[base.name] if from_signature else 1.0
    )
    specs = []
    for item in case.get("targets", []):
        name = str(item["behavior"])
        if name not in IMM5_BEHAVIORS:
            raise ValueError(
                f"unknown IMM5 behaviour {name!r}; "
                f"declared: {sorted(IMM5_BEHAVIORS)}"
            )
        profile = replace(
            base,
            imm5_behavior=IMM5_BEHAVIORS[name],
            name=f"{base.name}/{name}",
        )
        specs.append(
            StoneSPXTargetSpec(profile, float(item.get("hazard_multiplier", multiplier)))
        )
    if not specs:
        raise ValueError(f"case {case.get('name')!r} declares no targets")
    return tuple(specs)


def case_mission(mission: MissionConfig, case: dict) -> MissionConfig:
    """케이스별 AOI. 그 케이스의 최악 거동이 1,080 s 에 퍼지는 범위다."""

    radius = case.get("search_radius_m")
    if radius is None:
        raise KeyError(f"case {case.get('name')!r} must declare search_radius_m")
    return replace(mission, search_radius_m=float(radius))


def target_specs(config: ChapterConfig) -> tuple[StoneSPXTargetSpec, ...]:
    declared = list(config.get("stone_spx.targets", []) or [])
    if not declared:
        raise KeyError("config must declare stone_spx.targets")
    from_signature = bool(config.get("stone_spx.target_hazard_from_signature", False))
    multipliers = signature_hazard_multipliers(config) if from_signature else {}
    specs = []
    for item in declared:
        profile = target_profile(config, str(item["profile"]))
        if from_signature:
            multiplier = multipliers[profile.name]
        else:
            multiplier = float(item.get("hazard_multiplier", 1.0))
        specs.append(StoneSPXTargetSpec(profile, multiplier))
    return tuple(specs)


def launch_positions(
    mission: MissionConfig, *, ring_ratio: float
) -> tuple[Point2D, ...]:
    """발사지점을 AOI 안쪽 고리에 등간격으로 둔다.

    출발점은 계획법이 고르는 것이 아니라 **주어지는 조건**이다. 두 계획법이
    같은 출발점을 받아야 비교가 성립하므로 여기서 한 번만 만든다.
    """

    if not 0.0 <= ring_ratio < 1.0:
        raise ValueError("initial_position_ring_ratio must be in [0, 1)")
    radius = ring_ratio * mission.search_radius_m
    return tuple(
        Point2D(
            mission.center.x + radius * cos(index * tau / mission.uav_count),
            mission.center.y + radius * sin(index * tau / mission.uav_count),
        )
        for index in range(mission.uav_count)
    )


def route_config(
    grid: dict,
    *,
    options: RunOptions,
    terrain_weighting: bool,
) -> StoneSPXRouteConfig:
    """선언된 격자 조건 하나를 ``StoneSPXRouteConfig`` 로 번역한다.

    지형 가중을 끈 대조군은 **같은 격자에서 훨씬 어려운 정수 문제**가 된다.
    셀별 hazard 가 관측성 가중을 잃고 균일해지면서 대칭해가 폭발하고, 정수
    master 가 그 대칭을 깨지 못한다 (16셀 격자에서도 수 분 안에 끝나지
    않았다). 그래서 대조군에만 적용할 한도를 config 가
    ``terrain_unweighted_overrides`` 로 따로 선언할 수 있다. 한도를 줄인
    결과는 인증이 열린 채로 나오고, **그 사실이 결과에 남는다** —
    ``terrain_weighting_effect.both_arms_certified`` 가 거짓이 된다.
    """

    if not terrain_weighting:
        grid = {**grid, **dict(grid.get("terrain_unweighted_overrides", {}) or {})}
    return StoneSPXRouteConfig(
        grid_width=int(grid.get("grid_width", 3)),
        grid_height=int(grid.get("grid_height", 3)),
        time_slice_count=int(grid.get("time_slice_count", 3)),
        particle_count=min(
            options.particle_count, int(grid.get("particle_count", options.particle_count))
        ),
        visit_hazard=float(grid.get("visit_hazard", 1.0)),
        hazard_calibration=str(
            grid.get("hazard_calibration", "effective-sweep-width")
        ),
        occupancy_limit=int(grid.get("cell_occupancy_limit", 1)),
        forbid_opposing_edge_swaps=bool(
            grid.get("forbid_opposing_edge_swaps", True)
        ),
        relative_tolerance=float(grid.get("relative_tolerance", 1e-6)),
        max_iterations=int(grid.get("max_iterations", 40)),
        mip_relative_gap=float(grid.get("mip_relative_gap", 0.0)),
        grid_kind=str(grid.get("grid_kind", "square")),
        radial_step_m=float(grid.get("radial_step_m", 600.0)),
        angular_bin_count=int(grid.get("angular_bin_count", 24)),
        reservation_separation_m=float(grid.get("reservation_separation_m", 0.0)),
        aggregate_identical_searchers=bool(
            grid.get("aggregate_identical_searchers", False)
        ),
        master_time_limit_s=(
            float(grid["master_time_limit_s"])
            if grid.get("master_time_limit_s") is not None
            else None
        ),
        persistent_master=bool(grid.get("persistent_master", False)),
        continuous_relaxation_iterations=int(
            grid.get("continuous_relaxation_iterations", 0)
        ),
        local_improvement_passes=int(grid.get("local_improvement_passes", 0)),
        master_backend=str(grid.get("master_backend", "highs")),
        terrain_weighting=terrain_weighting,
        square_fit=str(grid.get("square_fit", "circumscribed")),
        sparse_transitions=grid.get("sparse_transitions", "auto"),
    )


def auto_time_slice_count(
    mission: MissionConfig, sensor: SensorSpec, grid: dict, search_time_s: float
) -> int:
    """셀 한 변을 한 슬라이스에 지나가도록 T 를 정한다.

        중심선속도 x (H/T) = c   ->   T = H x vc / c

    이렇게 잡으면 도달거리가 셀의 1.14 배가 되어 "한 슬라이스에 한 칸"이
    성립하고, 날 수 없는 이동이 원리적으로 생기지 않는다 (기구학 상한은
    T <= H x v / c 이고 위 값은 항상 그 안쪽이다). 슬라이스 수를 셀 크기와
    따로 고르면 둘 중 하나가 반드시 어긋난다.
    """

    width = int(grid.get("grid_width", 1))
    cell_m = 2.0 * mission.search_radius_m / max(width, 1)
    centerline = sensor.centerline_search_speed_mps(mission.search_speed_mps)
    return max(1, round(search_time_s * centerline / cell_m))


def resolved_grid(
    mission: MissionConfig, sensor: SensorSpec, grid: dict, search_time_s: float
) -> dict:
    """``time_slice_count: "auto"`` 를 실제 수로 바꾼 격자 선언.

    T 는 셀 크기와 함께 정해져야 한다(``auto_time_slice_count``). 이 해소를
    ``build_instance`` 안에만 두면 ``route_config`` 를 직접 부르는 챕터는
    문자열 ``"auto"`` 를 그대로 ``int()`` 에 넘기고 죽는다 — Chapter 4 가
    실제로 그랬다. 해소는 한 곳에 두고 두 경로가 같이 쓴다.
    """

    if grid.get("time_slice_count") != "auto":
        return grid
    return {
        **grid,
        "time_slice_count": auto_time_slice_count(mission, sensor, grid, search_time_s),
    }


def build_instance(
    config: ChapterConfig,
    options: RunOptions,
    mission: MissionConfig,
    sensor: SensorSpec,
    *,
    grid: dict,
    seed: int,
    terrain_weighting: bool,
    targets: tuple[StoneSPXTargetSpec, ...] | None = None,
) -> StoneGridInstance:
    """Chapter 2 와 Chapter 3·4 가 **같은 함수로** 인스턴스를 만든다.

    이 함수를 챕터마다 복사하면 조건이 소리 없이 갈린다. 여기 한 곳만 둔다.
    """

    prior = cue_prior(config, mission.search_radius_m)[0]
    terrain = build_synthetic_terrain(mission.search_radius_m, seed=seed)
    return build_stone_grid_instance(
        mission,
        sensor,
        prior=prior,
        terrain=terrain,
        targets=target_specs(config) if targets is None else targets,
        searcher_classes=searcher_classes(config, mission),
        initial_positions=launch_positions(
            mission,
            ring_ratio=float(
                config.get("stone_spx.initial_position_ring_ratio", 0.25)
            ),
        ),
        planning_time_s=options.mission_time_s,
        initial_belief_delay_s=float(
            config.get("stone_spx.initial_belief_delay_s", 0.0)
        ),
        seed=seed,
        config=route_config(
            resolved_grid(mission, sensor, grid, options.mission_time_s),
            options=options,
            terrain_weighting=terrain_weighting,
        ),
    )


def _certificate_block(rows: list[dict], *, required_gap: float) -> dict:
    """이 격자에서 최적성을 어디까지 증명했는가."""

    gaps = [
        row["relative_optimality_gap"]
        for row in rows
        if isfinite(row["relative_optimality_gap"])
    ]
    passed = bool(gaps) and len(gaps) == len(rows) and max(gaps) <= required_gap
    return {
        "required_relative_gap": required_gap,
        "seed_count": len(rows),
        "seeds_with_finite_gap": len(gaps),
        "worst_relative_gap": max(gaps) if gaps else None,
        "median_relative_gap": float(np.median(gaps)) if gaps else None,
        "certified_seed_count": sum(
            1 for row in rows if row["certified"]
        ),
        "passed": passed,
        "reason": (
            "every planning seed proved optimality within the declared gap"
            if passed
            else "at least one planning seed left the certificate open; the "
            "plan is feasible but not proven optimal"
        ),
        "mean_runtime_s": float(np.mean([row["runtime_s"] for row in rows]))
        if rows
        else None,
    }


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
        raise ValueError("chapter 2 requires at least one planning seed")
    grids = list(settings.get("grid_conditions", []) or [])
    if not grids:
        raise ValueError("chapter 2 requires at least one grid condition")
    contrast = bool(settings.get("terrain_weighting_contrast", True))
    required_gap = float(
        settings.get("certificate", {}).get("required_relative_gap", 0.01)
    )

    def grid_blocks_for(
        case_mission_config: MissionConfig,
        case_targets: tuple[StoneSPXTargetSpec, ...],
    ) -> list[dict]:
      grid_blocks: list[dict] = []
      for grid in grids:
        name = str(grid.get("name", grid.get("grid_kind", "grid")))
        # 격자별로 대조군을 끌 수 있다. 운용 해상도에서 SPX 한 번이 수 분이라
        # 대조군까지 돌리면 실행시간이 두 배가 되는데, 그 격자에서 재려는 것은
        # 지형효과가 아니라 **인증 붕괴**다. 지형효과는 인증이 닫히는 축소
        # 격자에서 재야 의미가 있다 — 열린 인증 위의 두 값을 짝지어 빼면
        # 최적해가 아닌 것끼리의 차이를 재게 된다.
        grid_contrast = bool(grid.get("terrain_weighting_contrast", contrast))
        weighted_rows: list[dict] = []
        unweighted_rows: list[dict] = []
        for seed in seeds:
            for terrain_weighting in (True, False) if grid_contrast else (True,):
                # 연구 측정(ch2_studies)과 **같은 코드**를 지난다. 두 경로가
                # 따로 있으면 조건이 갈리고, 그러면 챕터 결과와 논문 수치가
                # 서로 다른 것을 재게 된다. 순환 import 를 피해 여기서 읽는다.
                from . import ch2_studies

                row = ch2_studies.measure_one(
                    config,
                    case_mission_config,
                    sensor,
                    case=case,
                    # 케이스마다 인증된 격자가 다르다 (TEL 10x10 / Tank 8x8).
                    # 공통 격자 하나로 돌리면 한쪽은 인증이 안 된다.
                    width=int(
                        case.get("certified_grid_width", grid.get("grid_width", 10))
                    ),
                    seed=seed,
                    budget={
                        "master_time_limit_s": float(
                            grid.get("master_time_limit_s", 300.0)
                        ),
                        "max_iterations": int(grid.get("max_iterations", 20)),
                    },
                    search_time_s=float(options.mission_time_s),
                    terrain_weighting=terrain_weighting,
                    grid_overrides={
                        key: value
                        for key, value in grid.items()
                        if key in {"sparse_transitions", "cell_occupancy_limit",
                                   "forbid_opposing_edge_swaps", "hazard_calibration"}
                    },
                )
                (weighted_rows if terrain_weighting else unweighted_rows).append(row)

        block: dict = {
            "name": name,
            "declared_condition": grid,
            "terrain_weighting_contrast": grid_contrast,
            "terrain_weighted": weighted_rows,
            "optimality_certificate": _certificate_block(
                weighted_rows, required_gap=required_gap
            ),
        }
        if unweighted_rows:
            block["terrain_unweighted"] = unweighted_rows
            control_certificate = _certificate_block(
                unweighted_rows, required_gap=required_gap
            )
            block["terrain_unweighted_optimality_certificate"] = control_certificate
            # 같은 seed 에서 지형 가중만 켰다 껐다 한 짝지은 차이.
            # 지형 인스턴스 자체가 seed 로 생성되므로 짝이 정확하다.
            effect = paired_metric_report(
                weighted_rows,
                unweighted_rows,
                "worst_target_detection_probability",
                criteria=criteria,
                bootstrap_seed=options.seed ^ 0x7E88_A123,
                left_label="terrain-weighted SPX",
                right_label="terrain-unweighted SPX",
            )
            both_certified = bool(
                block["optimality_certificate"]["passed"]
                and control_certificate["passed"]
            )
            effect["both_arms_certified"] = both_certified
            # 한쪽 인증이 열려 있으면 이 차이는 "두 최적해의 차이"가 아니다.
            # 그 사실을 숫자 옆에 문장으로 붙인다 — 붙이지 않으면 인증된
            # 값과 섞여 읽힌다.
            effect["reading"] = (
                "두 팔 모두 최적성이 증명됐다. 차이를 지형 가중의 효과로 읽을 수 있다."
                if both_certified
                else (
                    "한쪽 이상의 최적성 인증이 열려 있다. 지형 가중을 끄면 "
                    "셀 hazard 가 균일해져 대칭해가 폭발하고 정수 master 가 "
                    "그 대칭을 깨지 못한다. 따라서 이 차이는 두 최적해의 "
                    "차이가 아니라 **인증된 해와 미인증 해의 차이**이고, "
                    "지형 가중 효과의 하한으로만 읽어야 한다."
                )
            )
            block["terrain_weighting_effect"] = effect
        grid_blocks.append(block)
      return grid_blocks

    # 케이스 하나가 minimax 하나다. 케이스마다 자기 AOI 와 자기 거동집합을
    # 쓰므로 격자도 케이스마다 다른 물리 크기를 갖는다.
    case_blocks: list[dict] = []
    for case in case_specs(config):
        case_mission_config = case_mission(mission, case)
        case_targets = case_target_specs(config, case)
        blocks = grid_blocks_for(case_mission_config, case_targets)
        case_blocks.append(
            {
                "name": str(case["name"]),
                "declared_case": case,
                "search_radius_m": case_mission_config.search_radius_m,
                "targets": [
                    {
                        "profile": spec.profile.name,
                        "imm5_behavior": spec.profile.imm5_behavior.name,
                        "mode_probabilities": list(
                            spec.profile.imm5_behavior.mode_probabilities
                        ),
                        "hazard_multiplier": spec.hazard_multiplier,
                    }
                    for spec in case_targets
                ],
                "grids": blocks,
            }
        )

    grid_blocks = [
        {**block, "case": case["name"]}
        for case in case_blocks
        for block in case["grids"]
    ]
    gate_passed = all(
        block["optimality_certificate"]["passed"] for block in grid_blocks
    )
    certified = [
        f"{block['case']}/{block['name']}"
        for block in grid_blocks
        if block["optimality_certificate"]["passed"]
    ]
    open_certificates = [
        f"{block['case']}/{block['name']}"
        for block in grid_blocks
        if not block["optimality_certificate"]["passed"]
    ]
    return {
        "chapter": "2",
        "provenance": config.provenance(),
        "condition": {
            "mission": {
                "uav_count": mission.uav_count,
                "search_radius_m": mission.search_radius_m,
                "mission_time_s": options.mission_time_s,
            },
            "planning_seeds": list(seeds),
            "cases": [
                {
                    "name": block["name"],
                    "search_radius_m": block["search_radius_m"],
                    "targets": block["targets"],
                }
                for block in case_blocks
            ],
            "searcher_classes": [
                {
                    "name": item.name,
                    "count": item.count,
                    "hazard_scale": item.hazard_scale,
                }
                for item in searcher_classes(config, mission)
            ],
            # 배율은 **케이스별 표적**에서 나온다. 예전 평면 목록
            # (``stone_spx.targets``) 은 케이스 구조로 옮겨가면서 config 에서
            # 사라졌다.
            "target_hazard_multipliers": {
                spec.profile.name: spec.hazard_multiplier
                for block in case_specs(config)
                for spec in case_target_specs(config, block)
            },
            "target_hazard_from_signature": bool(
                config.get("stone_spx.target_hazard_from_signature", False)
            ),
            "objective": "minimise the worst target's non-detection probability",
            "solver": "Stone Sect. 4.3.1 cutting planes; every master is a MILP",
            "terrain": "synthetic terrain per planning seed",
            "terrain_weighting_contrast": contrast,
            "certificate_policy": {
                "required_relative_gap": required_gap,
                "all_planning_seeds_must_pass": True,
            },
        },
        # 케이스별 구조가 1차 결과다. grid_conditions 는 케이스를 펼친
        # 평면 목록으로, 인증 관문과 기존 소비자(그림·요약)를 위해 남긴다.
        "cases": case_blocks,
        "grid_conditions": grid_blocks,
        "certificate_gate": {
            "passed": gate_passed,
            "certified_grids": certified,
            "open_certificate_grids": open_certificates,
            "reason": (
                "every declared grid proved optimality within the gap"
                if gate_passed
                else "at least one grid could not be certified; Chapter 3 "
                "interprets those instances with a learned policy instead"
            ),
        },
    }
