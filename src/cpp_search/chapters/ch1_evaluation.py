"""Chapter 1 — 평가도 정의와 탐색기 스펙 정의.

**묻는 것**: 이후 두 계획법(Stone SPX, MAPPO)을 비교하기 전에 무엇을 동일하게
고정해야 하는가?

계획법을 비교하려면 비교되지 않는 것들이 먼저 못 박혀 있어야 한다. 이 챕터는
그 못을 박고, 박았다는 사실을 결과 JSON 에 남긴다. 선언하는 것은 여섯이다.

1. **소인폭 표 W** — ``W = ∫ P_D(x) dx``. 800 m 지지폭은 W = 1600 m 가
   **아니다**. 측방항은 LM weave + EO/IR 주사를 합친 raised-cosine 포락선이고,
   중심선항은 그 표적 시그니처의 EO/IR 탐지확률이라, 어려운 표적에서는 W 가
   기하 최대값이 아니라 줄어든다.
2. **탐색기 스펙** — SR-Z50 순간 주사폭, weave 진폭·파장, 트랙 간격, 계획
   중심선거리 대비 **실제 비행거리**.
3. **표적 프로파일** — Tank/TEL 시그니처와 IMM5 5모드 거동.
4. **P_D/P_F 증거 계약** — 센서 보고 하나를 로그오즈 증분으로 바꾸는 식.
5. **Monte Carlo 설계** — 표본수, 기준 seed, 신뢰수준, 추론 방식.
6. **KPI 정의** — 5개 순위지표와 진단지표의 문장 정의.

**자원 상한**(``coverage_feasibility``)이 이 챕터의 핵심 판정이다.

    A_max   = n_uav * v_search * T * W        (전이시간 0, 중복 0 가정)
    ceiling = A_max / (pi R^2)

이 값이 1 보다 훨씬 작으면 "면적을 넓게 훑는" 전략은 **애초에 이길 수 없다**.
확률질량을 좇는 전략이 유리해지는 구간이고, 그것이 Chapter 2 이후 경로최적화의
전제다. 어떤 계획기 결과든 이 상한에 대고 읽어야 한다.

의존
----
* config: ``config/chapter1_evaluation_contract.json`` + ``config/common_experiment.json``
* 수식: ``cpp_search.core.search_envelope``(W 적분),
  ``cpp_search.core.sensor_observation``(표적·채널별 중심선 PD),
  ``cpp_search.core.profiles``(Tank/TEL + IMM5)
* 이 챕터의 출력(W, P_D/P_F, KPI 정의)은 Chapter 2-4 의 공통 전제다.
"""

from __future__ import annotations

from math import exp, log

import numpy as np

from cpp_search.core.belief import LogOddsEvidenceGrid
from cpp_search.core.models import MissionConfig, Point2D, SensorSpec
from cpp_search.core.sensor_observation import (
    SENSOR_PERFORMANCE_SR_Z50,
    SpatialDetectionModel,
)
from cpp_search.aoi import containment_radius_m, halt_boost_sweep, step_sensitivity
from cpp_search.config import ChapterConfig
from cpp_search.options import RunOptions
from cpp_search.core.profiles import NOMINAL_TARGET_PROFILES


KPI_DEFINITIONS: dict[str, str] = {
    "unique_area_coverage_ratio": (
        "union of sensor-on footprints inside the AOI, divided by AOI area; "
        "repeated coverage is counted once"
    ),
    "probability_mass_coverage": (
        "initial target-location probability mass falling inside the unique "
        "searched support"
    ),
    "detection_probability_within_limit": (
        "detection rate: fraction of the Monte Carlo target ensemble detected at any time during the full flight (mission_time_s); there is no separate cut-off"
    ),
    "conditional_mean_detection_time_s": (
        "mean detection time over detected targets only; read together with the detection rate"
    ),
    "restricted_mean_detection_time_s": (
        "diagnostic: mean detection time with every miss counted as the flight end"
    ),
    "failure_rate_within_limit": (
        "exactly 1 - detection_probability_within_limit"
    ),
    "coverage_redundancy_ratio": (
        "fraction of per-LM covered cells that at least one other LM also "
        "covered"
    ),
    "mean_route_distance_m": "mean flown distance per LM, weave included",
    "max_route_distance_m": "maximum flown distance over the LM team",
    "team_total_distance_m": "sum of flown distance over the LM team",
    "centerline_vs_actual_woven_distance_m": (
        "planned centerline distance versus the distance actually flown once "
        "the lateral weave is applied"
    ),
}


def sweep_width_table(
    sensor: SensorSpec,
    *,
    integration_steps: int = 256,
) -> list[dict]:
    """W = integral P_D(x) dx for every target profile and channel mode.

    The lateral term is the raised-cosine system envelope (LM weave plus EO/IR
    scan).  The on-centerline term is the EO/IR detection probability for that
    target signature, so W shrinks for a harder target rather than staying at
    its geometric maximum.
    """

    envelope = sensor.search_envelope
    if envelope is None:
        raise ValueError("sweep-width table requires the composite envelope")
    observer = Point2D(0.0, 0.0)
    rows: list[dict] = []
    for profile in NOMINAL_TARGET_PROFILES:
        for channel_mode in ("eo", "ir", "fused"):
            model = SpatialDetectionModel(
                condition="sensor",
                sensor_performance=SENSOR_PERFORMANCE_SR_Z50,
                target_signature=profile.signature,
                channel_mode=channel_mode,
            )
            support = envelope.support_half_width_m
            step = 2.0 * support / integration_steps
            values = []
            for index in range(integration_steps + 1):
                offset = -support + index * step
                geometric = envelope.lateral_detection_probability(offset)
                if geometric <= 0.0:
                    values.append(0.0)
                    continue
                target = Point2D(0.0, offset)
                values.append(
                    geometric
                    * model.clear_sensor_probability(target, observer, sensor)
                )
            total = values[0] + values[-1]
            total += 4.0 * sum(values[1:-1:2])
            total += 2.0 * sum(values[2:-1:2])
            sweep_width = step * total / 3.0
            rows.append(
                {
                    "target_profile": profile.name,
                    "channel_mode": channel_mode,
                    "support_half_width_m": support,
                    "geometric_sweep_width_m": envelope.sweep_width_m(),
                    "sweep_width_m": sweep_width,
                    "centerline_detection_probability": model.clear_sensor_probability(
                        Point2D(0.0, 0.0),
                        observer,
                        sensor,
                    ),
                }
            )
    return rows


def evidence_contract(detection_probability: float, false_alarm: float) -> dict:
    """선언된 P_D / P_F 가 함의하는 로그오즈 증분.

    값을 여기서 다시 계산하지 않고 **구현을 직접 돌려서** 뽑는다
    (``cpp_search.core.belief.LogOddsEvidenceGrid``). 같은 식을 두 곳에 적으면
    계약과 구현이 조용히 갈라진다 — 이 챕터가 하는 일은 계약을 선언하는
    것이므로, 선언값이 구현에서 나오지 않으면 선언의 의미가 없다.
    """

    if not 0.0 < detection_probability < 1.0:
        raise ValueError("detection_probability must be in (0, 1)")
    if not 0.0 < false_alarm < 1.0:
        raise ValueError("false_alarm_probability must be in (0, 1)")

    increments: dict[str, float] = {}
    for name, detected in (("positive", True), ("negative", False)):
        # ``from_prior`` 는 입력을 정규화한다. 그래서 두 칸에 같은 질량을
        # 주어 로그오즈 0 에서 출발시킨다 (한 칸만 주면 정규화가 1.0 으로
        # 만들어 로그오즈가 클리핑 상한으로 튄다). 그 상태에서 첫 칸에만
        # 보고 하나를 적용하면 남은 로그오즈가 곧 그 보고의 증분이다.
        grid = LogOddsEvidenceGrid.from_prior(np.full(2, 0.5))
        observed = np.array([True, False])
        grid.update(
            observed,
            detected=detected,
            detection_probability=detection_probability,
            false_alarm_probability=false_alarm,
        )
        increments[name] = float(grid.log_odds[0] - grid.log_odds[1])

    return {
        "detection_probability": detection_probability,
        "false_alarm_probability": false_alarm,
        "positive_report_log_odds_increment": increments["positive"],
        "negative_report_log_odds_increment": increments["negative"],
        "source": "cpp_search.core.belief.LogOddsEvidenceGrid.update",
        "formula": "l += ln(PD/PF) on detection, ln((1-PD)/(1-PF)) otherwise",
    }



def _coverage_feasibility(
    mission: MissionConfig,
    sensor: SensorSpec,
    mission_time_s: float,
    sweep_rows: list[dict],
) -> dict:
    """AOI 대비 자원 상한. 계획기가 아무리 좋아도 넘을 수 없는 값.

    폭을 두 개 쓴다 — 서로 다른 질문에 답하기 때문이다.

    * **기하 발자국 폭** ``2 x coverage_half_width_m`` = 800 m.
      ``unique_area_coverage_ratio`` 가 세는 폭이 바로 이것이므로, 그 지표와
      비교할 상한은 반드시 이 폭으로 계산해야 한다. (이 함수의 첫 판은
      유효 탐색폭 W로 계산해서 상한을 실제의 절반으로 만들었고, 그 결과
      Ch6 baseline이 "상한을 넘는" 것처럼 보였다.)
    * **유효 탐색폭 W** = ∫PD(x)dx ≈ 319 m. 탐지확률로 환산할 때 쓰는 폭이다.
      발자국 안에 들어와도 가장자리는 탐지확률이 낮으므로 W가 더 작다.

    두 상한 사이의 간격이 "발자국에는 들어왔지만 탐지는 못 한" 여지다.
    """

    footprint_width_m = 2.0 * sensor.coverage_half_width_m
    # 실측 W (측방 탐지확률 곡선의 적분). sensor.effective_sweep_width_m 은
    # 기하 등가폭(400 m)이라 여기 쓰면 안 된다 — 그건 발자국 폭의 절반일 뿐
    # 탐지확률을 반영하지 않는다.
    measured = [
        row["sweep_width_m"]
        for row in sweep_rows
        if row.get("channel_mode") == "fused"
    ] or [row["sweep_width_m"] for row in sweep_rows]
    sweep_width_m = (
        sum(measured) / len(measured) if measured else sensor.effective_sweep_width_m
    )
    centerline_m = (
        mission.uav_count
        * sensor.centerline_search_speed_mps(mission.search_speed_mps)
        * mission_time_s
    )
    geometric_area_m2 = centerline_m * footprint_width_m
    effective_area_m2 = centerline_m * sweep_width_m
    geometric_ceiling = geometric_area_m2 / mission.total_area_m2
    return {
        "team_centerline_distance_m": centerline_m,
        "footprint_width_m": footprint_width_m,
        "sweep_width_m": sweep_width_m,
        "search_area_m2": mission.total_area_m2,
        # unique_area_coverage_ratio 와 직접 비교할 상한.
        "unique_area_coverage_ceiling": geometric_ceiling,
        # 탐지확률 관점의 상한 (Koopman 무작위탐색 1 - exp(-WL/A) 의 지수부).
        "effective_coverage_ceiling": effective_area_m2 / mission.total_area_m2,
        # 주의: 이건 **표적이 AOI 전체에 균등분포일 때**의 무작위탐색 탐지확률이다.
        # 실제 사전분포는 고리 모양으로 집중돼 있으므로, 제대로 된 계획기는
        # 이 값을 크게 넘는 게 정상이다. "계획기가 무작위보다 나은가"의 기준선이
        # 아니라 "자원이 얼마나 부족한가"의 척도로 읽어야 한다.
        "uniform_prior_random_search_detection": 1.0
        - exp(-effective_area_m2 / mission.total_area_m2),
        # 전 면적을 발자국으로 한 번씩 덮으려면 몇 대가 / 몇 초가 필요한가.
        "uav_count_for_full_coverage": mission.uav_count / geometric_ceiling,
        "mission_time_s_for_full_coverage": mission_time_s / geometric_ceiling,
        "note": (
            "이동시간 0, 중복 0의 낙관 상한이다. 실측 면적탐색률이 이 값의 "
            "절반 근처면 계획기는 잘 하고 있는 것이고, 이 값 자체가 낮으면 "
            "문제는 계획기가 아니라 자원 배정이다. "
            "random_search_detection_ceiling 은 노력을 무작위로 뿌렸을 때의 "
            "탐지확률로, 계획기는 이 값을 넘어야 의미가 있다."
        ),
    }


def run(config: ChapterConfig, options: RunOptions) -> dict:
    mission: MissionConfig = config.mission()
    sensor = config.sensor()
    envelope = sensor.search_envelope
    assert envelope is not None
    centerline_m = 1_000.0
    detection = config.common_get("detection", {})
    # 한 번만 계산해서 결과 표와 자원상한 계산이 **같은 W**를 쓰게 한다.
    sweep_rows = sweep_width_table(sensor)
    return {
        "chapter": "1",
        "provenance": config.provenance(),
        "target_profiles": [
            {
                "name": profile.name,
                "signature": profile.signature.name,
                "length_m": profile.signature.length_m,
                "width_m": profile.signature.width_m,
                "height_m": profile.signature.height_m,
                "eo_contrast": profile.signature.eo_contrast,
                "ir_contrast": profile.signature.ir_contrast,
                "motion_model": profile.motion_spec(
                    max_speed_mps=mission.target_max_speed_mps
                ).motion_model,
                "imm5_mode_names": list(profile.imm5_behavior.mode_names)
                if hasattr(profile.imm5_behavior, "mode_names")
                else [],
                "imm5_mode_probabilities": list(
                    profile.imm5_behavior.mode_probabilities
                ),
                "calibration_status": profile.calibration_status,
            }
            for profile in NOMINAL_TARGET_PROFILES
        ],
        "mission": {
            "uav_count": mission.uav_count,
            "search_radius_m": mission.search_radius_m,
            "search_area_m2": mission.total_area_m2,
            "search_speed_mps": mission.search_speed_mps,
            "transit_speed_mps": mission.transit_speed_mps,
            "target_max_speed_mps": mission.target_max_speed_mps,
            "mission_time_s": options.mission_time_s,
        },
        # 자원이 AOI를 감당하는가. 이 상한을 넘는 면적탐색률은 물리적으로
        # 불가능하므로, 어떤 계획기 결과든 이 값에 대고 읽어야 한다.
        #   A_max   = n_uav * v_search * T * W   (전이시간 0, 중복 0 가정)
        #   ceiling = A_max / (pi R^2)
        # 이 값이 낮으면 '면적을 넓게 훑는' 전략은 애초에 이길 수 없다.
        # 확률질량을 좇는 전략이 유리해지는 구간이고, 그게 본 연구의 전제다.
        "coverage_feasibility": _coverage_feasibility(
            mission, sensor, options.mission_time_s, sweep_rows
        ),
        "search_envelope": {
            "instantaneous_swath_m": sensor.instantaneous_swath_m,
            "system_support_half_width_m": sensor.coverage_half_width_m,
            "geometric_equivalent_sweep_width_m": sensor.effective_sweep_width_m,
            "vehicle_weave_half_amplitude_m": envelope.weave.half_amplitude_m,
            "vehicle_weave_wavelength_m": envelope.weave.wavelength_m,
            "seeker_half_width_m": envelope.seeker_half_width_m,
            "planned_centerline_example_m": centerline_m,
            "actual_woven_distance_example_m": sensor.actual_search_distance_m(
                centerline_m
            ),
            "track_spacing_m": sensor.track_spacing_m,
            "calibration_status": envelope.calibration_status,
            "important": "800 m support is not W = 1600 m; W = integral(PD(x), x)",
        },
        "sweep_width_table": sweep_rows,
        "detection_contract": {
            **evidence_contract(
                float(detection.get("reference_detection_probability", 0.86)),
                float(detection.get("false_alarm_probability", 0.05)),
            ),
            "status": detection.get(
                "status",
                "A-grade modeling assumption; calibration flight test required",
            ),
        },
        "probability_contract": {
            "evidence_map": config.common_get(
                "belief.evidence_map", "binary log-odds occupancy"
            ),
            "allocation_map": config.common_get(
                "belief.allocation_map",
                "normalized single-target location probability",
            ),
            "fusion": config.common_get(
                "belief.fusion",
                "observation-ID deduplicated log-odds increments",
            ),
            "reported": ["POC", "POD", "POS"],
        },
        "monte_carlo": {
            "sample_count": config.sample_count,
            "base_seed": config.base_seed,
            "report_confidence": config.report_confidence,
            "inference": "paired seed effects with bootstrap confidence intervals",
        },
        "evaluation_protocol": {
            "mission_time_s": options.mission_time_s,
            "truth_motion_model": "IMM5",
            "truth_and_planner_terrain": "independent draws when map_error is on",
            "detection_model": "SR-Z50 EO/IR spatial model with terrain occlusion",
            "development_and_confirmatory_seeds_disjoint": True,
        },
        "kpi_definitions": {
            name: KPI_DEFINITIONS.get(name, "undocumented")
            for name in (config.kpi_names or tuple(KPI_DEFINITIONS))
        },
        # 두 모델링 상수의 근거는 아직 없다 — 그러면 "값이 이것이다" 대신
        # "이 범위에서는 결론이 바뀌지 않는다"를 보여야 한다.
        **_motion_sensitivity(config, mission),
    }


def _motion_sensitivity(config: ChapterConfig, mission: MissionConfig) -> dict:
    """``step_s`` / ``halt_probability_boost`` 민감도와 AOI 포함률 사이징.

    셋 다 ``tools/size_aoi.py`` 가 이미 재던 값이다. 도구로만 재고 결과에
    남기지 않으면 **그림 코드가 읽을 데이터가 없어서** Ch1 민감도 그림 3장이
    조용히 만들어지지 않는다. 실제로 그 상태였다.

    표본 수는 config 가 정한다. 도구 기본값 4000 을 그대로 쓰면 전체 실행에
    수 분이 붙으므로, 챕터 안에서는 기본 800 으로 두고 값을 결과에 적는다.
    도구 쪽 4000 표본 실행이 필요하면 그건 별도로 돌린다.
    """

    settings = config.common_get("motion_sensitivity", {}) or {}
    if not bool(settings.get("enabled", True)):
        return {
            "motion_sensitivity": {
                "status": "disabled by config",
                "reason": str(settings.get("disabled_reason", "")),
            }
        }
    sample_count = int(settings.get("sample_count", 800))
    horizon_s = float(settings.get("horizon_s", config.common_get("mission.mission_time_s", 300.0)))
    profiles = tuple(settings.get("profiles", ("tank", "tel")))
    steps_s = tuple(float(value) for value in settings.get("steps_s", (30.0, 10.0, 5.0)))
    boosts = tuple(float(value) for value in settings.get("halt_boosts", (0.0, 0.15, 0.35, 0.5)))
    shared = {"sample_count": sample_count, "horizon_s": horizon_s}

    step_block = {
        profile: step_sensitivity(
            config, mission, steps_s=steps_s, profile_key=profile, **shared
        )["rows"]
        for profile in profiles
    }
    halt_result = {
        profile: halt_boost_sweep(
            config, mission, boosts=boosts, profile_key=profile, **shared
        )
        for profile in profiles
    }
    aoi = containment_radius_m(config, mission, profile_keys=profiles, **shared)
    return {
        "motion_sensitivity": {
            "sample_count": sample_count,
            "horizon_s": horizon_s,
            "step_s": {
                "profiles": step_block,
                "question": (
                    "step_s 30 s 가 궤적을 과도하게 매끄럽게 만드는가. "
                    "순변위가 step 에 둔감하면 AOI 사이징에는 영향이 없다"
                ),
            },
            "halt_probability_boost": {
                "profiles": {
                    profile: result["rows"] for profile, result in halt_result.items()
                },
                "declared_boost": next(
                    iter(halt_result.values())
                )["declared_boost"] if halt_result else None,
                "question": (
                    "0.35 라는 값에 근거가 없다. 0~0.5 에서 종료반경 분포가 "
                    "AOI 결정을 뒤집는지 본다"
                ),
            },
        },
        "aoi_design": {
            "containment_probability": aoi["containment"],
            "derived_search_radius_m": aoi["required_search_radius_m"],
            "configured_search_radius_m": mission.search_radius_m,
            "max_speed_rule_radius_m": aoi["max_speed_rule_radius_m"],
            "containment_trade_off": aoi["containment_trade_off"],
            "profiles": {
                profile: {
                    "containment_radius_m": entry["containment_radius_m"],
                    "presence_containment_radius_m": entry[
                        "presence_containment_radius_m"
                    ],
                    "sample_count": entry["sample_count"],
                }
                for profile, entry in aoi["per_profile"].items()
            },
        },
    }
