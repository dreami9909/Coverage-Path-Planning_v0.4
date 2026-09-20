"""공통 성능지표 블록 — 모든 챕터가 같은 이름으로 같은 값을 낸다.

챕터마다 지표 이름과 위치가 다르면 비교가 불가능하다. 그래서 결과 JSON의
모든 조건은 아래 ``kpi`` 블록을 **똑같은 모양으로** 포함한다.

    kpi = {
      "unique_area_coverage_ratio":        면적의 몇 %를 훑었나
      "detection_probability_within_limit": 표적을 탐지할 확률
      "conditional_mean_detection_time_s":  탐지 성공 시 평균 탐지시간
      "restricted_mean_detection_time_s":   실패를 제한시간으로 절단한 평균
      "failure_rate_within_limit":          실패율 (= 1 - 탐지확률)
      "team": { ... }                        (자산 2대 이상일 때만)
    }

왜 탐지시간을 둘 다 남기는가
----------------------------
조건부 평균만 보면 **탐색을 못 할수록 좋아 보인다**. 쉬운 표적 몇 개만 잡고
나머지를 놓치면 성공 표본의 평균이 짧아지기 때문이다. Ch6 실측이 그 예다.

    Independent : Pd 0.247, 조건부 178 s, 절단 259 s
    제안        : Pd 0.611, 조건부 172 s, 절단 201 s

조건부만 보면 둘이 비슷해 보이지만 절단 평균은 58초 차이가 난다. 조건부는
"잡았을 때 얼마나 빨랐나", 절단은 "임무 전체로 보면 얼마나 빨랐나"를 답한다.
**순위를 매길 때는 절단 평균을 쓴다.**

수식
----
    Pd    = #{ T_d <= T_lim } / N
    fail  = 1 - Pd                                   (정의상 정확히 여집합)
    T_cond = mean{ T_d : T_d <= T_lim }              (성공 표본만)
    T_rest = mean{ min(T_d, T_lim) }                 (실패는 T_lim으로 절단)

의존
----
* 위: ``cpp_search.core.simulation.PerformanceMetrics``,
  ``cpp_search.core.evaluation.GeometricMetrics``.
* 아래: 모든 ``chapters/chapter*.py``와 ``runner``의 요약표.
"""

from __future__ import annotations

from dataclasses import replace
from math import isfinite
from typing import Any, Mapping, Sequence

from cpp_search.core.evaluation import GeometricMetrics
from cpp_search.core.simulation import PerformanceMetrics, detection_time_statistics
from cpp_search.reliability import (
    ReliabilityCriteria,
    bootstrap_weighted_mean_ci,
    required_binomial_samples,
    wilson_score_interval,
)


#: 모든 챕터가 반드시 내보내야 하는 다섯 개 지표.
KPI_KEYS: tuple[str, ...] = (
    "unique_area_coverage_ratio",
    "detection_probability_within_limit",
    "conditional_mean_detection_time_s",
    "restricted_mean_detection_time_s",
    "failure_rate_within_limit",
)

#: 진단용 파생지표. 순위를 매기는 값은 아니지만 결과를 읽으려면 필요하다.
DIAGNOSTIC_KPI_KEYS: tuple[str, ...] = (
    "escape_rate",
    "detection_probability_per_covered_area",
)

#: 자산이 둘 이상일 때만 의미가 있는 군집 지표.
TEAM_KPI_KEYS: tuple[str, ...] = (
    "coverage_redundancy_ratio",
    "mean_route_distance_m",
    "max_route_distance_m",
    "team_total_distance_m",
)


def mission_kpi(
    performance: PerformanceMetrics,
    *,
    uav_count: int = 1,
    geometry: GeometricMetrics | None = None,
    extra_team: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Monte Carlo 결과를 표준 KPI 블록으로 변환한다."""

    block: dict[str, Any] = {
        "unique_area_coverage_ratio": performance.unique_area_coverage_ratio,
        "detection_probability_within_limit": (
            performance.detection_probability_within_limit
        ),
        "conditional_mean_detection_time_s": (
            performance.conditional_mean_detection_time_s
        ),
        "restricted_mean_detection_time_s": (
            performance.restricted_mean_detection_time_s
        ),
        "failure_rate_within_limit": performance.failure_rate_within_limit,
        "sample_count": performance.sample_count,
        "detected_count": performance.detected_count,
        # 실패를 두 가지로 쪼갠다. AOI 밖으로 나가서 놓친 것(escape)은 AOI
        # 사이징 문제이고, 안에 있는데 놓친 것은 계획기 문제다. 하나로 뭉치면
        # 어느 쪽을 고쳐야 하는지 알 수 없다.
        "escape_rate": (
            performance.escaped_missed_count / performance.sample_count
            if performance.sample_count
            else 0.0
        ),
        "in_area_miss_rate": (
            performance.stayed_missed_count / performance.sample_count
            if performance.sample_count
            else 0.0
        ),
        # 훑은 면적 1단위당 탐지확률. 자원이 AOI를 못 덮는 구간
        # (Ch0 coverage_feasibility 참조)에서는 '얼마나 넓게'가 아니라
        # '얼마나 잘 골랐나'가 승부처이므로, 이 값이 전략 비교의 핵심이다.
        "detection_probability_per_covered_area": (
            performance.detection_probability_within_limit
            / performance.unique_area_coverage_ratio
            if performance.unique_area_coverage_ratio > 0.0
            else 0.0
        ),
    }
    if geometry is not None:
        block["planned_centerline_distance_m"] = (
            geometry.planned_centerline_distance_m
        )
        block["actual_woven_distance_m"] = geometry.total_distance_m
    if uav_count > 1:
        team = {
            "uav_count": uav_count,
            "coverage_redundancy_ratio": performance.coverage_redundancy_ratio,
            "mean_route_distance_m": performance.mean_route_distance_m,
            "max_route_distance_m": performance.max_route_distance_m,
            "team_total_distance_m": performance.total_distance_m,
        }
        if extra_team:
            team.update(extra_team)
        block["team"] = team
    return block


def mission_kpi_sample_prefixes(
    performance: PerformanceMetrics,
    checkpoints: Sequence[int],
    *,
    uav_count: int = 1,
    geometry: GeometricMetrics | None = None,
    extra_team: dict[str, Any] | None = None,
) -> dict[int, dict[str, Any]]:
    """Re-score nested trajectory prefixes without re-running the planner."""

    if not performance.detection_times_s:
        raise ValueError("performance does not retain raw detection times")
    reports: dict[int, dict[str, Any]] = {}
    for checkpoint in sorted(set(int(value) for value in checkpoints)):
        if checkpoint <= 0 or checkpoint > performance.sample_count:
            continue
        times = performance.detection_times_s[:checkpoint]
        statistics = detection_time_statistics(times, performance.evaluation_window_s)
        prefix = replace(
            performance,
            conditional_mean_detection_time_s=float(
                statistics["conditional_mean_detection_time_s"]
            ),
            restricted_mean_detection_time_s=float(
                statistics["restricted_mean_detection_time_s"]
            ),
            detection_probability_within_limit=float(
                statistics["detection_probability_within_limit"]
            ),
            failure_rate_within_limit=float(statistics["failure_rate_within_limit"]),
            detected_count=int(statistics["detected_count"]),
            sample_count=checkpoint,
            detection_times_s=times,
        )
        reports[checkpoint] = mission_kpi(
            prefix,
            uav_count=uav_count,
            geometry=geometry,
            extra_team=extra_team,
        )
    return reports


def static_kpi(
    *,
    unique_area_coverage_ratio: float,
    detection_probability: float,
    conditional_mean_detection_time_s: float,
    restricted_mean_detection_time_s: float,
    sample_count: int,
    detected_count: int,
    planned_centerline_distance_m: float | None = None,
    actual_woven_distance_m: float | None = None,
) -> dict[str, Any]:
    """정지표적 실험(Ch1)용 KPI 블록. 이동표적 쪽과 이름이 같아야 한다."""

    block: dict[str, Any] = {
        "unique_area_coverage_ratio": unique_area_coverage_ratio,
        "detection_probability_within_limit": detection_probability,
        "conditional_mean_detection_time_s": conditional_mean_detection_time_s,
        "restricted_mean_detection_time_s": restricted_mean_detection_time_s,
        "failure_rate_within_limit": 1.0 - detection_probability,
        "sample_count": sample_count,
        "detected_count": detected_count,
    }
    if planned_centerline_distance_m is not None:
        block["planned_centerline_distance_m"] = planned_centerline_distance_m
    if actual_woven_distance_m is not None:
        block["actual_woven_distance_m"] = actual_woven_distance_m
    return block


def summarise_kpi(
    blocks: list[dict[str, Any]],
    *,
    criteria: ReliabilityCriteria | None = None,
    bootstrap_seed: int = 20_260_830,
) -> dict[str, Any]:
    """Pool target trials and bootstrap whole planning/evaluation seed clusters."""

    if not blocks:
        return {}
    criteria = criteria or ReliabilityCriteria()
    summary: dict[str, Any] = {}
    keys = KPI_KEYS + DIAGNOSTIC_KPI_KEYS + ("in_area_miss_rate",)
    for metric_index, key in enumerate(keys):
        if key == "failure_rate_within_limit":
            continue
        values, weights = _metric_values_and_weights(blocks, key)
        if not values:
            summary[key] = _not_estimable(criteria)
            continue
        interval = bootstrap_weighted_mean_ci(
            values,
            weights,
            seed=bootstrap_seed + metric_index * 0x9E37,
            resamples=criteria.bootstrap_resamples,
            confidence=criteria.confidence,
        )
        assert interval is not None
        estimate, low, high = interval
        summary[key] = {
            "mean": estimate,
            "min": min(values),
            "max": max(values),
            "confidence_interval": [low, high] if len(values) >= 2 else None,
            "ci_half_width": 0.5 * (high - low) if len(values) >= 2 else None,
            "confidence": criteria.confidence,
            "ci_method": "planning-seed cluster percentile bootstrap",
            "valid_seed_count": len(values),
        }

    detection = summary["detection_probability_within_limit"]
    detection_interval = detection["confidence_interval"]
    summary["failure_rate_within_limit"] = {
        "mean": 1.0 - detection["mean"],
        "min": 1.0 - detection["max"],
        "max": 1.0 - detection["min"],
        "confidence_interval": (
            None
            if detection_interval is None
            else [1.0 - detection_interval[1], 1.0 - detection_interval[0]]
        ),
        "ci_half_width": detection["ci_half_width"],
        "confidence": criteria.confidence,
        "ci_method": detection["ci_method"],
        "valid_seed_count": detection["valid_seed_count"],
    }
    detected = sum(float(block.get("detected_count", 0.0)) for block in blocks)
    samples = sum(float(block.get("sample_count", 0.0)) for block in blocks)
    if samples > 0.0:
        detection["trajectory_level_wilson_interval"] = list(
            wilson_score_interval(detected, samples, confidence=criteria.confidence)
        )

    team_blocks = [block["team"] for block in blocks if "team" in block]
    if team_blocks:
        counts = {block.get("uav_count") for block in team_blocks}
        team_summary: dict[str, Any] = {
            "uav_count": counts.pop() if len(counts) == 1 else sorted(counts)
        }
        for metric_index, key in enumerate(TEAM_KPI_KEYS):
            values = [float(block[key]) for block in team_blocks if key in block]
            if not values:
                continue
            interval = bootstrap_weighted_mean_ci(
                values,
                seed=bootstrap_seed + 0x5445_414D + metric_index * 0x9E37,
                resamples=criteria.bootstrap_resamples,
                confidence=criteria.confidence,
            )
            assert interval is not None
            estimate, low, high = interval
            team_summary[key] = {
                "mean": estimate,
                "min": min(values),
                "max": max(values),
                "confidence_interval": [low, high] if len(values) >= 2 else None,
                "ci_half_width": 0.5 * (high - low) if len(values) >= 2 else None,
            }
        summary["team"] = team_summary

    sample_counts = [int(block.get("sample_count", 0)) for block in blocks]
    summary["seed_count"] = len(blocks)
    summary["total_sample_count"] = sum(sample_counts)
    summary["total_detected_count"] = detected
    summary["samples_per_seed"] = {
        "min": min(sample_counts),
        "max": max(sample_counts),
    }
    summary["reliability"] = _reliability_assessment(summary, criteria)
    return summary


def convergence_diagnostics(
    seed_records: list[dict[str, Any]],
    sample_records: Mapping[int, list[dict[str, Any]]],
    *,
    criteria: ReliabilityCriteria,
    seed_checkpoints: Sequence[int],
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Report sample-prefix and seed-prefix convergence for one condition."""

    sample_reports = [
        {
            "sample_count_per_seed": checkpoint,
            **_compact_summary(
                summarise_kpi(
                    records,
                    criteria=criteria,
                    bootstrap_seed=bootstrap_seed + checkpoint,
                )
            ),
        }
        for checkpoint, records in sorted(sample_records.items())
        if records
    ]
    seed_reports = []
    for checkpoint in sorted(set(int(value) for value in seed_checkpoints)):
        if checkpoint <= 0 or checkpoint > len(seed_records):
            continue
        seed_reports.append(
            {
                "seed_count": checkpoint,
                **_compact_summary(
                    summarise_kpi(
                        seed_records[:checkpoint],
                        criteria=criteria,
                        bootstrap_seed=bootstrap_seed + checkpoint * 0x9E37,
                    )
                ),
            }
        )
    final = summarise_kpi(
        seed_records,
        criteria=criteria,
        bootstrap_seed=bootstrap_seed,
    )
    return {
        "sample_checkpoints": sample_reports,
        "seed_checkpoints": seed_reports,
        "sample_stability": _stability(sample_reports, criteria),
        "seed_stability": _stability(seed_reports, criteria),
        "final_reliability": final["reliability"],
    }


def _metric_values_and_weights(
    blocks: list[dict[str, Any]],
    key: str,
) -> tuple[list[float], list[float]]:
    values: list[float] = []
    weights: list[float] = []
    for block in blocks:
        if key not in block:
            continue
        value = float(block[key])
        if not isfinite(value):
            continue
        if key == "conditional_mean_detection_time_s":
            weight = float(block.get("detected_count", 0.0))
        elif key in {
            "detection_probability_within_limit",
            "restricted_mean_detection_time_s",
            "escape_rate",
            "in_area_miss_rate",
        }:
            weight = float(block.get("sample_count", 1.0))
        else:
            weight = 1.0
        if weight <= 0.0:
            continue
        values.append(value)
        weights.append(weight)
    return values, weights


def _not_estimable(criteria: ReliabilityCriteria) -> dict[str, Any]:
    return {
        "mean": None,
        "min": None,
        "max": None,
        "confidence_interval": None,
        "ci_half_width": None,
        "confidence": criteria.confidence,
        "ci_method": "not estimable: no finite seed-level values",
        "valid_seed_count": 0,
    }


def _reliability_assessment(
    summary: dict[str, Any],
    criteria: ReliabilityCriteria,
) -> dict[str, Any]:
    seed_count = int(summary["seed_count"])
    minimum_samples = int(summary["samples_per_seed"]["min"])
    detection_half = summary["detection_probability_within_limit"]["ci_half_width"]
    restricted_half = summary["restricted_mean_detection_time_s"]["ci_half_width"]
    coverage_half = summary["unique_area_coverage_ratio"]["ci_half_width"]
    checks = {
        "minimum_seed_count": {
            "observed": seed_count,
            "required": criteria.minimum_seed_count,
            "passed": seed_count >= criteria.minimum_seed_count,
        },
        "minimum_samples_per_seed": {
            "observed": minimum_samples,
            "required": criteria.minimum_samples_per_seed,
            "passed": minimum_samples >= criteria.minimum_samples_per_seed,
        },
        "detection_ci_half_width": {
            "observed": detection_half,
            "required_maximum": criteria.maximum_detection_ci_half_width,
            "passed": detection_half is not None
            and detection_half <= criteria.maximum_detection_ci_half_width,
        },
        "restricted_time_ci_half_width_s": {
            "observed": restricted_half,
            "required_maximum": criteria.maximum_restricted_time_ci_half_width_s,
            "passed": restricted_half is not None
            and restricted_half <= criteria.maximum_restricted_time_ci_half_width_s,
        },
        "coverage_ci_half_width": {
            "observed": coverage_half,
            "required_maximum": criteria.maximum_coverage_ci_half_width,
            "passed": coverage_half is not None
            and coverage_half <= criteria.maximum_coverage_ci_half_width,
        },
    }
    design_passed = checks["minimum_seed_count"]["passed"] and checks[
        "minimum_samples_per_seed"
    ]["passed"]
    precision_passed = all(
        check["passed"]
        for name, check in checks.items()
        if name not in {"minimum_seed_count", "minimum_samples_per_seed"}
    )
    status = (
        "reliable"
        if design_passed and precision_passed
        else "insufficient-design"
        if not design_passed
        else "precision-not-met"
    )
    return {
        "status": status,
        "criteria": criteria.as_dict(),
        "checks": checks,
        "trajectory_level_required_sample_count": required_binomial_samples(
            criteria.maximum_detection_ci_half_width,
            confidence=criteria.confidence,
        ),
        "note": (
            "seed-cluster intervals are primary; Wilson only measures nested "
            "trajectory-level Bernoulli uncertainty"
        ),
    }


def _compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "detection_probability": summary["detection_probability_within_limit"],
        "restricted_mean_detection_time_s": summary[
            "restricted_mean_detection_time_s"
        ],
        "unique_area_coverage_ratio": summary["unique_area_coverage_ratio"],
        "reliability_status": summary["reliability"]["status"],
    }


def _stability(
    reports: list[dict[str, Any]],
    criteria: ReliabilityCriteria,
) -> dict[str, Any]:
    if len(reports) < 2:
        return {"stable": False, "status": "at least two checkpoints required"}
    previous, current = reports[-2], reports[-1]
    detection_change = abs(
        current["detection_probability"]["mean"]
        - previous["detection_probability"]["mean"]
    )
    time_change = abs(
        current["restricted_mean_detection_time_s"]["mean"]
        - previous["restricted_mean_detection_time_s"]["mean"]
    )
    return {
        "detection_probability_change": detection_change,
        "restricted_mean_detection_time_change_s": time_change,
        "maximum_detection_change": criteria.maximum_detection_change,
        "maximum_restricted_time_change_s": (
            criteria.maximum_restricted_time_change_s
        ),
        "stable": detection_change <= criteria.maximum_detection_change
        and time_change <= criteria.maximum_restricted_time_change_s,
    }
