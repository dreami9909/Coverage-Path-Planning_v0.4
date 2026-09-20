"""Statistical reliability contract for Chapter 2-5 experiments.

The independent inference unit is a planning/evaluation seed. Target
trajectories nested under one seed reduce within-scenario Monte Carlo noise,
but they do not replace independent terrain, belief, route, and network seeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from statistics import NormalDist
from typing import Any, Mapping, Sequence

import numpy as np

from cpp_search.config import derive_seeds


@dataclass(frozen=True, slots=True)
class ReliabilityCriteria:
    confidence: float = 0.95
    bootstrap_resamples: int = 5_000
    minimum_seed_count: int = 20
    minimum_samples_per_seed: int = 1_000
    maximum_detection_ci_half_width: float = 0.03
    maximum_restricted_time_ci_half_width_s: float = 10.0
    maximum_coverage_ci_half_width: float = 0.02
    maximum_detection_change: float = 0.01
    maximum_restricted_time_change_s: float = 5.0

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must be in (0, 1)")
        if self.bootstrap_resamples <= 0:
            raise ValueError("bootstrap_resamples must be positive")
        if self.minimum_seed_count < 2:
            raise ValueError("minimum_seed_count must be at least two")
        if self.minimum_samples_per_seed <= 0:
            raise ValueError("minimum_samples_per_seed must be positive")
        tolerances = (
            self.maximum_detection_ci_half_width,
            self.maximum_restricted_time_ci_half_width_s,
            self.maximum_coverage_ci_half_width,
            self.maximum_detection_change,
            self.maximum_restricted_time_change_s,
        )
        if min(tolerances) <= 0.0:
            raise ValueError("reliability tolerances must be positive")

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any] | None,
    ) -> "ReliabilityCriteria":
        values = values or {}
        return cls(
            confidence=float(values.get("confidence", 0.95)),
            bootstrap_resamples=int(values.get("bootstrap_resamples", 5_000)),
            minimum_seed_count=int(values.get("minimum_seed_count", 20)),
            minimum_samples_per_seed=int(
                values.get("minimum_samples_per_seed", 1_000)
            ),
            maximum_detection_ci_half_width=float(
                values.get("maximum_detection_ci_half_width", 0.03)
            ),
            maximum_restricted_time_ci_half_width_s=float(
                values.get("maximum_restricted_time_ci_half_width_s", 10.0)
            ),
            maximum_coverage_ci_half_width=float(
                values.get("maximum_coverage_ci_half_width", 0.02)
            ),
            maximum_detection_change=float(
                values.get("maximum_detection_change", 0.01)
            ),
            maximum_restricted_time_change_s=float(
                values.get("maximum_restricted_time_change_s", 5.0)
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


def resolve_evaluation_seeds(
    declared: Sequence[int] | None,
    *,
    base_seed: int,
    default_count: int,
    limit: int | None,
    salt: int,
) -> tuple[int, ...]:
    seeds = (
        tuple(int(seed) for seed in declared)
        if declared
        else derive_seeds(base_seed, default_count, salt=salt)
    )
    if len(set(seeds)) != len(seeds):
        raise ValueError("evaluation seeds must be unique")
    if limit is not None:
        if limit <= 0:
            raise ValueError("evaluation seed limit must be positive")
        seeds = seeds[:limit]
    if not seeds:
        raise ValueError("at least one evaluation seed is required")
    return seeds


def wilson_score_interval(
    successes: float,
    trials: float,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    if trials <= 0.0 or not 0.0 <= successes <= trials:
        raise ValueError("successes/trials are inconsistent")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    half_width = (
        z
        * (
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials**2)
        )
        ** 0.5
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def required_binomial_samples(
    maximum_half_width: float,
    *,
    confidence: float = 0.95,
) -> int:
    if not 0.0 < maximum_half_width < 1.0:
        raise ValueError("maximum_half_width must be in (0, 1)")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    return ceil(z * z * 0.25 / maximum_half_width**2)


def bootstrap_weighted_mean_ci(
    values: Sequence[float],
    weights: Sequence[float] | None = None,
    *,
    seed: int,
    resamples: int,
    confidence: float,
) -> tuple[float, float, float] | None:
    """Percentile interval from resampling whole seed clusters."""

    numeric = np.asarray(tuple(values), dtype=float)
    if numeric.size == 0:
        return None
    mass = (
        np.ones(numeric.size, dtype=float)
        if weights is None
        else np.asarray(tuple(weights), dtype=float)
    )
    if mass.shape != numeric.shape or np.any(mass < 0.0) or mass.sum() <= 0.0:
        raise ValueError("bootstrap weights must have positive mass")
    estimate = float(np.dot(numeric, mass) / mass.sum())
    if numeric.size < 2:
        return estimate, estimate, estimate
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, numeric.size, size=(resamples, numeric.size))
    sampled_values = numeric[indices]
    sampled_mass = mass[indices]
    estimates = np.sum(sampled_values * sampled_mass, axis=1) / np.sum(
        sampled_mass,
        axis=1,
    )
    alpha = 0.5 * (1.0 - confidence)
    return (
        estimate,
        float(np.quantile(estimates, alpha)),
        float(np.quantile(estimates, 1.0 - alpha)),
    )


def paired_metric_report(
    proposed: Sequence[Mapping[str, Any]],
    comparator: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    criteria: ReliabilityCriteria,
    bootstrap_seed: int,
    left_label: str | None = None,
    right_label: str | None = None,
) -> dict[str, Any]:
    """Validate exact seed pairing before bootstrapping paired differences.

    차이는 항상 **첫 인자 빼기 둘째 인자**다. 그런데 호출부마다 무엇을 먼저
    넘기는지가 달랐다 — Ch2 / Ch5-a / Ch6 는 제안을 먼저 넘겨 ``제안 - 조건``
    을 냈고 Ch5-b-2 는 조건을 먼저 넘겨 ``조건 - 기준`` 을 냈다. 블록 이름은
    양쪽 다 ``paired_comparisons_vs_<기준>`` 이라 구별이 되지 않았고, 그림은
    한 가지 규약을 가정해 축 라벨을 붙였다. 그래서 **Ch6 의 핵심 그림이 부호가
    반대로 읽혔다** — 기준선이 제안보다 0.31 나은 것처럼 보였는데 실제는 제안이
    0.35 낫다.

    그래서 ``left_label`` / ``right_label`` 을 받아 무엇에서 무엇을 뺐는지
    ``mean_difference_is`` 로 함께 내보낸다. 그림은 그 문자열로 축을 짓는다.
    저장된 숫자의 부호를 뒤집는 대신 방향을 명시하는 쪽을 골랐다 — 부호를
    뒤집으면 이미 서술된 결과와 어긋나고, 명시는 어긋날 수 없다.
    """

    left = _records_by_seed(proposed)
    right = _records_by_seed(comparator)
    if set(left) != set(right):
        raise ValueError("paired comparison requires identical seed sets")
    seeds = sorted(left)
    differences = [
        float(left[seed][metric]) - float(right[seed][metric])
        for seed in seeds
    ]
    interval = bootstrap_weighted_mean_ci(
        differences,
        seed=bootstrap_seed,
        resamples=criteria.bootstrap_resamples,
        confidence=criteria.confidence,
    )
    if interval is None:
        raise ValueError("paired comparison requires at least one seed")
    estimate, low, high = interval
    inference_valid = len(seeds) >= criteria.minimum_seed_count
    return {
        "mean_difference": estimate,
        # 부호의 방향. 없으면 그림이 규약을 추측해야 하고, 추측은 틀렸다.
        "mean_difference_is": (
            f"{left_label} minus {right_label}"
            if left_label and right_label
            else "first argument minus second argument (labels not supplied)"
        ),
        "confidence_interval": [low, high] if len(seeds) >= 2 else None,
        "confidence": criteria.confidence,
        "ci_method": "paired planning-seed cluster percentile bootstrap",
        "seed_count": len(seeds),
        "paired_seeds": seeds,
        "inference_valid": inference_valid,
        "significant": bool(inference_valid and (low > 0.0 or high < 0.0)),
        "status": "valid" if inference_valid else "insufficient-seeds",
    }


def _records_by_seed(
    records: Sequence[Mapping[str, Any]],
) -> dict[int, Mapping[str, Any]]:
    indexed: dict[int, Mapping[str, Any]] = {}
    for record in records:
        if "seed" not in record:
            raise ValueError("paired records must contain a seed")
        seed = int(record["seed"])
        if seed in indexed:
            raise ValueError(f"duplicate paired record for seed {seed}")
        indexed[seed] = record
    return indexed
