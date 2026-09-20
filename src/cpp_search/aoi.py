"""AOI 크기와 표적 운동 모델의 민감도를 **측정**한다 (가정하지 않는다).

왜 이 파일이 필요한가
---------------------
탐색반경을 처음에는 ``R = v_max x T_lead`` 로 잡았다. 40 km/h x 300 s = 3333 m.
이 규칙에는 두 가지 결함이 있다.

1. **표적은 최대속도로 다니지 않는다.** IMM5는 모드별 속도구간에서 균등추출
   하므로 앙상블 평균속도는 최대속도의 40~60% 수준이다
   (MANEUVER_HEAVY 15.4, RELOCATION_HEAVY 24.1 km/h). 최대속도 규칙은 과대추정.
2. **그런데 꼬리는 규칙보다 멀리 간다.** 초기위치가 링 분포이고 개별 표적이
   HIGH_CV를 계속 유지하면 종료반경이 3333 m를 넘는다. 최대속도 규칙은
   동시에 과소추정이기도 하다.

두 오차가 방향이 반대라 "적당히 맞겠지"가 통하지 않는다. 그래서 규칙을 바꾼다::

    R_AOI = quantile( ||x_i(T)|| , q )      q = config aoi.containment (0.99)

즉 **임무 종료 시점 표적 위치 분포의 q-분위수**. 이 측정이 성립하려면 진리
궤적이 경계에서 끊기면 안 된다 (``truth_boundary_mode = "open"``). 끊긴 궤적으로
재면 항상 "기존 R + 한 스텝"이 나와서 자기충족적 답이 된다.

고정점 함정 (반드시 읽을 것)
----------------------------
``TargetPrior(kind="moving-ring")`` 는 초기 분포를 **AOI 반경의 비율**로 정의한다
(mu = 0.65 R, sigma = 0.15 R). 그래서 위 규칙을 그대로 반복 적용하면 발산한다::

    R_new = quantile(||x_0(R) + dx(T)||) ~ R + (변위 꼬리)    ->    R -> infinity

실제로 R = 3333 m 로 재면 5395 m 가 나오고, 5395 m 로 다시 재면 또 그만큼
늘어난다. 이건 표적이 멀리 간다는 뜻이 아니라 **규칙이 잘못 세워졌다**는 뜻이다.

바로잡는 방법은 두 양을 분리하는 것이다.

* **초기 표적위치 불확실성 (cue ring)** — 발사 시점의 표적 위치를 얼마나
  아는가. 이건 표적 정보/센서 정확도가 정하는 **절대 미터** 값이지,
  우리가 그리는 AOI의 함수가 아니다. config ``aoi.cue_ring`` 에 미터로 적는다.
* **AOI 반경** — 그 cue ring에서 출발한 표적이 T초 뒤 어디 있는지의 q-분위수.

이렇게 하면 R_AOI 는 R 에 의존하지 않는 **한 번에 정해지는 값**이 된다.
기본 cue ring 은 원래 규칙과의 연속성을 위해 기준반경 3333.3 m 에서
(mu = 2166.7 m, sigma = 500 m) 로 잡았고, 이건 명시적 가정이므로
표적 정보 정확도가 확정되면 그 값으로 갈아끼워야 한다.

무엇을 재는가
-------------
* :func:`end_radius_samples` — 종료반경 표본.
* :func:`containment_radius_m` — 위 분위수. AOI 사이징의 답.
* :func:`step_sensitivity` — ``step_s`` 를 30/10/5 s 로 바꿨을 때 변위 분포가
  얼마나 움직이는가. 30 s면 임무 300 s 동안 모드 전이가 10번뿐이라, 이산화가
  거칠어 실제보다 직선적인 궤적이 나올 수 있다. 그 편향의 크기를 잰다.
* :func:`halt_boost_sweep` — ``terrain_coupling.halt_probability_boost`` 는
  근거가 없는 계수다. 값을 쓸 수 없으면 최소한 결과가 그 값에 얼마나
  민감한지는 밝혀야 한다.

의존
----
* 위: ``research.truth``(궤적 생성), ``cpp_search.core.terrain``, ``cpp_search.core.simulation``.
* 아래: ``tools/size_aoi.py``(CLI), ``tests/test_aoi.py``.
"""

from __future__ import annotations

from dataclasses import replace
from statistics import mean, median
from typing import Any, Sequence

from cpp_search.core.models import MissionConfig
from cpp_search.core.simulation import EvaluationConfig
from cpp_search.core.terrain import build_synthetic_terrain
from cpp_search.config import ChapterConfig
from cpp_search.truth import (
    REFERENCE_CUE_RADIUS_M,
    TerrainCoupling,
    cue_prior,
    generate_truth,
    target_profile,
    terrain_coupling,
)


def _quantile(values: Sequence[float], q: float) -> float:
    """선형보간 분위수. numpy 없이도 같은 답이 나오게 명시적으로 적는다."""

    if not values:
        raise ValueError("빈 표본에서는 분위수를 잴 수 없다")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _path_length_m(points) -> float:
    return sum(
        earlier.distance_to(later) for earlier, later in zip(points, points[1:])
    )


def end_radius_samples(
    config: ChapterConfig,
    mission: MissionConfig,
    *,
    profile_key: str = "tank",
    sample_count: int = 4000,
    seed: int = 20260830,
    horizon_s: float = 300.0,
    use_terrain: bool = True,
    step_s: float | None = None,
    coupling: TerrainCoupling | None = None,
) -> dict[str, list[float]]:
    """임무 종료 시점의 중심거리·경로장·순변위 표본.

    두 가지를 강제한다.

    * ``boundary_mode="open"`` — 닫힌 채로 재면 분위수가 기존 AOI에 붙는다.
    * **cue ring 고정** — 초기분포를 현재 AOI가 아니라 절대 미터 링에서 뽑는다.
      이게 없으면 R 을 키울수록 답도 같이 커져 규칙이 발산한다(모듈 docstring).
    """

    profile = target_profile(config, profile_key)
    prior, prior_radius_m, _ = cue_prior(config)
    # 초기분포·지형 척도는 cue 기준반경에 고정한다. AOI 후보 반경(mission)이
    # 초기분포를 밀어내면 측정이 자기 자신을 재는 꼴이 된다.
    cue_mission = replace(mission, search_radius_m=prior_radius_m)
    evaluation = EvaluationConfig(
        sample_count=sample_count,
        seed=seed,
        detection_time_limit_s=horizon_s,
    )
    terrain = (
        build_synthetic_terrain(prior_radius_m, seed=seed) if use_terrain else None
    )
    trajectories = generate_truth(
        config,
        cue_mission,
        evaluation,
        profile,
        horizon_s=horizon_s,
        terrain=terrain,
        prior=prior,
        step_s=step_s,
        coupling=coupling,
        boundary_mode="open",
    )
    end_radius: list[float] = []
    path_length: list[float] = []
    displacement: list[float] = []
    presence_radius: list[float] = []
    for trajectory in trajectories:
        end_radius.append(cue_mission.center.distance_to(trajectory.points[-1]))
        path_length.append(_path_length_m(trajectory.points))
        displacement.append(trajectory.points[0].distance_to(trajectory.points[-1]))
        # 체류 가중 반경: 탐지는 T 시점이 아니라 임무 내내 일어난다. 종료위치만
        # 보면 "마지막에 잠깐 밖으로 나간 표적" 때문에 AOI를 과하게 키우게 된다.
        presence_radius.extend(
            cue_mission.center.distance_to(point) for point in trajectory.points
        )
    return {
        "end_radius_m": end_radius,
        "path_length_m": path_length,
        "net_displacement_m": displacement,
        "presence_radius_m": presence_radius,
    }


def describe(values: Sequence[float]) -> dict[str, float]:
    """분포 한 줄 요약. 꼬리를 봐야 하므로 90/95/99를 모두 낸다."""

    return {
        "mean": mean(values),
        "median": median(values),
        "p05": _quantile(values, 0.05),
        "p50": _quantile(values, 0.50),
        "p90": _quantile(values, 0.90),
        "p95": _quantile(values, 0.95),
        "p99": _quantile(values, 0.99),
        "max": max(values),
    }


def containment_radius_m(
    config: ChapterConfig,
    mission: MissionConfig,
    *,
    containment: float | None = None,
    profile_keys: Sequence[str] = ("tank", "tel"),
    **kwargs: Any,
) -> dict[str, Any]:
    """포함률 규칙으로 AOI 반경을 정한다.

    Tank와 TEL은 거동이 다르므로 분위수도 다르다. 하나의 AOI로 둘 다 담아야
    하니 **둘 중 큰 쪽**을 쓴다.
    """

    q = float(
        containment
        if containment is not None
        else config.common_get("aoi.containment", 0.99)
    )
    per_profile: dict[str, Any] = {}
    for key in profile_keys:
        samples = end_radius_samples(config, mission, profile_key=key, **kwargs)
        per_profile[key] = {
            "end_radius_m": describe(samples["end_radius_m"]),
            "path_length_m": describe(samples["path_length_m"]),
            "net_displacement_m": describe(samples["net_displacement_m"]),
            "presence_radius_m": describe(samples["presence_radius_m"]),
            "containment_radius_m": _quantile(samples["end_radius_m"], q),
            "presence_containment_radius_m": _quantile(
                samples["presence_radius_m"], q
            ),
            "sample_count": len(samples["end_radius_m"]),
        }
    required = max(entry["containment_radius_m"] for entry in per_profile.values())
    # 포함률을 낮추면 AOI가 얼마나 줄고 면적이 얼마나 싸지는가. 0.99가
    # 유일한 답이 아니라는 걸 결과에 남겨서 선택이 가능하게 한다.
    trade_off = []
    for level in (0.90, 0.95, 0.99):
        radius = max(
            _quantile(
                end_radius_samples(config, mission, profile_key=key, **kwargs)[
                    "end_radius_m"
                ],
                level,
            )
            for key in profile_keys
        )
        trade_off.append(
            {
                "containment": level,
                "search_radius_m": radius,
                "area_ratio_vs_current": (radius / mission.search_radius_m) ** 2,
            }
        )
    _, _, cue = cue_prior(config)
    return {
        "containment": q,
        "cue_ring": cue,
        "current_search_radius_m": mission.search_radius_m,
        "required_search_radius_m": required,
        "max_speed_rule_radius_m": (
            mission.target_max_speed_mps
            * float(config.common_get("mission.mission_time_s", 300.0))
        ),
        "per_profile": per_profile,
        "containment_trade_off": trade_off,
        "presence_required_search_radius_m": max(
            entry["presence_containment_radius_m"] for entry in per_profile.values()
        ),
        "fixed_point_safe": True,
        "fixed_point_note": (
            "cue ring이 절대 미터로 고정돼 있으므로 이 값은 현재 AOI 반경에 "
            "의존하지 않는다. --apply 후 다시 재도 같은 값이 나와야 한다."
        ),
    }


def step_sensitivity(
    config: ChapterConfig,
    mission: MissionConfig,
    *,
    steps_s: Sequence[float] = (30.0, 10.0, 5.0),
    profile_key: str = "tank",
    **kwargs: Any,
) -> dict[str, Any]:
    """``step_s`` 이산화 편향 측정.

    같은 seed·같은 프로파일로 step만 바꾼다. 경로장이 step에 따라 계속 늘면
    (Brownian 성분이 지배) 30 s는 궤적을 과도하게 매끄럽게 만들고 있다는 뜻.
    순변위가 거의 변하지 않으면 AOI 사이징에는 영향이 없다는 뜻이다.
    """

    rows: list[dict[str, Any]] = []
    for step in steps_s:
        samples = end_radius_samples(
            config, mission, profile_key=profile_key, step_s=step, **kwargs
        )
        rows.append(
            {
                "step_s": step,
                "end_radius_m": describe(samples["end_radius_m"]),
                "path_length_m": describe(samples["path_length_m"]),
                "net_displacement_m": describe(samples["net_displacement_m"]),
            }
        )
    baseline = rows[0]
    for row in rows[1:]:
        row["vs_baseline"] = {
            key: row[key]["p95"] - baseline[key]["p95"]
            for key in ("end_radius_m", "path_length_m", "net_displacement_m")
        }
    return {"profile": profile_key, "baseline_step_s": rows[0]["step_s"], "rows": rows}


def halt_boost_sweep(
    config: ChapterConfig,
    mission: MissionConfig,
    *,
    boosts: Sequence[float] = (0.0, 0.15, 0.35, 0.5),
    profile_key: str = "tank",
    **kwargs: Any,
) -> dict[str, Any]:
    """``halt_probability_boost`` 민감도.

    이 계수는 은폐지형에서 정지확률을 올린다. 교리·정보 근거가 없으므로
    "값이 0.35다"라고 주장하는 대신 "0~0.5 범위에서 결론이 뒤집히지 않는다"를
    보이는 것이 정직하다.
    """

    base = terrain_coupling(config)
    rows: list[dict[str, Any]] = []
    for boost in boosts:
        samples = end_radius_samples(
            config,
            mission,
            profile_key=profile_key,
            coupling=replace(base, halt_probability_boost=boost),
            **kwargs,
        )
        rows.append(
            {
                "halt_probability_boost": boost,
                "end_radius_m": describe(samples["end_radius_m"]),
                "path_length_m": describe(samples["path_length_m"]),
            }
        )
    declared = [row for row in rows if row["halt_probability_boost"] == base.halt_probability_boost]
    reference = declared[0] if declared else rows[0]
    for row in rows:
        row["end_radius_p99_delta_m"] = (
            row["end_radius_m"]["p99"] - reference["end_radius_m"]["p99"]
        )
    return {
        "profile": profile_key,
        "declared_boost": base.halt_probability_boost,
        "rows": rows,
    }


def net_displacement_samples(
    config: ChapterConfig,
    mission: MissionConfig,
    *,
    profile_key: str = "tank",
    sample_count: int = 4000,
    seed: int = 20260830,
    horizon_s: float = 480.0,
) -> list[float]:
    """TP 기준 순변위 |x(T) - x(0)| 표본. AOI 를 TP 중심으로 잡을 때의 사이징 근거.

    ``end_radius_samples`` 가 AOI **중심**까지의 거리라면, 이것은 각 표적이
    **자기 출발점**에서 얼마나 멀어졌는가다. 사전분포의 반경 오프셋을 빼고
    순수 이동만 남기므로, 표적이 TP 에 있다는 시나리오의 AOI 는 이 분위수다.
    """

    from math import hypot

    prior = cue_prior(config, mission.search_radius_m)[0]
    profile = target_profile(config, profile_key)
    evaluation = EvaluationConfig(
        sample_count=sample_count, seed=seed, detection_time_limit_s=horizon_s
    )
    terrain = build_synthetic_terrain(mission.search_radius_m, seed=seed)
    trajectories = generate_truth(
        config, mission, evaluation, profile, horizon_s=horizon_s,
        terrain=terrain, prior=prior, boundary_mode="open",
    )
    return [
        hypot(t.points[-1].x - t.points[0].x, t.points[-1].y - t.points[0].y)
        for t in trajectories
    ]


def net_displacement_containment_radius_m(
    config: ChapterConfig,
    mission: MissionConfig,
    *,
    containment: float = 0.99,
    **kwargs,
) -> dict[str, Any]:
    """두 표적 프로파일의 순변위 분위수와 그 최댓값(= AOI 반경)."""

    per_profile = {
        key: _quantile(net_displacement_samples(config, mission, profile_key=key, **kwargs), containment)
        for key in ("tank", "tel")
    }
    return {
        "rule": "net-displacement containment",
        "containment": containment,
        "per_profile_radius_m": per_profile,
        "required_search_radius_m": max(per_profile.values()),
    }
