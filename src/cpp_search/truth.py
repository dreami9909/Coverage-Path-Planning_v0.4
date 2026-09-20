"""진리 표적 앙상블 생성 — 조건은 오직 config에서 온다.

이 파일이 생기기 전에는 다섯 챕터가 각자 이렇게 적고 있었다::

    generate_moving_targets(..., terrain_bias_strength=0.6,
        terrain_offroad_probability=0.2, terrain_halt_probability_boost=0.35)

같은 숫자가 다섯 곳에 하드코딩돼 있어서 한 곳만 고치면 실험이 조용히
어긋난다. 이제 전부 ``config/chapter0_common_experiment.json``의
``target_motion`` 블록에서 읽는다.

담당하는 결정
-------------
* **거동 프로파일 배정** — Tank는 MANEUVER_HEAVY, TEL은 RELOCATION_HEAVY.
  config ``target_motion.assignment``가 정한다. 두 표적이 같은 거동을 쓰면
  로드맵의 "Tank/TEL별 평가"가 센서 시그니처 차이만 재게 된다.
* **모드 속도구간** — 모드별 [low, high] km/h. 구간 내 균등분포이므로
  앙상블 평균속도는 최대속도보다 한참 낮다 (MANEUVER_HEAVY 15.4 km/h,
  RELOCATION_HEAVY 24.1 km/h, 최대 40 km/h 대비 각각 39% / 60%).
* **경계 처리** — 진리 궤적은 ``boundary_mode="open"``으로 **끊지 않는다**.
  끊으면 종료 반경이 항상 "AOI + 한 스텝"에서 잘려서, 표적이 실제로 어디까지
  가는지 관측할 수 없다. 그 관측이 AOI 크기의 유일한 근거다.
  탐지 실패 판정은 별도로 마지막 위치가 AOI 밖인지로 한다.
* **지형 결합** — bias / offroad / halt boost 세 계수.

의존
----
* 위: ``cpp_search.core.simulation``(궤적 생성), ``cpp_search.core.motion``,
  ``cpp_search.core.profiles``, ``research.config``.
* 아래: ``chapters/chapter4,3,5a,5b2,6,7``, ``research/aoi``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from cpp_search.core.models import MissionConfig
from cpp_search.core.motion import TargetBehaviorProfile, TargetMotionSpec
from cpp_search.core.probability import TargetPrior
from cpp_search.core.profiles import (
    TANK_OPERATIONAL_PROFILE,
    TEL_OPERATIONAL_PROFILE,
    TargetOperationalProfile,
)
from cpp_search.core.simulation import EvaluationConfig, TargetTrajectory, generate_moving_targets
from cpp_search.config import ChapterConfig


PROFILE_KEYS = {"tank": "Tank-nominal", "tel": "TEL-nominal"}
_BASE_PROFILES = {
    "Tank-nominal": TANK_OPERATIONAL_PROFILE,
    "TEL-nominal": TEL_OPERATIONAL_PROFILE,
}


@dataclass(frozen=True, slots=True)
class TerrainCoupling:
    """지형이 표적 운동에 개입하는 세 계수."""

    bias_strength: float = 0.6
    offroad_probability: float = 0.2
    halt_probability_boost: float = 0.35


#: cue ring 기본값이 유도된 기준반경. AOI를 키워도 이 값은 따라 움직이지 않는다.
REFERENCE_CUE_RADIUS_M = 3333.3333333333335


def cue_prior(
    config: ChapterConfig, search_radius_m: float | None = None
) -> tuple[TargetPrior, float, dict[str, float]]:
    """초기 표적위치 불확실성을 **절대 미터**로 고정해 반환한다.

    반환값은 ``(prior, 기준반경, 기록용 dict)``.

    주의: :class:`TargetPrior` 는 링을 **비율**로 들고 있고, 실제 미터는
    ``mission.search_radius_m`` 를 곱해서 나온다. 그래서 절대 링을 유지하려면
    그 미션의 반경에 맞춰 비율을 다시 계산해 줘야 한다. 이걸 빼먹으면
    AOI를 3333 -> 5436 m 로 키웠을 때 표적 링도 2167 -> 3533 m 로 같이
    밀려나가서, 포함률 사이징이 무의미해진다 (``research/aoi.py`` 참조).

    ``search_radius_m`` 을 주면 그 반경 기준으로, 생략하면 cue 기준반경
    기준으로 비율을 만든다.
    """

    block = config.common_get("aoi.cue_ring", {}) or {}
    reference = float(block.get("reference_radius_m", REFERENCE_CUE_RADIUS_M))
    kind = str(block.get("kind", "moving-ring"))
    if kind not in {"moving-ring", "tp-centered"}:
        raise ValueError(f"aoi.cue_ring.kind must be moving-ring or tp-centered, got {kind}")
    # tp-centered: 표적이 TP(격자 중심) 근처에 있고 그 오차가 sigma 다. 링이 아니라
    # 중심 집중형 절단 Gaussian. 2026-09-13 부터 기본 시나리오.
    mean_radius_m = (
        0.0 if kind == "tp-centered"
        else float(block.get("mean_radius_m", 0.65 * reference))
    )
    sigma_m = float(block.get("sigma_m", 0.15 * reference))
    if sigma_m <= 0.0:
        raise ValueError("aoi.cue_ring.sigma_m must be positive (a point prior stalls rejection sampling)")
    scale = float(search_radius_m) if search_radius_m else reference
    prior = TargetPrior(
        kind=kind,
        mean_radius_ratio=mean_radius_m / scale,
        sigma_ratio=sigma_m / scale,
    )
    return (
        prior,
        reference,
        {
            "kind": kind,
            "mean_radius_m": mean_radius_m,
            "sigma_m": sigma_m,
            "reference_radius_m": reference,
            "applied_to_search_radius_m": scale,
        },
    )


def terrain_coupling(config: ChapterConfig) -> TerrainCoupling:
    block = config.common_get("target_motion.terrain_coupling", {})
    return TerrainCoupling(
        bias_strength=float(block.get("bias_strength", 0.6)),
        offroad_probability=float(block.get("offroad_probability", 0.2)),
        halt_probability_boost=float(block.get("halt_probability_boost", 0.35)),
    )


def motion_step_s(config: ChapterConfig) -> float:
    return float(config.common_get("target_motion.step_s", 30.0))


def truth_boundary_mode(config: ChapterConfig) -> str:
    return str(config.common_get("target_motion.truth_boundary_mode", "open"))


def behaviour_profile(config: ChapterConfig, name: str) -> TargetBehaviorProfile:
    """config가 선언한 거동 프로파일을 만든다."""

    declared = config.common_get(f"target_motion.behaviour_profiles.{name}")
    if declared is None:
        raise KeyError(f"config에 선언되지 않은 거동 프로파일: {name}")
    bounds = declared.get(
        "mode_speed_bounds_kph",
        [[0, 0], [0, 20], [20, 40], [2, 32], [0, 40]],
    )
    return TargetBehaviorProfile(
        name=name,
        target_class="MIXED" if name == "MIXED_GROUND" else "GROUND",
        description=str(declared.get("description", name)),
        mode_probabilities=tuple(
            float(value) for value in declared["mode_probabilities"]
        ),
        persistence_alpha=float(declared.get("persistence_alpha", 0.5)),
        process_noise_scale=float(declared.get("process_noise_scale", 1.0)),
        turn_rate_scale=float(declared.get("turn_rate_scale", 1.0)),
        mode_speed_bounds_kph=tuple(
            (float(low), float(high)) for low, high in bounds
        ),
    )


def target_profile(config: ChapterConfig, key: str) -> TargetOperationalProfile:
    """``"tank"`` / ``"tel"`` -> config가 배정한 거동을 붙인 운용 프로파일.

    ``"both"``는 **계획 프로파일**을 고르는 자리에서는 성립하지 않는다.
    계획기는 어떤 표적이 나올지 모르는 채로 경로를 한 번만 세우므로, 표적
    계층 두 개를 동시에 넣을 수 없다. Ch6의 설계가 정확히 그것이다 —
    **계획은 한 번, 평가는 계층마다**, 그리고 ``target_stratum_weights``로
    통합한다(``chapters/chapter6.py::_weighted_record``).

    그래서 ``"both"``는 config가 선언한 첫 계층으로 해석한다. 평가 대상
    계층은 이 값이 아니라 ``target_strata``가 정한다.

    (이전에는 runner가 통합확인 챕터에 ``"both"``를 넘기는데 여기서 받지
    못해 그 챕터가 그대로 죽었다.)
    """

    name = PROFILE_KEYS.get(key, key)
    if name in {"both", "all"}:
        declared = config.get("target_strata") or []
        name = str(declared[0]) if declared else "Tank-nominal"
    base = _BASE_PROFILES.get(name)
    if base is None:
        raise KeyError(f"알 수 없는 표적 프로파일: {key}")
    assigned = config.common_get(f"target_motion.assignment.{name}")
    if not assigned:
        return base
    return replace(base, imm5_behavior=behaviour_profile(config, assigned))


def truth_motion_spec(
    config: ChapterConfig,
    profile: TargetOperationalProfile,
    mission: MissionConfig,
    *,
    boundary_mode: str | None = None,
    step_s: float | None = None,
) -> TargetMotionSpec:
    """진리 생성용 운동 사양. 기본은 경계를 두지 않는 ``open``."""

    return profile.motion_spec(
        max_speed_mps=mission.target_max_speed_mps,
        step_s=step_s if step_s is not None else motion_step_s(config),
        boundary_mode=boundary_mode or truth_boundary_mode(config),
    )


def generate_truth(
    config: ChapterConfig,
    mission: MissionConfig,
    evaluation: EvaluationConfig,
    profile: TargetOperationalProfile,
    *,
    horizon_s: float,
    terrain=None,
    prior: TargetPrior | None = None,
    step_s: float | None = None,
    coupling: TerrainCoupling | None = None,
    boundary_mode: str | None = None,
) -> list[TargetTrajectory]:
    """config가 선언한 조건으로 진리 표적 앙상블을 만든다."""

    link = coupling or terrain_coupling(config)
    # 초기분포 기본값은 config가 절대 미터로 고정한 cue ring이다.
    # 비율 사전분포(0.65R)를 쓰면 AOI를 키울 때 표적도 같이 밀려나가
    # 포함률 사이징이 자기 자신을 재는 꼴이 된다 (research/aoi.py 참조).
    return generate_moving_targets(
        mission,
        evaluation,
        prior if prior is not None else cue_prior(config, mission.search_radius_m)[0],
        truth_motion_spec(
            config,
            profile,
            mission,
            boundary_mode=boundary_mode,
            step_s=step_s,
        ),
        horizon_s=horizon_s,
        terrain=terrain,
        terrain_bias_strength=link.bias_strength,
        terrain_offroad_probability=link.offroad_probability,
        terrain_halt_probability_boost=link.halt_probability_boost,
    )


def truth_provenance(
    config: ChapterConfig, profile: TargetOperationalProfile
) -> dict:
    """결과 JSON에 "이번 실행이 실제로 쓴 표적 조건"을 그대로 박아둔다.

    config를 바꿔놓고 옛 결과와 비교하는 사고를 막는 유일한 방법은 조건을
    결과 안에 적어두는 것이다. 모든 챕터의 ``condition`` 블록이 이걸 넣는다.
    """

    link = terrain_coupling(config)
    behaviour = profile.imm5_behavior
    return {
        "motion_model": "IMM5",
        "behaviour_profile": behaviour.name,
        "mode_names": list(
            config.common_get(
                "target_motion.mode_names",
                ["HALT", "LOW_CV", "HIGH_CV", "CTRV", "MANEUVER"],
            )
        ),
        "mode_probabilities": list(behaviour.mode_probabilities),
        "mode_speed_bounds_kph": [
            list(bound) for bound in behaviour.mode_speed_bounds_kph
        ],
        "persistence_alpha": behaviour.persistence_alpha,
        # 최대속도가 아니라 이 값이 실제 이동거리를 정한다.
        "ensemble_mean_speed_kph": behaviour.mean_speed_kph,
        "step_s": motion_step_s(config),
        "boundary_mode": truth_boundary_mode(config),
        "terrain_coupling": {
            "bias_strength": link.bias_strength,
            "offroad_probability": link.offroad_probability,
            "halt_probability_boost": link.halt_probability_boost,
        },
    }
