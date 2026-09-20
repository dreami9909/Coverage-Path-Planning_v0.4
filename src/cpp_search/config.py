"""실험 조건 로딩 — 입력값의 단일 출처.

실험 조건을 정하는 값은 전부 ``config/chapter*.json``에 있다. 러너와 챕터
모듈은 절대 조건값을 코드에 박지 않는다. 해석 순서는

    명령행 명시값  >  config 파일  >  코드 기본값       (``ChapterConfig.resolve``)

CLI 인자의 argparse 기본값이 모두 ``None``이라, 플래그를 생략하면 config가
반드시 이긴다. 이 규칙이 깨지면 ``tests/test_config_loader.py``가 실패한다.

* ``load_chapter_config(ch)`` — 해당 챕터 config + 공통조건(``common_experiment.json``)을 함께 로드.
  파일이 둘 이상인 챕터는 병합하고, 같은 키가 충돌하면 ``_conflicts``에
  남긴다(조용히 덮어쓰지 않는다).
* ``ChapterConfig.mission() / .sensor()`` — config 값으로 도메인 객체를 만든다.
  속도는 kph -> m/s (``/ 3.6``) 변환만 한다.
* ``provenance()`` — 결과 JSON에 "어느 config 파일에서 읽었는지"를 남긴다.

의존
----
* 위: ``cpp_search.core.models``(MissionConfig / SensorSpec 생성).
* 아래: ``runner``와 모든 ``chapters/chapter*.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from json import dumps, loads
from pathlib import Path
from typing import Any, Iterable
from warnings import catch_warnings, simplefilter

from cpp_search.core.models import MissionConfig, Point2D, SensorChannelSpec, SensorSpec
from cpp_search.core.motion import GROUND_MOTION_MODE_NAMES, TargetBehaviorProfile
from cpp_search.core.profiles import (
    NOMINAL_TARGET_PROFILES,
    TargetOperationalProfile,
)
from cpp_search.core.search_envelope import CompositeSearchEnvelope
from cpp_search.core.sensor_observation import TANK_NOMINAL, TEL_NOMINAL


CHAPTER_CONFIG_FILES: dict[str, tuple[str, ...]] = {
    "1": ("chapter1_evaluation_contract.json",),
    "2": ("chapter2_stone_spx_terrain.json",),
    "3": ("chapter3_mappo_stone.json",),
    "4": ("chapter4_synthetic_terrain.json",),
}

COMMON_CONFIG_FILE = "common_experiment.json"

#: 결과 JSON의 모양이 바뀌면 올린다. 저장된 결과와 현재 코드가 서로 다른
#: 계약을 따르고 있는지 한 눈에 보게 하려는 값이다.
#: 2 — Ch6 generalisation_verdict 가 significantly_worse_than / claim_status 를
#:     내보내고, provenance 에 코드 지문이 붙었다.
RESULT_SCHEMA_VERSION = 2

#: ``provenance.configuration_snapshot`` 기록 형식의 버전.
#: 1 — 공통설정 + 챕터설정 + 실행옵션 + **챕터 식별자**.
#: 2 — 챕터 식별자를 뺐다. ``--chapter 6`` 과 ``--chapter 6a1`` 은 같은 config
#:     로 같은 실험을 돌리는 별칭인데 식별자 때문에 지문이 갈렸다.
CONFIGURATION_SNAPSHOT_VERSION = 2


def find_config_root(start: Path | None = None) -> Path:
    """Return the repository ``config`` directory."""

    candidates: list[Path] = []
    if start is not None:
        candidates.append(Path(start))
    here = Path(__file__).resolve()
    candidates.extend(here.parents)
    candidates.append(Path.cwd().resolve())
    for candidate in candidates:
        for parent in (candidate, *candidate.parents):
            config_directory = parent / "config"
            if (config_directory / COMMON_CONFIG_FILE).is_file():
                return config_directory
    raise FileNotFoundError(
        "config directory containing chapter0_common_experiment.json was not found"
    )


@dataclass(frozen=True, slots=True)
class ChapterConfig:
    """Read-only view over one chapter's declared experiment condition."""

    chapter: str
    data: dict[str, Any]
    common: dict[str, Any]
    source_paths: tuple[Path, ...]

    def get(self, dotted_path: str, default: Any = None) -> Any:
        return _dig(self.data, dotted_path, default)

    def common_get(self, dotted_path: str, default: Any = None) -> Any:
        return _dig(self.common, dotted_path, default)

    def require(self, dotted_path: str) -> Any:
        value = _dig(self.data, dotted_path, _MISSING)
        if value is _MISSING:
            raise KeyError(
                f"chapter {self.chapter} config is missing required key '{dotted_path}'"
            )
        return value

    def resolve(self, override: Any, dotted_path: str, default: Any) -> Any:
        """Command-line override first, then the config file, then the default."""

        if override is not None:
            return override
        value = _dig(self.data, dotted_path, _MISSING)
        if value is _MISSING:
            value = _dig(self.common, dotted_path, _MISSING)
        return default if value is _MISSING else value

    # -- shared objects built from the common (Chapter 0) condition ---------

    def mission(self, *, uav_count: int | None = None) -> MissionConfig:
        mission = self.common_get("mission", {})
        built = MissionConfig(
            center=Point2D(0.0, 0.0),
            search_radius_m=float(mission.get("search_radius_m", 3_333.3333333333335)),
            uav_count=int(mission.get("uav_count", 6)),
            max_speed_mps=float(mission.get("transit_speed_kph", 160.0)) / 3.6,
            transit_speed_mps=float(mission.get("transit_speed_kph", 160.0)) / 3.6,
            search_speed_mps=float(mission.get("search_speed_kph", 100.0)) / 3.6,
            target_max_speed_mps=float(mission.get("target_max_speed_kph", 40.0)) / 3.6,
        )
        if uav_count is not None:
            built = replace(built, uav_count=uav_count)
        return built

    def sensor(self) -> SensorSpec:
        sensor = self.common_get("sensor", {})
        envelope_data = self.common_get("system_search_envelope", {})
        envelope = CompositeSearchEnvelope.operational(
            support_half_width_m=float(
                envelope_data.get("support_half_width_m", 400.0)
            ),
            seeker_half_width_m=float(
                envelope_data.get("seeker_half_width_m", 300.0)
            ),
            vehicle_weave_half_amplitude_m=float(
                envelope_data.get("vehicle_weave_half_amplitude_m", 100.0)
            ),
            vehicle_weave_wavelength_m=float(
                envelope_data.get("vehicle_weave_wavelength_m", 800.0)
            ),
            calibration_status=str(
                envelope_data.get(
                    "calibration_status",
                    "uncalibrated operational assumption",
                )
            ),
        )
        eo_width, eo_height = sensor.get("eo_resolution", [1_280, 720])
        ir_width, ir_height = sensor.get("ir_resolution", [1_280, 720])
        frame_rate_hz = float(sensor.get("frame_rate_hz", 30.0))
        search_hfov_deg = float(sensor.get("fused_search_hfov_deg", 18.0))
        return SensorSpec(
            altitude_m=float(sensor.get("altitude_m", 600.0)),
            eo=SensorChannelSpec(
                "EO",
                int(eo_width),
                int(eo_height),
                frame_rate_hz,
                search_hfov_deg,
            ),
            ir=SensorChannelSpec(
                "IR",
                int(ir_width),
                int(ir_height),
                frame_rate_hz,
                search_hfov_deg,
            ),
            max_gimbal_angle_deg=float(sensor.get("gimbal_max_angle_deg", 45.0)),
            gimbal_max_rate_dps=float(sensor.get("gimbal_max_rate_dps", 60.0)),
            gimbal_scan_rate_dps=float(sensor.get("gimbal_scan_rate_dps", 30.0)),
            gimbal_settle_time_s=float(sensor.get("gimbal_settle_time_s", 0.1)),
            overlap_ratio=float(sensor.get("overlap_ratio", 0.2)),
            independent_look_interval_s=float(
                sensor.get("independent_look_interval_s", 3.2)
            ),
            search_envelope=envelope,
        )

    def target_profiles(self) -> tuple[TargetOperationalProfile, ...]:
        """Build the configured Tank/TEL profiles used by truth and planning."""

        declared = self.common_get("target_profiles", [])
        if not declared:
            return NOMINAL_TARGET_PROFILES
        signatures = {
            "Tank-nominal": TANK_NOMINAL,
            "TEL-nominal": TEL_NOMINAL,
        }
        profiles: list[TargetOperationalProfile] = []
        for item in declared:
            name = str(item["name"])
            if name not in signatures:
                raise ValueError(f"unknown configured target profile: {name}")
            imm5 = item.get("imm5", {})
            if imm5:
                probabilities = imm5.get("mode_occupancy")
                speed_ranges = imm5.get("mode_speed_ranges_kph")
                if not isinstance(probabilities, dict) or not isinstance(
                    speed_ranges, dict
                ):
                    raise ValueError(
                        f"{name} must configure IMM5 occupancy and speed ranges"
                    )
                mode_probabilities = tuple(
                    float(probabilities[mode]) for mode in GROUND_MOTION_MODE_NAMES
                )
                mode_speed_bounds = tuple(
                    tuple(float(value) for value in speed_ranges[mode])
                    for mode in GROUND_MOTION_MODE_NAMES
                )
                behavior_name = str(imm5.get("name", f"{name.upper()}_IMM5"))
                integration_step_s = float(imm5.get("integration_step_s", 30.0))
                boundary_mode = str(imm5.get("boundary_mode", "escape"))
            else:
                behavior_name = str(
                    self.common_get(f"target_motion.assignment.{name}", "")
                )
                configured = self.common_get(
                    f"target_motion.behaviour_profiles.{behavior_name}",
                    {},
                )
                if not configured:
                    raise ValueError(f"{name} has no configured behavior assignment")
                mode_probabilities = tuple(
                    float(value) for value in configured["mode_probabilities"]
                )
                mode_speed_bounds = tuple(
                    tuple(float(value) for value in pair)
                    for pair in configured["mode_speed_bounds_kph"]
                )
                imm5 = configured
                integration_step_s = float(
                    self.common_get("target_motion.step_s", 30.0)
                )
                boundary_mode = "escape"
            behavior = TargetBehaviorProfile(
                name=behavior_name,
                target_class="GROUND",
                description=str(imm5.get("description", "uncalibrated prior")),
                mode_probabilities=mode_probabilities,
                mode_speed_bounds_kph=mode_speed_bounds,
                persistence_alpha=float(imm5["persistence_alpha"]),
                persistence_reference_s=float(
                    imm5.get("persistence_reference_s", 30.0)
                ),
                process_noise_reference_s=float(
                    imm5.get("process_noise_reference_s", 30.0)
                ),
                process_noise_scale=float(imm5.get("process_noise_scale", 1.0)),
                turn_rate_scale=float(imm5.get("turn_rate_scale", 1.0)),
            )
            profiles.append(
                TargetOperationalProfile(
                    name=name,
                    signature=signatures[name],
                    imm5_behavior=behavior,
                    integration_step_s=integration_step_s,
                    boundary_mode=boundary_mode,
                    calibration_status=str(
                        item.get(
                            "calibration_status",
                            "uncalibrated research prior; track calibration pending",
                        )
                    ),
                )
            )
        return tuple(profiles)

    def target_profile(self, selector: str) -> TargetOperationalProfile:
        normalized = selector.strip().lower()
        aliases = {"tank": "tank-nominal", "tel": "tel-nominal"}
        normalized = aliases.get(normalized, normalized)
        for profile in self.target_profiles():
            if profile.name.lower() == normalized:
                return profile
        raise KeyError(f"configured target profile not found: {selector}")

    @property
    def terrain_halt_probability_boost(self) -> float:
        return float(
            self.common_get(
                "target_motion.terrain_coupling.halt_probability_boost",
                self.common_get("terrain.halt_probability_boost.baseline", 0.0),
            )
        )

    @property
    def mission_time_s(self) -> float:
        return float(self.common_get("mission.mission_time_s", 300.0))

    @property
    def base_seed(self) -> int:
        return int(self.common_get("monte_carlo.base_seed", 20_260_830))

    @property
    def sample_count(self) -> int:
        return int(self.common_get("monte_carlo.sample_count", 1_000))

    @property
    def report_confidence(self) -> float:
        return float(self.common_get("monte_carlo.report_confidence", 0.95))

    @property
    def kpi_names(self) -> tuple[str, ...]:
        return tuple(self.common_get("kpi", ()))

    def provenance(self, *, run_options: dict[str, Any] | None = None) -> dict[str, Any]:
        """결과 파일이 "어느 조건 + **어느 코드**"에서 나왔는지 남긴다.

        config 출처만 남기면 절반이다. 실제로 저장된 결과 중에 지금 코드가
        만들 수 없는 키를 가진 것이 있었고(Ch6의 ``claim_status``), 어느
        버전에서 나온 것인지 되짚을 방법이 없었다. 소스 지문을 같이 남기면
        결과와 코드가 어긋난 것을 바로 알 수 있다.

        run_options를 전달하면 로드된 공통·챕터 설정과 실행 옵션을 함께
        보존한다. 입력 기록이며, 알고리즘 검증 또는 성능 인증을 뜻하지 않는다.

        **스냅샷에는 챕터 식별자를 넣지 않는다** (형식 2). ``--chapter 6``과
        ``--chapter 6a1``은 같은 config 파일로 같은 실험을 돌리는 별칭인데,
        식별자를 해시에 넣으면 결과가 비트 단위로 같은데도 지문이 갈렸다.
        지문은 "어떤 설정으로 돌렸나"를 가리켜야 하고, 그 설정은 공통설정 +
        챕터설정 + 실행옵션이 전부 결정한다. 챕터 이름과 config 파일 목록은
        해시 밖 ``chapter``/``config_files``에 그대로 남는다.

        이 해시는 **소스 코드를 덮지 않는다.** 코드가 바뀌어 결과가 달라져도
        설정이 같으면 값이 그대로다. 코드 동일성은 ``code_fingerprint``로
        따로 확인한다.
        """

        result = {
            "chapter": self.chapter,
            "config_files": [path.name for path in self.source_paths],
            "schema_version": self.get("schema_version", 1),
            **source_provenance(),
        }
        if run_options is not None:
            serialized = dumps(
                {
                    "common_config": self.common,
                    "chapter_config": self.data,
                    "run_options": run_options,
                },
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            result.update(
                configuration_snapshot_version=CONFIGURATION_SNAPSHOT_VERSION,
                configuration_snapshot=loads(serialized),
                configuration_sha256=sha256(serialized.encode("utf-8")).hexdigest(),
            )
        return result


_MISSING = object()


def package_root() -> Path:
    """Return the ``cpp_search`` package directory."""

    return Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def source_fingerprint() -> str:
    """패키지 ``.py`` 전체의 내용 해시. 코드가 바뀌면 값이 바뀐다.

    git 커밋에 기대지 않는다. 이 저장소는 커밋이 없는 상태로도 실행되고,
    커밋이 있어도 작업 트리가 더러우면 커밋 해시가 거짓말을 한다. 파일
    내용을 직접 훑는 편이 "이 결과를 만든 코드"를 정확히 가리킨다.

    **프로세스당 한 번만 계산한다.** 캐시가 없으면 ``--chapter all`` 이
    도는 40분 동안 누가 소스를 고칠 때 앞 챕터와 뒤 챕터의 지문이 갈린다.
    실제로 그렇게 됐다 — 실행 중 주석 한 줄을 고쳤더니 열 챕터가 두 지문으로
    쪼개졌다. 한 번의 실행은 하나의 코드 상태를 가리켜야 한다.
    """

    digest = sha256()
    for path in sorted(package_root().rglob("*.py")):
        digest.update(path.relative_to(package_root()).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def source_provenance() -> dict[str, Any]:
    """결과 JSON에 박을 코드 출처 블록."""

    try:
        fingerprint = source_fingerprint()
    except OSError as error:  # 읽을 수 없으면 침묵하지 말고 이유를 남긴다
        fingerprint = f"unavailable: {type(error).__name__}"
    return {
        "code_fingerprint": fingerprint,
        "package_version": _package_version(),
        "result_schema_version": RESULT_SCHEMA_VERSION,
    }


def _package_version() -> str:
    try:
        # 설치 메타데이터가 있어도 Version 항목이 비어 있을 수 있다(오래된
        # egg-info). 지금은 None 이 오면서 DeprecationWarning 을 내고, 곧
        # KeyError 를 던지도록 바뀐다. 둘 다 막고 경고도 삼킨다 — 여기서
        # 버전을 못 읽는 것은 정상 경로이지 사용자가 볼 문제가 아니다.
        with catch_warnings():
            simplefilter("ignore", DeprecationWarning)
            return version("cpp-search-sim") or "source"
    except (PackageNotFoundError, KeyError):
        # 설치 없이 (`python3 run_research.py`) 실행하는 것이 기본 경로다.
        return "source"


def _dig(data: dict[str, Any], dotted_path: str, default: Any) -> Any:
    node: Any = data
    for key in dotted_path.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def load_chapter_config(
    chapter: str,
    *,
    config_root: Path | None = None,
) -> ChapterConfig:
    """Load one chapter's config plus the shared Chapter 0 condition."""

    if chapter not in CHAPTER_CONFIG_FILES:
        raise KeyError(f"unknown chapter identifier: {chapter}")
    root = config_root or find_config_root()
    common = loads((root / COMMON_CONFIG_FILE).read_text(encoding="utf-8"))
    merged: dict[str, Any] = {}
    paths: list[Path] = []
    for name in CHAPTER_CONFIG_FILES[chapter]:
        path = root / name
        payload = loads(path.read_text(encoding="utf-8"))
        paths.append(path)
        merged = _merge(merged, payload)
    return ChapterConfig(chapter, merged, common, tuple(paths))


def _merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge(merged[key], value)
        elif key in merged and merged[key] != value and key not in {"schema_version", "chapter"}:
            # Two files describe the same chapter (4-b-1); keep both under
            # their own namespace instead of silently dropping one.
            merged.setdefault("_conflicts", {})
            merged["_conflicts"][key] = [merged[key], value]
            merged[key] = value
        else:
            merged[key] = value
    return merged


def derive_seeds(base_seed: int, count: int, *, salt: int = 0) -> tuple[int, ...]:
    """Deterministically derive independent seeds from one base seed."""

    if count <= 0:
        raise ValueError("count must be positive")
    # 괄호가 의미를 정한다. `a + b & c` 는 파이썬 우선순위상 `(a + b) & c` 로
    # 읽히고 그것이 의도이지만, 마스크가 덧셈보다 먼저 걸리는 것으로 잘못
    # 읽기 쉬운 자리라 명시한다.
    return tuple(
        ((base_seed ^ salt) + index * 0x9E37_79B9) & 0x7FFF_FFFF
        for index in range(count)
    )


def as_int_tuple(values: Iterable[Any]) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


def as_float_tuple(values: Iterable[Any]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)
