"""Chapter 1-4 연구 스택의 명령행 진입점.

러너가 하는 일은 정확히 셋이다 — 명령행 해석, 챕터 config 로딩, 결과 저장.
실험 논리는 전부 :mod:`cpp_search.chapters` 에 있다.

모든 값의 해석 순서는 ``명령행 > config 파일 > 코드 기본값`` 이다. CLI 인자의
argparse 기본값이 모두 ``None`` 이라 플래그를 생략하면 config 가 반드시 이긴다.
즉 플래그 없이 실행하면 config 에 적힌 조건이 그대로 재현된다.
"""

from __future__ import annotations

import argparse
import json
from math import isfinite
from pathlib import Path
from time import perf_counter

import numpy as np

from cpp_search.config import load_chapter_config
from cpp_search.chapters import CHAPTER_MODULES
from cpp_search.kpi import DIAGNOSTIC_KPI_KEYS, KPI_KEYS
from cpp_search.options import RunOptions


CHAPTERS = ("1", "2", "3", "4", "all")

ALL_CHAPTERS = tuple(chapter for chapter in CHAPTERS if chapter != "all")

#: ``--seed-count`` 가 챕터별로 **실제로 읽히는** 키에만 쓰이도록 선언한다.
#:
#: 그 챕터가 읽지도 않는 값을 ``config.data`` 에 넣으면 그대로
#: ``configuration_snapshot`` 에 실려 지문을 바꾼다. KPI 가 비트 단위로 같은데
#: 쓰이지 않은 seed 한도 때문에 ``configuration_sha256`` 이 갈리는 사고가
#: v0.3 에서 실제로 있었다.
#:
#: Chapter 1 은 seed 를 쓰지 않으므로 빈 튜플이 "적용되지 않는다"는 선언이다.
SEED_LIMIT_KEYS: dict[str, tuple[str, ...]] = {
    "1": (),
    "2": ("stone_spx.planning_seed_limit",),
    "3": ("stone_spx.planning_seed_limit",),
    "4": ("stone_spx.evaluation_seed_limit",),
}


def main(argv: list[str] | None = None) -> None:
    arguments = _parse_arguments(argv)
    if arguments.chapter == "all":
        _run_all(arguments)
        return
    output_path = _run_one(arguments.chapter, arguments)
    print(output_path.resolve())
    # 그림/CSV 실패는 결과 JSON 을 잃지 않으려고 삼키지만, 실행 자체를
    # 성공으로 보고하면 안 된다. Ch8 의 figure NameError 가 종료코드 0 으로
    # 넘어가면서 산출물이 빠진 채로 "정상 종료"했었다.
    error = _artifact_error(json.loads(output_path.read_text(encoding="utf-8")))
    if error:
        raise SystemExit(
            f"chapter {arguments.chapter}: 산출물 생성 실패 — {error}"
        )


def _run_all(arguments: argparse.Namespace) -> None:
    """Run every chapter in order and write a combined headline summary."""

    directory = (arguments.output or Path("results")).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    headlines: dict[str, dict] = {}
    failures: list[str] = []
    artifact_failures: dict[str, str] = {}
    for chapter in ALL_CHAPTERS:
        started = perf_counter()
        # 한 챕터가 죽어도 나머지 챕터와 이미 끝난 결과를 잃지 않는다.
        # 실패 사실은 summary.json 에 남기고, 조용히 넘어가지 않는다.
        try:
            path = _run_one(
                chapter,
                arguments,
                output_path=directory / f"chapter-{chapter}-result.json",
            )
        except Exception as error:  # noqa: BLE001 - 챕터 하나의 실패를 격리한다
            elapsed = perf_counter() - started
            message = f"{type(error).__name__}: {error}"
            print(f"\n[Chapter {chapter}]  실패 after {elapsed:.1f}s — {message}")
            headlines[chapter] = {
                "runtime_s": round(elapsed, 1),
                "status": "failed",
                "error": message,
            }
            failures.append(chapter)
            continue
        elapsed = perf_counter() - started
        result = json.loads(path.read_text(encoding="utf-8"))
        artifact_error = _artifact_error(result)
        if artifact_error:
            artifact_failures[chapter] = artifact_error
            print(f"\n[Chapter {chapter}]  산출물 생성 실패 — {artifact_error}")
        headlines[chapter] = {
            "runtime_s": round(elapsed, 1),
            "result_file": path.name,
            "status": "artifacts-failed" if artifact_error else "ok",
            "code_fingerprint": (
                result.get("provenance", {}).get("code_fingerprint")
            ),
            **({"artifact_error": artifact_error} if artifact_error else {}),
            **_headline(chapter, result),
        }
    run_block = _run_block(headlines, failures, artifact_failures)
    headlines["_run"] = run_block
    summary_path = directory / "summary.json"
    summary_path.write_text(
        json.dumps(headlines, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _print_summary(headlines)
    print(summary_path)
    if run_block["fingerprint_conflict"]:
        print(
            "\n[경고] 한 실행 안에서 코드 지문이 갈렸다: "
            + ", ".join(
                f"{fingerprint}({', '.join(chapters)})"
                for fingerprint, chapters in sorted(
                    run_block["fingerprints"].items()
                )
            )
            + "\n지문이 다른 결과끼리 비교하면 안 된다 — 실행 중 소스가 바뀌었는지"
            " 확인하고 전 챕터를 다시 돌린다."
        )
    if failures:
        print(f"\n실패한 챕터: {', '.join(failures)} — summary.json 의 error 를 확인한다.")
    if artifact_failures:
        print(
            "산출물 생성 실패: "
            f"{', '.join(sorted(artifact_failures))} — summary.json 의 "
            "artifact_error 를 확인한다."
        )
    if failures or artifact_failures or run_block["fingerprint_conflict"]:
        raise SystemExit(1)


def _artifact_error(result: dict) -> str | None:
    """그림/CSV 생성이 실패했으면 그 이유, 아니면 ``None``."""

    artifacts = result.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    error = artifacts.get("error")
    return str(error) if error else None


def _run_block(
    headlines: dict[str, dict],
    failures: list[str],
    artifact_failures: dict[str, str],
) -> dict:
    """실행 전체에 대한 사실. 챕터 키와 섞이지 않도록 ``_run`` 아래 둔다.

    한 번의 ``--chapter all`` 은 **하나의 코드 상태**를 가리켜야 한다.
    저장된 결과 묶음이 서로 다른 지문 여섯 개로 쪼개져 있었는데, 그 사실이
    각 결과의 ``provenance`` 에 이미 적혀 있었는데도 아무도 대조하지 않아
    드러나지 않았다. 여기서 한 번 모아 본다.
    """

    fingerprints: dict[str, list[str]] = {}
    for chapter, entry in headlines.items():
        fingerprint = entry.get("code_fingerprint")
        if fingerprint:
            fingerprints.setdefault(str(fingerprint), []).append(chapter)
    return {
        "code_fingerprint": (
            next(iter(fingerprints)) if len(fingerprints) == 1 else None
        ),
        "fingerprints": {key: sorted(value) for key, value in fingerprints.items()},
        "fingerprint_conflict": len(fingerprints) > 1,
        "chapter_failures": sorted(failures),
        "artifact_failures": dict(sorted(artifact_failures.items())),
    }


def _run_one(
    chapter: str,
    arguments: argparse.Namespace,
    *,
    output_path: Path | None = None,
) -> Path:
    config = load_chapter_config(chapter)
    if arguments.seed_count is not None:
        _apply_seed_limit(config, chapter, arguments.seed_count)

    options = RunOptions(
        sample_count=int(
            config.resolve(arguments.samples, "monte_carlo.sample_count", 200)
        ),
        particle_count=int(
            config.resolve(arguments.particles, "particle_count", 1_200)
        ),
        episode_count=int(
            config.resolve(arguments.episodes, "mappo.iterations", 40)
        ),
        target_profile=arguments.profile or "both",
        seed=int(config.resolve(arguments.seed, "monte_carlo.base_seed", 20_260_830)),
        map_error=bool(arguments.map_error),
        communication_loss_probability=float(
            arguments.communication_loss
            if arguments.communication_loss is not None
            else 0.0
        ),
        communication_latency_slices=int(
            arguments.communication_latency
            if arguments.communication_latency is not None
            else 0
        ),
        mission_time_s=float(
            config.resolve(arguments.mission_time, "mission.mission_time_s", 300.0)
        ),
    )

    result = CHAPTER_MODULES[chapter].run(config, options)
    destination = output_path or arguments.output or Path(
        f"results/ch{chapter}/chapter-{chapter}-result.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    serializable = _json_safe(result)

    # 결과를 **먼저** 저장한다. 예전에는 그림을 다 그린 뒤에야 JSON을 썼는데,
    # 그러면 figure 단계의 실패(예: matplotlib 미설치) 하나로 Ch6 20 seed x
    # 1,000 표본처럼 몇 시간짜리 계산이 통째로 사라졌다. 계산 결과는 그림보다
    # 비싸므로 먼저 디스크에 내린다.
    _write_result(destination, serializable)
    serializable["artifacts"] = _write_artifacts(chapter, serializable, destination)
    _write_result(destination, serializable)
    return destination


def _write_result(destination: Path, payload: dict) -> None:
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_artifacts(chapter: str, serializable: dict, destination: Path) -> dict:
    """그림·CSV를 만든다. 실패해도 이미 저장된 결과를 잃지 않는다."""

    try:
        from cpp_search.figures import write_chapter_artifacts

        return write_chapter_artifacts(chapter, serializable, destination)
    except Exception as error:  # noqa: BLE001 - 산출물은 결과보다 부차적이다
        print(
            f"  [warn] chapter {chapter}: 그림/CSV 생성 실패 "
            f"({type(error).__name__}: {error}). 결과 JSON은 저장되었다."
        )
        return {"error": f"{type(error).__name__}: {error}", "files": []}


def seed_limit_keys(chapter: str, config) -> tuple[str, ...]:
    """이 챕터에서 ``--seed-count`` 가 실제로 바꾸는 config 키."""

    return SEED_LIMIT_KEYS.get(chapter, ())


def _apply_seed_limit(config, chapter: str, seed_count: int) -> tuple[str, ...]:
    """선언된 키만 덮어쓴다. 안 읽히는 키는 지문만 흐리므로 건드리지 않는다."""

    if seed_count <= 0:
        raise ValueError("seed_count must be positive")
    keys = seed_limit_keys(chapter, config)
    for key in keys:
        head, _, tail = key.partition(".")
        if not tail:
            config.data[key] = seed_count
            continue
        branch = config.data.get(head)
        if isinstance(branch, dict):
            branch[tail] = seed_count
    if not keys:
        print(
            f"  [note] chapter {chapter}: --seed-count 는 이 챕터에 적용되지 "
            "않는다 (seed 목록을 config 가 직접 선언한다)."
        )
    return keys


def collect_kpi(result: dict) -> dict[str, dict]:
    """결과 트리를 훑어 ``kpi`` 블록을 가진 조건을 전부 모은다.

    챕터마다 조건이 놓인 자리가 다르지만(``allocation_strategies`` /
    ``conditions`` / ``methods`` / ``common_kpi`` ...), 모든 조건이 같은 모양의
    ``kpi`` 블록을 들고 있으므로 이 함수 하나로 전 챕터를 비교할 수 있다.
    """

    found: dict[str, dict] = {}

    def walk(node, path: str) -> None:
        if isinstance(node, dict):
            if isinstance(node.get("kpi"), dict):
                found[path or "(root)"] = node["kpi"]
            for key, value in node.items():
                if key == "kpi":
                    continue
                walk(value, f"{path}.{key}" if path else str(key))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")

    walk(result, "")
    return {_leaf(name): block for name, block in found.items()}


def _leaf(path: str) -> str:
    return path.rsplit(".", 1)[-1] if "." in path else path


def _scalar(value):
    """seed 요약 딕셔너리는 평균으로, 그 외는 그대로."""

    if isinstance(value, dict):
        return value.get("mean")
    return value


def _headline(chapter: str, result: dict) -> dict:
    """챕터별 요약: 조건마다 공통 KPI 다섯 개 + 그 챕터의 결론."""

    # Ch6는 seed 요약이라 값이 {mean, min, max} 딕셔너리다. 요약표에서는
    # 평균만 뽑아 다른 챕터와 같은 모양으로 맞춘다.
    conditions = {
        name: {
            key: _scalar(block[key])
            for key in KPI_KEYS + DIAGNOSTIC_KPI_KEYS
            if key in block
        }
        for name, block in collect_kpi(result).items()
    }
    headline: dict = {"conditions": conditions}
    if conditions:
        best = max(
            conditions.items(),
            key=lambda item: item[1].get("detection_probability_within_limit", -1.0),
        )
        headline["best_by_detection_probability"] = best[0]
        fastest = min(
            conditions.items(),
            key=lambda item: item[1].get(
                "conditional_mean_detection_time_s", float("inf")
            ),
        )
        headline["best_by_mean_detection_time"] = fastest[0]

    for key in (
        "best_strategy",
        "best_condition",
        "best_method",
        "selected_cooperation_structure",
        "cooperation_gain",
        "temporal_gain",
        "terrain_component_effect_vs_adapted",
        "generalisation_verdict",
        "terrain_ranking_by_simulated_detection_probability",
    ):
        if key in result:
            headline[key] = result[key]
    if chapter == "1":
        rows = result.get("sweep_width_table", [])
        fused = [row for row in rows if row.get("channel_mode") == "fused"]
        headline["sweep_width_rows"] = len(rows)
        headline["fused_sweep_width_m"] = (
            round(fused[0]["sweep_width_m"], 1) if fused else None
        )
        headline["coverage_ceiling"] = (result.get("coverage_feasibility") or {}).get(
            "effective_coverage_ceiling"
        )
    if chapter in {"2", "3"}:
        # 격자별 한 줄. 이 챕터들의 결론은 KPI 표가 아니라 "어느 격자에서
        # 최적성이 증명됐고, 학습 정책이 거기에 얼마나 붙었는가" 다.
        grids: dict[str, dict] = {}
        for block in result.get("grid_conditions", []) or []:
            row: dict = {}
            certificate = block.get("optimality_certificate")
            if certificate:
                row["certified"] = certificate.get("passed")
                row["worst_relative_gap"] = certificate.get("worst_relative_gap")
            effect = block.get("terrain_weighting_effect")
            if effect:
                row["terrain_weighting_mean_difference"] = effect.get(
                    "mean_difference"
                )
                row["terrain_weighting_significant"] = effect.get("significant")
            achievement = block.get("achievement")
            if achievement:
                row["mappo_achievement_ratio_mean"] = achievement.get("mean")
            learning = block.get("learning_effect")
            if learning:
                row["mappo_learning_effect_mean"] = learning.get("mean")
            if row:
                grids[str(block.get("name", "grid"))] = row
        if grids:
            headline["grids"] = grids
        if "certificate_gate" in result:
            headline["certificate_gate_passed"] = result["certificate_gate"].get(
                "passed"
            )
    if chapter == "4":
        families: dict[str, dict] = {}
        for family in result.get("terrain_families", []) or []:
            paired = family.get("paired_spx_minus_mappo", {}) or {}
            detection = paired.get("detection_probability_within_limit", {}) or {}
            families[str(family.get("family", "family"))] = {
                "spx_minus_mappo_detection_rate": detection.get("mean_difference"),
                "spx_minus_mappo_mean_detection_time_s": (
                    paired.get("conditional_mean_detection_time_s", {}) or {}
                ).get("mean_difference"),
                "significant": detection.get("significant"),
            }
        if families:
            headline["terrain_families"] = families
    return headline


_COLUMNS = (
    # 헤드라인 두 개: 비행 전체(8분)에서 탐지된 비율, 그리고 탐지된 것들의
    # 평균 탐지시간. 2026-09-13 부터 별도의 5분 절단은 없다 — 한계시간이 곧
    # 비행 종료다.
    ("탐지율", "detection_probability_within_limit", "{:>9.3f}"),
    ("평균탐지시간", "conditional_mean_detection_time_s", "{:>13.0f}"),
    ("면적탐색률", "unique_area_coverage_ratio", "{:>10.3f}"),
    ("이탈률", "escape_rate", "{:>8.3f}"),
    ("면적당탐지", "detection_probability_per_covered_area", "{:>11.2f}"),
    # 절단시간(미탐지를 비행종료로 절단한 평균)은 순위지표가 아니라 진단이다.
    ("절단시간", "restricted_mean_detection_time_s", "{:>9.0f}"),
)


def _print_summary(headlines: dict[str, dict]) -> None:
    for chapter, entry in headlines.items():
        # ``_`` 로 시작하는 키는 챕터가 아니라 실행 전체에 대한 블록이다.
        if chapter.startswith("_"):
            continue
        conditions = entry.get("conditions") or {}
        print()
        print(f"[Chapter {chapter}]  {entry['runtime_s']:.1f}s")
        if entry.get("status") == "failed":
            print(f"  실패 — {entry.get('error')}")
            continue
        if entry.get("status") == "artifacts-failed":
            print(f"  산출물 실패 — {entry.get('artifact_error')}")
        if not conditions:
            # Chapter 1-3 은 실비행 KPI 를 내지 않는다. 1 은 조건 선언이고,
            # 2-3 은 계획모형 안에서의 값과 최적성 인증이다. 실비행 KPI 는
            # Chapter 4 에만 있다.
            print(
                "  (실비행 KPI 없음 — 조건 선언 또는 계획모형 안의 값. "
                "격자별 요약은 summary.json 의 grids 를 본다)"
            )
            continue
        header = f"  {'조건':<26}" + "".join(
            f"{label:>11}" for label, _, _ in _COLUMNS
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for name, block in conditions.items():
            cells = []
            for _, key, form in _COLUMNS:
                value = block.get(key)
                cells.append(
                    form.format(value) if isinstance(value, (int, float)) else f"{'-':>10}"
                )
            print(f"  {name:<26}" + "".join(cells))
    print()


def _parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Chapter 1-4 research stack"
    )
    parser.add_argument(
        "--chapter",
        choices=CHAPTERS,
        default="1",
        help="chapter id, or 'all' to run every chapter and write summary.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="result file; with --chapter all this is the output directory",
    )
    parser.add_argument(
        "--samples",
        type=int,
        help="Monte Carlo target samples (default: config monte_carlo.sample_count)",
    )
    parser.add_argument(
        "--particles",
        type=int,
        help="particle count (default: config particle_count)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        help="MAPPO training iterations (default: config mappo.iterations)",
    )
    parser.add_argument("--profile", choices=("tank", "tel", "both"))
    parser.add_argument(
        "--seed",
        type=int,
        help="base seed (default: config monte_carlo.base_seed)",
    )
    parser.add_argument(
        "--mission-time",
        type=float,
        help="mission time in seconds (default: config mission.mission_time_s)",
    )
    parser.add_argument(
        "--seed-count",
        type=int,
        help="use the first N independent evaluation/confirmatory seeds",
    )
    parser.add_argument("--map-error", action="store_true")
    parser.add_argument("--communication-loss", type=float)
    parser.add_argument("--communication-latency", type=int)
    return parser.parse_args(argv)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if isfinite(number) else None
    return value
