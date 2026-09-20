"""결과 트리에서 결정론적으로 만들어지는 CSV · PNG 산출물.

그림은 결과보다 **부차적이다**. 그래서 러너는 결과 JSON 을 먼저 디스크에
내리고 그 다음에 이 모듈을 부른다 — 여기서 예외가 나도 몇 시간짜리 계산이
사라지지 않는다. 대신 실패 사실은 결과의 ``artifacts.error`` 에 남고 러너가
종료코드 1 로 끝난다 (조용히 성공으로 넘어가지 않는다).

파일 번호는 **생성 순서대로** 붙는다. 그래서 아래 호출 순서가 곧 그림 순서고,
순서는 읽는 흐름에 맞춘다: 이 챕터가 보이려는 것 -> 통계 비교 -> 신뢰성.

의존
----
* 위: ``research/kpi``, matplotlib(Agg 백엔드).
* 아래: ``research/runner`` 만.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cpp_search.kpi import KPI_KEYS


COLORS = ("#176B87", "#C84B31", "#5A8F29", "#7A5C99", "#D39A2C", "#4B5563")

#: 한글을 그릴 수 있는 폰트 후보. 첫 번째로 설치된 것을 쓴다.
KOREAN_FONT_CANDIDATES = (
    "AppleSDGothicNeo-Regular",
    "Apple SD Gothic Neo",
    "AppleGothic",
    "Noto Sans CJK KR",
    "NanumGothic",
    "Malgun Gothic",
    "Source Han Sans KR",
)


def _configure_korean_font() -> bool:
    """한글 폰트를 찾아 등록한다. 없으면 ``False``.

    matplotlib 기본 폰트(DejaVu Sans)에는 한글 글리프가 없다. 그대로 두면
    제목과 축 라벨이 □ 로 렌더되고, 경고는 stderr 로만 나가서 눈에 띄지
    않는다 — 실제로 그런 그림이 v0.3 에 남아 있었다. 그래서 폰트가 없을 때는
    조용히 □ 를 그리는 대신 **영문 라벨로 떨어진다** (``_text`` 참조).
    """

    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in KOREAN_FONT_CANDIDATES:
        if candidate in available:
            plt.rcParams["font.family"] = candidate
            # 한글 폰트는 마이너스 기호를 제대로 안 그리는 경우가 있다.
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


KOREAN_FONT_AVAILABLE = _configure_korean_font()


def _text(korean: str, english: str) -> str:
    """한글 폰트가 있으면 한글, 없으면 영문. 그림에 □ 가 남지 않게."""

    return korean if KOREAN_FONT_AVAILABLE else english


class _FigureNumbering:
    """``chapter-<n>-01-<slug>.png`` 처럼 생성 순서로 번호를 붙인다."""

    def __init__(self, chapter: str, directory: Path) -> None:
        self.chapter = chapter
        self.directory = directory
        self._index = 0

    def path(self, slug: str, *, suffix: str = "png") -> Path:
        self._index += 1
        return self.directory / (
            f"chapter-{self.chapter}-{self._index:02d}-{slug}.{suffix}"
        )


def write_chapter_artifacts(
    chapter: str,
    result: dict,
    result_path: Path,
) -> dict[str, Any]:
    # 파일명이 이미 ``chapter-<n>-NN-<slug>`` 이고 결과가 ``results/ch<n>/`` 에
    # 놓이므로 하위 ``chapter-<n>/`` 한 단은 같은 말을 세 번 하는 셈이다.
    directory = result_path.parent / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    numbering = _FigureNumbering(chapter, directory)
    files: list[Path] = []
    if chapter == "1":
        files.extend(_chapter1_artifacts(result, numbering))
    elif chapter == "2":
        files.extend(_chapter2_artifacts(result, numbering))
    elif chapter == "3":
        files.extend(_chapter3_artifacts(result, numbering))
    elif chapter == "4":
        files.extend(_chapter4_artifacts(result, numbering))
    stale = _remove_stale_artifacts(directory, files)
    return {
        "directory": str(directory),
        "files": [str(path) for path in files],
        "removed_stale": [str(path) for path in stale],
    }


# ----------------------------------------------------------------- Chapter 1


def _chapter1_artifacts(result: dict, numbering: _FigureNumbering) -> list[Path]:
    files: list[Path] = []
    rows = list(result.get("sweep_width_table", []) or [])
    if rows:
        path = numbering.path("sweep-width-table", suffix="csv")
        _write_csv(path, rows)
        files.append(path)

        fused = [row for row in rows if row.get("channel_mode") == "fused"]
        if fused:
            labels = [str(row.get("target_profile", "?")) for row in fused]
            widths = [float(row.get("sweep_width_m", 0.0)) for row in fused]
            support = result.get("search_envelope", {}).get(
                "system_support_half_width_m"
            )
            figure, axes = plt.subplots(figsize=(6.4, 4.0))
            axes.bar(labels, widths, color=COLORS[0])
            if support:
                # W 를 지지폭과 같은 그림에 둔다. 800 m 지지폭이 W = 1600 m
                # 가 아니라는 것이 이 챕터의 첫 주장이다.
                axes.axhline(
                    2.0 * float(support),
                    color=COLORS[1],
                    linestyle="--",
                    label=f"geometric support {2.0 * float(support):.0f} m",
                )
                axes.legend(fontsize=8)
            axes.set_ylabel("effective sweep width W (m)")
            axes.set_title("W = ∫P_D(x)dx, fused EO/IR channel")
            files.append(_save(figure, numbering.path("sweep-width")))

    feasibility = result.get("coverage_feasibility") or {}
    if feasibility:
        path = numbering.path("coverage-feasibility", suffix="csv")
        _write_csv(path, [_flatten(feasibility)])
        files.append(path)
    return files


# ----------------------------------------------------------------- Chapter 2


def _chapter2_artifacts(result: dict, numbering: _FigureNumbering) -> list[Path]:
    files: list[Path] = []
    grids = list(result.get("grid_conditions", []) or [])
    if not grids:
        return files

    certificate_rows = [
        {
            "grid": block.get("name"),
            "cell_count": (block.get("terrain_weighted") or [{}])[0].get("cell_count"),
            "time_slice_count": (block.get("terrain_weighted") or [{}])[0].get(
                "time_slice_count"
            ),
            **{
                key: (block.get("optimality_certificate") or {}).get(key)
                for key in (
                    "required_relative_gap",
                    "worst_relative_gap",
                    "median_relative_gap",
                    "certified_seed_count",
                    "seed_count",
                    "passed",
                    "mean_runtime_s",
                )
            },
        }
        for block in grids
    ]
    path = numbering.path("optimality-certificate", suffix="csv")
    _write_csv(path, certificate_rows)
    files.append(path)

    # 인증폭 그림. 세로축을 로그로 두는 이유는 인증된 격자의 gap 이 1e-16
    # 수준이고 실패한 격자가 1e-1 수준이라 선형축에서는 전자가 0 으로 뭉개진다.
    gaps = [
        (str(row["grid"]), row.get("worst_relative_gap"))
        for row in certificate_rows
        if row.get("worst_relative_gap") is not None
    ]
    if gaps:
        required = certificate_rows[0].get("required_relative_gap") or 0.01
        figure, axes = plt.subplots(figsize=(6.8, 4.2))
        labels = [name for name, _ in gaps]
        values = [max(float(value), 1e-18) for _, value in gaps]
        colours = [
            COLORS[2] if value <= required else COLORS[1] for value in values
        ]
        axes.bar(labels, values, color=colours)
        axes.axhline(
            required,
            color="black",
            linestyle="--",
            label=f"declared gate {required:.0%}",
        )
        axes.set_yscale("log")
        axes.set_ylabel("worst relative PND gap (log scale)")
        axes.set_title(
            _text(
                "어느 격자까지 최적성이 증명되는가",
                "How far the optimality certificate holds",
            )
        )
        axes.legend(fontsize=8)
        axes.tick_params(axis="x", labelrotation=15)
        files.append(_save(figure, numbering.path("certificate-gap")))

    effects = [
        (str(block.get("name")), block.get("terrain_weighting_effect"))
        for block in grids
        if block.get("terrain_weighting_effect")
    ]
    if effects:
        rows = [
            {"grid": name, **_flatten(effect)} for name, effect in effects
        ]
        path = numbering.path("terrain-weighting-effect", suffix="csv")
        _write_csv(path, rows)
        files.append(path)
        files.append(
            _forest_plot(
                [(name, effect) for name, effect in effects],
                title=_text(
                    "지형 가중 효과 (가중 - 비가중, 최악표적 PD)",
                    "Terrain weighting effect (weighted - unweighted, worst-target PD)",
                ),
                xlabel="paired mean difference",
                path=numbering.path("terrain-weighting-forest"),
            )
        )
    return files


# ----------------------------------------------------------------- Chapter 3


def _chapter3_artifacts(result: dict, numbering: _FigureNumbering) -> list[Path]:
    files: list[Path] = []
    grids = list(result.get("grid_conditions", []) or [])
    if not grids:
        return files

    rows: list[dict] = []
    for block in grids:
        name = str(block.get("name"))
        achievement = block.get("achievement") or {}
        learning = block.get("learning_effect") or {}
        runtime = block.get("runtime") or {}
        rows.append(
            {
                "grid": name,
                "spx_certified_seed_count": len(
                    achievement.get("spx_certified_seeds") or []
                ),
                "spx_worst_relative_gap": achievement.get("spx_worst_relative_gap"),
                "achievement_ratio_mean": achievement.get("mean"),
                "achievement_ratio_min": achievement.get("min"),
                "achievement_ratio_max": achievement.get("max"),
                "learning_effect_mean": learning.get("mean"),
                "spx_mean_runtime_s": runtime.get("spx_mean_s"),
                "mappo_mean_runtime_s": runtime.get("mappo_mean_s"),
            }
        )
    path = numbering.path("spx-vs-mappo", suffix="csv")
    _write_csv(path, rows)
    files.append(path)

    # 같은 인스턴스에서 두 계획법의 최악표적 PD. seed 를 점으로 찍어 짝을 보인다.
    figure, axes = plt.subplots(figsize=(6.8, 4.4))
    for index, block in enumerate(grids):
        spx = {row["seed"]: row for row in block.get("spx", [])}
        mappo = {row["seed"]: row for row in block.get("mappo", [])}
        seeds = sorted(set(spx) & set(mappo))
        if not seeds:
            continue
        x = [spx[seed]["worst_target_detection_probability"] for seed in seeds]
        y = [mappo[seed]["worst_target_detection_probability"] for seed in seeds]
        certified = all(spx[seed]["certified_within_tolerance"] for seed in seeds)
        axes.scatter(
            x,
            y,
            color=COLORS[index % len(COLORS)],
            marker="o" if certified else "x",
            label=(
                f"{block.get('name')}"
                + (" (certified)" if certified else " (certificate open)")
            ),
        )
    limits = axes.get_xlim() + axes.get_ylim()
    low, high = min(limits), max(limits)
    axes.plot([low, high], [low, high], color="black", linewidth=0.8, linestyle=":")
    axes.set_xlabel("Stone SPX worst-target PD")
    axes.set_ylabel("MAPPO worst-target PD")
    axes.set_title(
        _text("같은 인스턴스, 다른 해법", "Same instance, different solvers")
    )
    axes.legend(fontsize=8)
    files.append(_save(figure, numbering.path("spx-vs-mappo-scatter")))

    paired = [
        (str(block.get("name")), block.get("paired_spx_minus_mappo"))
        for block in grids
        if block.get("paired_spx_minus_mappo")
    ]
    if paired:
        files.append(
            _forest_plot(
                paired,
                title=_text(
                    "SPX - MAPPO (최악표적 PD, 짝지은 계획 seed)",
                    "SPX - MAPPO (worst-target PD, paired planning seeds)",
                ),
                xlabel="paired mean difference",
                path=numbering.path("spx-minus-mappo-forest"),
            )
        )

    # 학습곡선. config 가 선언한 격자에만 담겨 있다.
    for block in grids:
        curves = [
            (row["seed"], row["learning_curve"])
            for row in block.get("mappo", [])
            if row.get("learning_curve")
        ]
        if not curves:
            continue
        figure, axes = plt.subplots(figsize=(6.8, 4.2))
        for index, (seed, curve) in enumerate(curves):
            iterations = [entry["iteration"] for entry in curve]
            values = [
                entry["worst_target_detection_probability"] for entry in curve
            ]
            axes.plot(
                iterations,
                values,
                color=COLORS[index % len(COLORS)],
                linewidth=1.2,
                label=f"seed {seed}",
            )
        spx_values = [
            row["worst_target_detection_probability"] for row in block.get("spx", [])
        ]
        if spx_values:
            axes.axhline(
                float(np.mean(spx_values)),
                color="black",
                linestyle="--",
                label="SPX mean (same instances)",
            )
        axes.set_xlabel("MAPPO update iteration")
        axes.set_ylabel("worst-target PD (deterministic evaluation)")
        axes.set_title(
            f"{_text('학습곡선', 'Learning curve')} — {block.get('name')}"
        )
        axes.legend(fontsize=8)
        files.append(
            _save(
                figure,
                numbering.path(f"learning-curve-{_slug(str(block.get('name')))}"),
            )
        )
    return files


# ----------------------------------------------------------------- Chapter 4


def _chapter4_artifacts(result: dict, numbering: _FigureNumbering) -> list[Path]:
    files: list[Path] = []
    families = list(result.get("terrain_families", []) or [])
    if not families:
        return files

    rows: list[dict] = []
    for family in families:
        for planner, block in (family.get("kpi") or {}).items():
            combined = block.get("combined") or {}
            row: dict = {"family": family.get("family"), "planner": planner}
            for key in KPI_KEYS:
                entry = combined.get(key)
                if isinstance(entry, dict):
                    row[key] = entry.get("mean")
                    row[f"{key}_ci_low"] = (entry.get("confidence_interval") or [None, None])[0]
                    row[f"{key}_ci_high"] = (entry.get("confidence_interval") or [None, None])[1]
            row["model_minus_flown_mean"] = block.get("model_minus_flown_mean")
            rows.append(row)
    path = numbering.path("flown-kpi", suffix="csv")
    _write_csv(path, rows)
    files.append(path)

    # 실비행 탐지확률: 지형 계열 x 계획법. CI 를 오차막대로.
    planners = sorted({str(row["planner"]) for row in rows})
    family_names = [str(family.get("family")) for family in families]
    figure, axes = plt.subplots(figsize=(7.2, 4.4))
    width = 0.8 / max(len(planners), 1)
    for index, planner in enumerate(planners):
        values: list[float] = []
        # yerr 은 **두 리스트의 리스트**로 넘긴다. ``(2, N)`` ndarray 로 넘기면
        # 막대가 하나일 때 matplotlib 이 그 배열을 스칼라로 변환하려 하면서
        # NumPy 1.25 DeprecationWarning 이 뜬다 (장래에는 오류가 된다).
        lower: list[float] = []
        upper: list[float] = []
        for name in family_names:
            match = next(
                (
                    row
                    for row in rows
                    if row["family"] == name and row["planner"] == planner
                ),
                {},
            )
            mean = match.get("detection_probability_within_limit")
            low = match.get("detection_probability_within_limit_ci_low")
            high = match.get("detection_probability_within_limit_ci_high")
            values.append(float(mean) if mean is not None else 0.0)
            lower.append(
                (float(mean) - float(low)) if None not in (mean, low) else 0.0
            )
            upper.append(
                (float(high) - float(mean)) if None not in (mean, high) else 0.0
            )
        offsets = (np.arange(len(family_names)) + index * width).tolist()
        axes.bar(
            offsets,
            values,
            width=width,
            color=COLORS[index % len(COLORS)],
            label=planner,
            yerr=[lower, upper],
            capsize=3,
        )
    axes.set_xticks(np.arange(len(family_names)) + 0.4 - width / 2)
    axes.set_xticklabels(family_names)
    axes.set_ylabel(_text("실비행 탐지율 (비행 전체)", "flown detection rate (whole flight)"))
    axes.set_title(
        _text(
            "합성지형 실비행 탐지율 (95% bootstrap CI)",
            "Flown results on synthetic terrain (95% bootstrap CI)",
        )
    )
    axes.legend(fontsize=8)
    files.append(_save(figure, numbering.path("flown-detection-probability")))

    # 평균탐지시간 — 탐지율과 짝으로 읽어야 하는 두 번째 헤드라인.
    figure, axes = plt.subplots(figsize=(7.2, 4.4))
    for index, planner in enumerate(planners):
        values, lower, upper = [], [], []
        for name in family_names:
            match = next((row for row in rows if row["family"] == name and row["planner"] == planner), {})
            mean = match.get("conditional_mean_detection_time_s")
            low = match.get("conditional_mean_detection_time_s_ci_low")
            high = match.get("conditional_mean_detection_time_s_ci_high")
            values.append(float(mean) if mean is not None else 0.0)
            lower.append((float(mean) - float(low)) if None not in (mean, low) else 0.0)
            upper.append((float(high) - float(mean)) if None not in (mean, high) else 0.0)
        offsets = (np.arange(len(family_names)) + index * width).tolist()
        axes.bar(offsets, values, width=width, color=COLORS[index % len(COLORS)], label=planner, yerr=[lower, upper], capsize=3)
    axes.set_xticks(np.arange(len(family_names)) + 0.4 - width / 2)
    axes.set_xticklabels(family_names)
    axes.set_ylabel(_text("평균 탐지시간 (탐지된 표적, s)", "mean detection time (detected targets, s)"))
    axes.set_title(_text("실비행 평균 탐지시간 (95% bootstrap CI)", "Flown mean detection time (95% bootstrap CI)"))
    axes.legend(fontsize=8)
    files.append(_save(figure, numbering.path("flown-mean-detection-time")))

    # 계획모형이 실제를 얼마나 과대평가하는가.
    figure, axes = plt.subplots(figsize=(6.8, 4.0))
    for index, planner in enumerate(planners):
        values = [
            next(
                (
                    row.get("model_minus_flown_mean")
                    for row in rows
                    if row["family"] == name and row["planner"] == planner
                ),
                None,
            )
            for name in family_names
        ]
        axes.plot(
            family_names,
            [float(value) if value is not None else np.nan for value in values],
            marker="o",
            color=COLORS[index % len(COLORS)],
            label=planner,
        )
    axes.axhline(0.0, color="black", linewidth=0.8)
    axes.set_ylabel("model PD - flown PD")
    axes.set_title(
        _text(
            "계획모형 - 실비행 차이",
            "Planning model minus flown detection probability",
        )
    )
    axes.legend(fontsize=8)
    files.append(_save(figure, numbering.path("model-minus-flown")))

    paired: list[tuple[str, dict]] = []
    for family in families:
        for metric, block in (family.get("paired_spx_minus_mappo") or {}).items():
            paired.append((f"{family.get('family')} / {metric}", block))
    if paired:
        detection = [item for item in paired if "detection_probability" in item[0]]
        if detection:
            files.append(
                _forest_plot(
                    detection,
                    title=_text(
                        "SPX - MAPPO 실비행 탐지확률 (짝지은 seed)",
                        "SPX - MAPPO flown detection probability (paired seeds)",
                    ),
                    xlabel="paired mean difference",
                    path=numbering.path("flown-paired-forest"),
                )
            )
        path = numbering.path("paired-comparisons", suffix="csv")
        _write_csv(path, [{"comparison": name, **_flatten(block)} for name, block in paired])
        files.append(path)
    return files


# -------------------------------------------------------------------- helpers


def _forest_plot(
    entries: list[tuple[str, dict]],
    *,
    title: str,
    xlabel: str,
    path: Path,
) -> Path:
    """짝지은 차이와 신뢰구간. 부호 방향은 결과가 문자열로 들고 있다."""

    figure, axes = plt.subplots(figsize=(7.2, 0.7 * len(entries) + 2.0))
    for index, (name, block) in enumerate(entries):
        mean = block.get("mean_difference")
        interval = block.get("confidence_interval") or [None, None]
        if mean is None:
            continue
        low = interval[0] if interval[0] is not None else mean
        high = interval[1] if interval[1] is not None else mean
        significant = bool(block.get("significant"))
        axes.errorbar(
            [float(mean)],
            [index],
            xerr=[[float(mean) - float(low)], [float(high) - float(mean)]],
            fmt="o",
            color=COLORS[2] if significant else COLORS[5],
            capsize=4,
        )
    axes.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes.set_yticks(range(len(entries)))
    axes.set_yticklabels([name for name, _ in entries], fontsize=8)
    axes.set_xlabel(xlabel)
    axes.set_title(title)
    direction = next(
        (
            block.get("mean_difference_is")
            for _, block in entries
            if block.get("mean_difference_is")
        ),
        None,
    )
    if direction:
        # 부호 방향을 그림에 적는다. v0.3 에서 이 문자열이 없어서 핵심 그림
        # 하나가 부호를 반대로 읽혔다.
        axes.annotate(
            f"{_text('차이', 'difference')} = {direction}",
            xy=(0.01, 0.01),
            xycoords="axes fraction",
            fontsize=7,
            color="#4B5563",
        )
    return _save(figure, path)


def _save(figure, path: Path) -> Path:
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _flatten(block: dict, prefix: str = "") -> dict:
    """CSV 한 행으로 만들기 위해 한 단계 평탄화한다."""

    flat: dict = {}
    for key, value in block.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, prefix=f"{name}_"))
        elif isinstance(value, (list, tuple)):
            flat[name] = ";".join(str(item) for item in value)
        else:
            flat[name] = value
    return flat


def _slug(text: str) -> str:
    return "".join(
        character if character.isalnum() else "-" for character in text.lower()
    ).strip("-")


def _remove_stale_artifacts(directory: Path, files: list[Path]) -> list[Path]:
    """이번 실행이 만들지 않은 옛 산출물을 지운다.

    챕터 구성이 바뀌면 옛 그림이 남아 새 결과처럼 읽힌다. 실제로 v0.3 에서
    재배치 이전 그림이 섞여 있었다.
    """

    kept = {path.name for path in files}
    removed: list[Path] = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.name not in kept:
            path.unlink()
            removed.append(path)
    return removed
