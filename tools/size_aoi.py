"""표적 운동 모델을 측정해서 AOI 반경을 정한다.

    python3 tools/size_aoi.py                 # 전부 (권장, ~2분)
    python3 tools/size_aoi.py --only aoi      # 포함률 기반 AOI만
    python3 tools/size_aoi.py --only step     # step_s 민감도만
    python3 tools/size_aoi.py --only halt     # halt boost 민감도만
    python3 tools/size_aoi.py --apply         # 결과를 config에 기록

``--apply`` 는 ``config/common_experiment.json`` 의
``mission.search_radius_m`` 를 포함률 반경으로 바꾸고, 측정 근거를
``aoi.measured`` 에 남긴다. 근거 없이 숫자만 바뀌는 일이 없도록 항상 같이 쓴다.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from cpp_search.aoi import (  # noqa: E402
    containment_radius_m,
    halt_boost_sweep,
    step_sensitivity,
)
from cpp_search.config import load_chapter_config  # noqa: E402

CONFIG_PATH = pathlib.Path("config/common_experiment.json")


def _fmt(block: dict) -> str:
    return "  ".join(
        f"{key}={block[key]:8.1f}" for key in ("mean", "p90", "p95", "p99", "max")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["aoi", "step", "halt"], default=None)
    parser.add_argument("--sample-count", type=int, default=4000)
    parser.add_argument("--horizon-s", type=float, default=480.0)
    parser.add_argument("--rule", choices=["center", "net-displacement"], default="net-displacement",
                        help="center: AOI 중심까지 거리의 포함률 / net-displacement: TP 기준 순변위 포함률(기본)")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", type=pathlib.Path, default=None)
    args = parser.parse_args()

    config = load_chapter_config("1")
    mission = config.mission()
    shared = {"sample_count": args.sample_count, "horizon_s": args.horizon_s}
    report: dict = {}

    if args.only in (None, "aoi") and args.rule == "net-displacement":
        from cpp_search.aoi import net_displacement_containment_radius_m
        nd = net_displacement_containment_radius_m(config, mission, **shared)
        report["aoi_net_displacement"] = nd
        print("\n== AOI 사이징 (TP 기준 순변위 포함률) ==")
        for key, radius in nd["per_profile_radius_m"].items():
            print(f"  {key:5s} 순변위 {nd['containment']:.0%} : {radius:8.0f} m")
        print(f"  필요 반경 (최대)        : {nd['required_search_radius_m']:8.0f} m")
        if args.apply:
            path = pathlib.Path(__file__).resolve().parents[1] / "config" / "common_experiment.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            previous = data["mission"]["search_radius_m"]
            radius = round(nd["required_search_radius_m"], 1)
            data["mission"]["search_radius_m"] = radius
            data["aoi"]["sizing_rule"] = "net-displacement containment"
            data["aoi"]["measured"] = {**nd, "horizon_s": args.horizon_s, "sample_count": args.sample_count,
                                       "previous_search_radius_m": previous,
                                       "note": "tools/size_aoi.py --rule net-displacement --apply 로 측정. boundary_mode=open."}
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"\nconfig 갱신: search_radius_m {previous:.0f} -> {radius:.0f} m")
        return 0
    if args.only in (None, "aoi"):
        aoi = containment_radius_m(config, mission, **shared)
        report["aoi"] = aoi
        print("\n== AOI 포함률 사이징 ==")
        print(f"최대속도 규칙 반경    : {aoi['max_speed_rule_radius_m']:8.0f} m")
        print(f"현재 config 반경      : {aoi['current_search_radius_m']:8.0f} m")
        print(
            f"포함률 {aoi['containment']:.0%} 필요 반경 : "
            f"{aoi['required_search_radius_m']:8.0f} m"
        )
        print(
            f"체류가중 {aoi['containment']:.0%} 반경  : "
            f"{aoi['presence_required_search_radius_m']:8.0f} m"
            "  (탐지는 종료시점이 아니라 임무 내내 일어난다)"
        )
        for key, entry in aoi["per_profile"].items():
            print(f"  [{key}] 종료반경  {_fmt(entry['end_radius_m'])}")
            print(f"  [{key}] 체류반경  {_fmt(entry['presence_radius_m'])}")
            print(f"  [{key}] 경로장    {_fmt(entry['path_length_m'])}")
            print(f"  [{key}] 순변위    {_fmt(entry['net_displacement_m'])}")
        print("  포함률 선택지 (반경 / 면적배수):")
        for row in aoi["containment_trade_off"]:
            print(
                f"    {row['containment']:.0%}  ->  "
                f"{row['search_radius_m']:7.0f} m   "
                f"x{row['area_ratio_vs_current']:.2f} 면적"
            )

    if args.only in (None, "step"):
        print("\n== step_s 민감도 (30 / 10 / 5 s) ==")
        report["step_sensitivity"] = {}
        for profile_key in ("tank", "tel"):
            sensitivity = step_sensitivity(
                config, mission, profile_key=profile_key, **shared
            )
            report["step_sensitivity"][profile_key] = sensitivity
            for row in sensitivity["rows"]:
                delta = row.get("vs_baseline", {})
                suffix = (
                    ""
                    if not delta
                    else "   Δp95(변위)={:+7.1f} m  Δp95(경로장)={:+7.1f} m".format(
                        delta["net_displacement_m"], delta["path_length_m"]
                    )
                )
                print(
                    f"  [{profile_key}] step={row['step_s']:5.1f}s "
                    f"종료반경 p95={row['end_radius_m']['p95']:7.1f} "
                    f"경로장 mean={row['path_length_m']['mean']:7.1f}{suffix}"
                )

    if args.only in (None, "halt"):
        print("\n== halt_probability_boost 민감도 ==")
        report["halt_sensitivity"] = {}
        for profile_key in ("tank", "tel"):
            sweep = halt_boost_sweep(config, mission, profile_key=profile_key, **shared)
            report["halt_sensitivity"][profile_key] = sweep
            for row in sweep["rows"]:
                marker = (
                    " <- 선언값"
                    if row["halt_probability_boost"] == sweep["declared_boost"]
                    else ""
                )
                print(
                    f"  [{profile_key}] boost={row['halt_probability_boost']:.2f} "
                    f"종료반경 p99={row['end_radius_m']['p99']:7.1f} "
                    f"(Δ{row['end_radius_p99_delta_m']:+7.1f}) "
                    f"경로장 mean={row['path_length_m']['mean']:7.1f}{marker}"
                )

    if args.json:
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\n측정 결과 저장: {args.json}")

    if args.apply:
        if "aoi" not in report:
            print("\n--apply 는 AOI 측정이 있어야 한다 (--only aoi 또는 전체 실행)")
            return 2
        data = json.loads(CONFIG_PATH.read_text())
        aoi = report["aoi"]
        radius = round(aoi["required_search_radius_m"], 1)
        previous = data["mission"]["search_radius_m"]
        data["mission"]["search_radius_m"] = radius
        data["aoi"]["measured"] = {
            "containment": aoi["containment"],
            "required_search_radius_m": radius,
            "previous_search_radius_m": previous,
            "max_speed_rule_radius_m": aoi["max_speed_rule_radius_m"],
            "sample_count": args.sample_count,
            "horizon_s": args.horizon_s,
            "per_profile_containment_radius_m": {
                key: entry["containment_radius_m"]
                for key, entry in aoi["per_profile"].items()
            },
            "note": (
                "tools/size_aoi.py --apply 로 측정. 진리 궤적은 boundary_mode=open "
                "으로 생성해 경계에서 끊지 않았다."
            ),
        }
        CONFIG_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        print(f"\nconfig 갱신: search_radius_m {previous:.0f} -> {radius:.0f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
