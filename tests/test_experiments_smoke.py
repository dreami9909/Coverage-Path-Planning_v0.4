"""네 챕터가 **축소 조건으로** 끝까지 돌고, 선언한 산출물 키를 실제로 내보내는지.

성능을 검사하는 파일이 아니다. 검사하는 것은 계약이다 — 챕터가 약속한 블록이
결과에 있는지, 러너가 그것을 JSON 으로 쓸 수 있는지, 그림 단계가 그 트리를
읽을 수 있는지. 실제 조건으로 돌리면 한 번에 수십 분이라 여기서는 seed 1 개,
표본 40 개, 학습 6 iteration 으로 줄인다.

**줄인 결과는 성능 주장에 쓸 수 없다.** 그래서 이 파일은 값의 크기를 보지 않고
구조만 본다.
"""

from __future__ import annotations

import unittest
from math import log

from cpp_search.config import load_chapter_config
from cpp_search.chapters import CHAPTER_MODULES
from cpp_search.options import RunOptions


def _options(**overrides) -> RunOptions:
    values = {
        "sample_count": 40,
        "particle_count": 200,
        "episode_count": 6,
        "target_profile": "both",
        "seed": 20_260_913,
        "map_error": False,
        "communication_loss_probability": 0.0,
        "communication_latency_slices": 0,
        "mission_time_s": 480.0,
    }
    values.update(overrides)
    return RunOptions(**values)


#: 스모크가 쓰는 축소 격자·예산. 선언된 10x10 · master 300 s x 20 회는
#: 실측 5,455 s 짜리라 그대로 두면 테스트가 개발 루프에서 쓸 수 없다.
#: 스모크의 목적은 "챕터가 끝까지 돌고 계약된 키를 내놓는가" 이므로
#: 인증 품질이 아니라 완주 여부만 본다. 인증 성립을 단언하는 곳은 없다.
_SMOKE_GRID = {
    "grid_width": 4,
    "grid_height": 4,
    "particle_count": 4 * 4 * 50,
    # master 시간만 줄이면 HiGHS 가 ``mip_relative_gap=0`` 때문에 루트
    # 휴리스틱(solveSubMip)에서 한도를 다 쓴다. 느슨한 상대 간격을 함께 주어야
    # 조기 종료한다. 연속완화와 국소개선도 스모크에는 필요 없다.
    "master_time_limit_s": 5.0,
    "max_iterations": 3,
    "mip_relative_gap": 0.05,
    "continuous_relaxation_iterations": 1,
    "local_improvement_passes": 0,
}


def _reduced_config(chapter: str):
    """축소 격자 하나 · seed 하나 · 학습 6 iteration 으로 조건을 낮춘다."""

    config = load_chapter_config(chapter)
    block = config.data.get("stone_spx")
    if isinstance(block, dict):
        block["planning_seeds"] = block["planning_seeds"][:1]
        grids = block.get("grid_conditions")
        if grids:
            reduced = dict(
                min(
                    grids,
                    key=lambda grid: grid.get("grid_width", 99)
                    * grid.get("grid_height", 99),
                )
            )
            # 격자 개수·시드·학습 반복만 줄이고 **솔버 예산은 그대로 두면**
            # 스모크가 운용 조건으로 실제 인증 계산을 돌린다. 선언된 10x10 ·
            # master 300 s x 20 회는 실측 5,455 s 짜리이고, 챕터 세 개면 4 시간이
            # 넘어 테스트가 개발 루프에서 쓸 수 없게 된다. 스모크의 목적은
            # "챕터가 끝까지 돌고 계약된 키를 내놓는가" 이지 인증 품질이 아니므로
            # 격자와 예산도 함께 낮춘다. 인증 성립을 단언하는 곳은 없다.
            reduced.update(_SMOKE_GRID)
            block["grid_conditions"] = [reduced]
        # Ch4 는 ``grid_conditions`` 가 아니라 ``evaluation_grid`` 를 읽는다.
        # 격자 출처가 챕터마다 다르므로 **선언된 모든 격자**를 낮춘다.
        if isinstance(block.get("evaluation_grid"), dict):
            block["evaluation_grid"] = {**block["evaluation_grid"], **_SMOKE_GRID}
        # Ch2 는 격자 폭을 ``case["certified_grid_width"]`` 에서 읽고(Tank 8 /
        # TEL 10), ``ch2_studies._grid_condition`` 이 입자수를 그 폭에서
        # ``width**2 * 50`` 으로 다시 만든다. 즉 ``grid_width`` 만 낮추면
        # 아무 효과가 없고, 스모크가 실제 인증 격자를 그대로 돌아 빌드에만
        # 케이스당 60~115 s 가 든다. 케이스 선언도 같이 낮춘다.
        for case in block.get("cases", []) or []:
            if isinstance(case, dict) and "certified_grid_width" in case:
                case["certified_grid_width"] = _SMOKE_GRID["grid_width"]
        if "terrain_families" in block:
            block["terrain_families"] = ["synthetic"]
    mappo = config.data.get("mappo")
    if isinstance(mappo, dict):
        mappo["iterations"] = 6
        mappo["rollouts_per_iteration"] = 2
        mappo["hidden_size"] = 16
    return config


class ChapterOutputTests(unittest.TestCase):
    def test_chapter1_declares_the_evaluation_contract(self) -> None:
        config = load_chapter_config("1")
        result = CHAPTER_MODULES["1"].run(config, _options())
        for key in (
            "sweep_width_table",
            "search_envelope",
            "target_profiles",
            "detection_contract",
            "monte_carlo",
            "kpi_definitions",
            "coverage_feasibility",
        ):
            self.assertIn(key, result)
        self.assertTrue(result["sweep_width_table"])
        # 800 m 지지폭이 W = 1600 m 가 아니라는 것이 이 챕터의 첫 주장이다.
        fused = [
            row
            for row in result["sweep_width_table"]
            if row.get("channel_mode") == "fused"
        ]
        self.assertTrue(fused)
        support = result["search_envelope"]["system_support_half_width_m"]
        for row in fused:
            self.assertLess(row["sweep_width_m"], 2.0 * support)

        # 증거 계약은 선언값이 아니라 **구현을 돌려서** 나온 값이어야 한다.
        contract = result["detection_contract"]
        self.assertEqual(
            contract["source"], "cpp_search.core.belief.LogOddsEvidenceGrid.update"
        )
        pd = contract["detection_probability"]
        pf = contract["false_alarm_probability"]
        self.assertAlmostEqual(
            contract["positive_report_log_odds_increment"],
            log(pd / pf),
            places=12,
        )
        self.assertAlmostEqual(
            contract["negative_report_log_odds_increment"],
            log((1.0 - pd) / (1.0 - pf)),
            places=12,
        )
        # 미탐지 보고는 belief 를 내려야 한다 (PD > PF 이므로 음수).
        self.assertLess(contract["negative_report_log_odds_increment"], 0.0)

    def test_chapter2_reports_an_optimality_certificate_per_grid(self) -> None:
        result = CHAPTER_MODULES["2"].run(_reduced_config("2"), _options())
        self.assertIn("certificate_gate", result)
        grids = result["grid_conditions"]
        self.assertTrue(grids)
        for block in grids:
            certificate = block["optimality_certificate"]
            for key in (
                "required_relative_gap",
                "worst_relative_gap",
                "passed",
                "seed_count",
            ):
                self.assertIn(key, certificate)
            for row in block["terrain_weighted"]:
                self.assertTrue(row["terrain_weighting"])
            # 대조군은 **선언될 때만** 돈다. 그 선언이 결과에 남아 있어야
            # 대조군 없는 블록과 대조군이 조용히 빠진 블록을 구별할 수 있다.
            self.assertIn("terrain_weighting_contrast", block)
            if not block["terrain_weighting_contrast"]:
                self.assertNotIn("terrain_weighting_effect", block)
                self.assertNotIn("terrain_unweighted", block)
                continue
            effect = block["terrain_weighting_effect"]
            # 부호 방향이 결과에 문자열로 박혀 있어야 그림이 추측하지 않는다.
            self.assertIn("minus", effect["mean_difference_is"])
            for row in block["terrain_unweighted"]:
                self.assertFalse(row["terrain_weighting"])

    def test_chapter3_compares_both_planners_on_one_instance(self) -> None:
        result = CHAPTER_MODULES["3"].run(_reduced_config("3"), _options())
        self.assertIn("policy_capacity", result)
        for block in result["grid_conditions"]:
            spx = {row["seed"]: row for row in block["spx"]}
            mappo = {row["seed"]: row for row in block["mappo"]}
            self.assertEqual(sorted(spx), sorted(mappo))
            for seed in spx:
                # 비교가 성립하는 유일한 근거: 같은 인스턴스.
                self.assertEqual(
                    spx[seed]["common_input_fingerprint"],
                    mappo[seed]["common_input_fingerprint"],
                )
            self.assertIn("paired_spx_minus_mappo", block)
            self.assertIn("achievement", block)
            self.assertIn("learning_effect", block)
            for row in block["mappo"]:
                # MAPPO 는 실행가능집합을 벗어나지 않는다.
                self.assertEqual(row["occupancy_violations"], 0)
                self.assertEqual(row["separation_violations"], 0)
                self.assertLessEqual(row["action_candidate_coverage"], 1.0)

    def test_chapter4_flies_both_planners_through_the_same_evaluator(self) -> None:
        result = CHAPTER_MODULES["4"].run(_reduced_config("4"), _options())
        families = result["terrain_families"]
        self.assertTrue(families)
        for family in families:
            self.assertIn("kpi", family)
            for planner in ("stone-spx", "mappo-stone"):
                self.assertIn(planner, family["kpi"])
                combined = family["kpi"][planner]["combined"]
                self.assertIn("detection_probability_within_limit", combined)
            for row in family["per_seed"]:
                # 두 계획법이 같은 인스턴스를 받았다는 것이 chapter 4 의 전제다.
                self.assertIn("common_input_fingerprint", row)
            self.assertIn("paired_spx_minus_mappo", family)


if __name__ == "__main__":
    unittest.main()
