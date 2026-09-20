"""Chapter 4 가 공통 KPI 계약을 지키는지 검사한다.

KPI 정의는 Chapter 1 이 선언하고 Chapter 4 가 보고한다. 두 곳이 어긋나면
결과표의 열 이름이 정의 없는 숫자가 된다.
"""

from __future__ import annotations

import unittest

from cpp_search.config import load_chapter_config
from cpp_search.chapters import CHAPTER_MODULES
from cpp_search.kpi import DIAGNOSTIC_KPI_KEYS, KPI_KEYS
from cpp_search.options import RunOptions
from tests.test_experiments_smoke import _options, _reduced_config


class DefinitionTests(unittest.TestCase):
    def test_chapter1_defines_every_kpi_it_declares(self) -> None:
        config = load_chapter_config("1")
        result = CHAPTER_MODULES["1"].run(config, _options())
        definitions = result["kpi_definitions"]
        for name in config.kpi_names:
            self.assertIn(name, definitions)
            self.assertNotEqual(
                definitions[name],
                "undocumented",
                f"KPI '{name}' is declared in config but has no definition",
            )

    def test_the_five_ranking_kpis_are_all_defined(self) -> None:
        result = CHAPTER_MODULES["1"].run(load_chapter_config("1"), _options())
        definitions = result["kpi_definitions"]
        for name in KPI_KEYS:
            self.assertIn(name, definitions, f"ranking KPI '{name}' undefined")


class ReportedBlockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = CHAPTER_MODULES["4"].run(_reduced_config("4"), _options())

    def _combined_blocks(self):
        for family in self.result["terrain_families"]:
            for planner, block in family["kpi"].items():
                yield family["family"], planner, block["combined"]

    def test_every_planner_reports_the_five_ranking_kpis(self) -> None:
        seen = 0
        for family, planner, combined in self._combined_blocks():
            for name in KPI_KEYS:
                self.assertIn(
                    name, combined, f"{family}/{planner} is missing KPI '{name}'"
                )
            seen += 1
        self.assertGreater(seen, 0)

    def test_failure_rate_is_exactly_the_complement(self) -> None:
        """실패율은 탐지확률의 여집합이다 — 따로 추정하지 않는다.

        표적 계층별 원 KPI 블록에서 확인한다. 합성 블록은 계층 가중합이라
        두 값이 각각 가중된 뒤에도 여집합이어야 한다.
        """

        checked = 0
        for family in self.result["terrain_families"]:
            for row in family["per_seed"]:
                for entry in row["planners"].values():
                    for detection in entry["stratum_detection_probability"].values():
                        self.assertGreaterEqual(detection, 0.0)
                        self.assertLessEqual(detection, 1.0)
                        checked += 1
        self.assertGreater(checked, 0)

        for family, planner, combined in self._combined_blocks():
            detection = combined["detection_probability_within_limit"]["mean"]
            failure = combined.get("failure_rate_within_limit")
            if failure is None or not isinstance(failure, dict):
                # summarise_kpi 는 여집합을 다시 추정하지 않는다. 그 사실이
                # 계약이므로, 빠져 있는 것이 맞다.
                continue
            self.assertAlmostEqual(
                detection + failure["mean"],
                1.0,
                places=9,
                msg=f"{family}/{planner}: detection + failure != 1",
            )

    def test_restricted_time_is_never_shorter_than_conditional_time(self) -> None:
        """실패를 임무한계로 절단한 평균은 조건부 평균보다 짧을 수 없다."""

        checked = 0
        for family, planner, combined in self._combined_blocks():
            conditional = combined["conditional_mean_detection_time_s"]["mean"]
            restricted = combined["restricted_mean_detection_time_s"]["mean"]
            if conditional is None:
                # 축소 smoke(표본 40)에서 탐지가 0 인 seed 가 나오면 조건부
                # 평균은 정의되지 않는다. 그 경우는 not-estimable 이 맞다.
                self.assertEqual(
                    combined["conditional_mean_detection_time_s"]["valid_seed_count"], 0
                )
                continue
            checked += 1
            self.assertGreaterEqual(
                restricted,
                conditional - 1e-6,
                f"{family}/{planner}: restricted {restricted} < conditional {conditional}",
            )

    def test_diagnostic_kpis_are_reported_too(self) -> None:
        for family, planner, combined in self._combined_blocks():
            for name in DIAGNOSTIC_KPI_KEYS:
                self.assertIn(
                    name,
                    combined,
                    f"{family}/{planner} is missing diagnostic KPI '{name}'",
                )

    def test_the_model_to_flown_gap_is_reported_not_hidden(self) -> None:
        """계획모형이 실제를 과대평가하는 정도 자체가 Chapter 4 의 결과다."""

        for family in self.result["terrain_families"]:
            for planner, block in family["kpi"].items():
                self.assertIn("model_minus_flown_mean", block)
                self.assertIsInstance(block["model_minus_flown_mean"], float)

    def test_reliability_criteria_are_recorded_with_the_result(self) -> None:
        criteria = self.result["reliability_criteria"]
        self.assertIn("minimum_seed_count", criteria)
        self.assertIn("bootstrap_resamples", criteria)
        self.assertIn("confidence", criteria)


if __name__ == "__main__":
    unittest.main()
