"""조건값의 단일 출처가 지켜지는지 검사한다.

해석 순서는 ``명령행 > config 파일 > 코드 기본값`` 이다. 이 순서가 깨지면
"플래그 없이 실행하면 config 가 그대로 재현된다"가 거짓이 되고, 저장된 결과와
재실행 결과를 비교할 근거가 사라진다.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from cpp_search.config import (
    CHAPTER_CONFIG_FILES,
    COMMON_CONFIG_FILE,
    find_config_root,
    load_chapter_config,
)
from cpp_search.chapters import CHAPTER_MODULES
from cpp_search.runner import ALL_CHAPTERS, SEED_LIMIT_KEYS


class ConfigInventoryTests(unittest.TestCase):
    def test_every_chapter_module_has_declared_config_files(self) -> None:
        self.assertEqual(sorted(CHAPTER_MODULES), sorted(CHAPTER_CONFIG_FILES))

    def test_the_runner_dispatches_exactly_the_declared_chapters(self) -> None:
        self.assertEqual(sorted(ALL_CHAPTERS), sorted(CHAPTER_MODULES))

    def test_every_declared_config_file_exists_and_is_valid_json(self) -> None:
        root = find_config_root()
        names = {COMMON_CONFIG_FILE}
        for files in CHAPTER_CONFIG_FILES.values():
            names.update(files)
        for name in sorted(names):
            path = root / name
            self.assertTrue(path.is_file(), f"missing config file: {name}")
            json.loads(path.read_text(encoding="utf-8"))

    def test_no_orphan_config_files_are_left_behind(self) -> None:
        """읽히지 않는 config 는 지운다. 남아 있으면 조건이 둘로 보인다."""

        root = find_config_root()
        declared = {COMMON_CONFIG_FILE}
        for files in CHAPTER_CONFIG_FILES.values():
            declared.update(files)
        present = {path.name for path in root.glob("*.json")}
        self.assertEqual(sorted(present - declared), [])

    def test_every_chapter_declares_which_seed_limit_key_it_reads(self) -> None:
        self.assertEqual(sorted(SEED_LIMIT_KEYS), sorted(CHAPTER_MODULES))


class ResolutionOrderTests(unittest.TestCase):
    def test_an_explicit_value_beats_the_config_file(self) -> None:
        config = load_chapter_config("2")
        declared = config.common_get("monte_carlo.sample_count")
        self.assertIsNotNone(declared)
        self.assertEqual(
            config.resolve(7, "monte_carlo.sample_count", 1), 7
        )

    def test_the_config_file_beats_the_code_default(self) -> None:
        config = load_chapter_config("2")
        declared = config.common_get("mission.mission_time_s")
        self.assertEqual(
            config.resolve(None, "mission.mission_time_s", 1.0), declared
        )

    def test_the_code_default_only_applies_when_nothing_is_declared(self) -> None:
        config = load_chapter_config("2")
        self.assertEqual(
            config.resolve(None, "a.key.nobody.declares", 42), 42
        )


class ChapterConditionTests(unittest.TestCase):
    def test_chapter_2_and_3_declare_the_same_certified_grid(self) -> None:
        """Chapter 3 의 도달률 주장은 Chapter 2 가 인증한 격자에서만 유효하다.

        두 config 가 그 격자를 다르게 선언하면 주장이 서로 다른 문제에
        대한 것이 된다.
        """

        second = {
            grid["name"]: grid
            for grid in load_chapter_config("2").get("stone_spx.grid_conditions")
        }
        third = {
            grid["name"]: grid
            for grid in load_chapter_config("3").get("stone_spx.grid_conditions")
        }
        shared = sorted(set(second) & set(third))
        self.assertTrue(shared, "chapter 2 and 3 share no grid condition")
        compared = (
            "grid_kind",
            "grid_width",
            "grid_height",
            "time_slice_count",
            "radial_step_m",
            "angular_bin_count",
            "cell_occupancy_limit",
            "reservation_separation_m",
            "hazard_calibration",
        )
        for name in shared:
            for key in compared:
                self.assertEqual(
                    second[name].get(key),
                    third[name].get(key),
                    f"grid '{name}' differs on {key} between chapters 2 and 3",
                )

    def test_chapter_4_evaluates_on_a_grid_chapter_2_can_certify(self) -> None:
        second = {
            grid["name"]: grid
            for grid in load_chapter_config("2").get("stone_spx.grid_conditions")
        }
        grid = load_chapter_config("4").get("stone_spx.evaluation_grid")
        self.assertIn(grid["name"], second)

    def test_the_certificate_gate_is_declared_before_the_run(self) -> None:
        config = load_chapter_config("2")
        required = config.get("stone_spx.certificate.required_relative_gap")
        self.assertIsInstance(required, float)
        self.assertGreater(required, 0.0)
        self.assertLess(required, 1.0)


if __name__ == "__main__":
    unittest.main()
