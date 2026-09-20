import unittest
from hashlib import sha256
from json import dumps, loads
from pathlib import Path

from cpp_search.config import ChapterConfig


class ConfigurationSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ChapterConfig(
            "6",
            {"schema_version": 2, "example": {"count": 4}},
            {"label": "shared", "values": [1, 2]},
            (Path("chapter6.json"),),
        )
        self.options = {"seed": 17, "fast": True}

    def test_snapshot_preserves_all_three_inputs(self) -> None:
        result = self.config.provenance(run_options=self.options)
        snapshot = result["configuration_snapshot"]
        self.assertEqual(snapshot["chapter_config"], self.config.data)
        self.assertEqual(snapshot["common_config"], self.config.common)
        self.assertEqual(snapshot["run_options"], self.options)
        self.assertEqual(result["configuration_snapshot_version"], 2)
        self.assertEqual(loads(dumps(result)), result)

    def test_chapter_identifier_stays_outside_the_hashed_snapshot(self) -> None:
        """``--chapter 6`` 과 ``6a1`` 은 같은 설정의 별칭이다 (형식 2).

        식별자를 해시에 넣으면 결과가 비트 단위로 같은데도 지문이 갈린다.
        챕터 이름 자체는 해시 밖에 그대로 남아야 한다.
        """

        alias = ChapterConfig(
            "6a1", self.config.data, self.config.common, self.config.source_paths
        )
        first = self.config.provenance(run_options=self.options)
        second = alias.provenance(run_options=self.options)

        self.assertNotIn("chapter", first["configuration_snapshot"])
        self.assertEqual(first["configuration_sha256"], second["configuration_sha256"])
        self.assertEqual(first["chapter"], "6")
        self.assertEqual(second["chapter"], "6a1")

    def test_hash_can_be_recomputed_from_saved_snapshot(self) -> None:
        result = self.config.provenance(run_options=self.options)
        serialized = dumps(
            result["configuration_snapshot"],
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        self.assertEqual(
            result["configuration_sha256"],
            sha256(serialized.encode("utf-8")).hexdigest(),
        )

    def test_key_order_does_not_change_hash(self) -> None:
        first = self.config.provenance(run_options=self.options)
        reordered = dict(reversed(list(self.options.items())))
        second = self.config.provenance(run_options=reordered)
        self.assertEqual(first["configuration_sha256"], second["configuration_sha256"])

    def test_each_input_changes_hash_without_changing_filename(self) -> None:
        previous = self.config.provenance(run_options=self.options)
        changes = (
            (self.config.data["example"], "count", 5),
            (self.config.common, "label", "changed"),
            (self.options, "seed", 18),
        )
        for mapping, key, value in changes:
            with self.subTest(key=key):
                mapping[key] = value
                current = self.config.provenance(run_options=self.options)
                self.assertNotEqual(
                    previous["configuration_sha256"], current["configuration_sha256"]
                )
                self.assertEqual(previous["config_files"], current["config_files"])
                previous = current

    def test_snapshot_is_detached_from_mutable_inputs(self) -> None:
        result = self.config.provenance(run_options=self.options)
        self.config.data["example"]["count"] = 999
        self.config.common["values"].append(3)
        self.options["seed"] = 999
        snapshot = result["configuration_snapshot"]
        self.assertEqual(snapshot["chapter_config"]["example"]["count"], 4)
        self.assertEqual(snapshot["common_config"]["values"], [1, 2])
        self.assertEqual(snapshot["run_options"]["seed"], 17)

    def test_existing_callers_keep_the_previous_fields(self) -> None:
        result = self.config.provenance()
        self.assertNotIn("configuration_snapshot", result)
        self.assertNotIn("configuration_sha256", result)
        self.assertIn("code_fingerprint", result)

    def test_non_finite_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.config.provenance(run_options={"value": float("nan")})


class SeedLimitScopeTests(unittest.TestCase):
    """``--seed-count`` 는 그 챕터가 읽는 키만 바꿔야 한다.

    네 개 키를 무조건 다 쓰면 안 읽히는 값이 ``configuration_snapshot`` 에
    실려, KPI 가 비트 단위로 같은 두 실행의 ``configuration_sha256`` 이
    갈린다. v0.3 Ch6a-2 저장본과 재실행본이 실제로 그렇게 갈렸다.
    """

    def test_declared_keys_match_what_each_chapter_reads(self) -> None:
        from cpp_search.config import load_chapter_config
        from cpp_search.runner import ALL_CHAPTERS, seed_limit_keys

        expected = {
            "1": (),
            "2": ("stone_spx.planning_seed_limit",),
            "3": ("stone_spx.planning_seed_limit",),
            "4": ("stone_spx.evaluation_seed_limit",),
        }
        for chapter in ALL_CHAPTERS:
            with self.subTest(chapter=chapter):
                config = load_chapter_config(chapter)
                self.assertEqual(seed_limit_keys(chapter, config), expected[chapter])

    def test_dotted_keys_reach_their_branch(self) -> None:
        from cpp_search.runner import _apply_seed_limit

        config = ChapterConfig(
            "2", {"stone_spx": {"planning_seeds": [1, 2, 3]}}, {}, ()
        )
        _apply_seed_limit(config, "2", 2)
        self.assertEqual(config.data["stone_spx"]["planning_seed_limit"], 2)

    def test_unread_keys_are_left_out_of_the_snapshot(self) -> None:
        """Chapter 2 는 evaluation 한도를 읽지 않는다 — 넣으면 지문만 흐린다."""

        from cpp_search.runner import _apply_seed_limit

        config = ChapterConfig(
            "2", {"stone_spx": {"planning_seeds": [1, 2, 3]}}, {}, ()
        )
        _apply_seed_limit(config, "2", 3)
        block = config.data["stone_spx"]
        self.assertEqual(block["planning_seed_limit"], 3)
        self.assertNotIn("evaluation_seed_limit", block)
        self.assertNotIn("planning_seed_limit", config.data)

    def test_chapter_4_reads_the_evaluation_limit_not_the_planning_one(self) -> None:
        from cpp_search.runner import _apply_seed_limit

        config = ChapterConfig(
            "4", {"stone_spx": {"planning_seeds": [1, 2, 3]}}, {}, ()
        )
        _apply_seed_limit(config, "4", 2)
        block = config.data["stone_spx"]
        self.assertEqual(block["evaluation_seed_limit"], 2)
        self.assertNotIn("planning_seed_limit", block)

    def test_a_chapter_without_a_limit_key_is_untouched(self) -> None:
        """Chapter 1 은 seed 를 쓰지 않는다. --seed-count 가 무엇도 바꾸면 안 된다."""

        from cpp_search.runner import _apply_seed_limit

        config = ChapterConfig("1", {"motion_sensitivity": {}}, {}, ())
        _apply_seed_limit(config, "1", 2)
        self.assertEqual(config.data, {"motion_sensitivity": {}})


class RunLevelSummaryTests(unittest.TestCase):
    """한 번의 ``--chapter all`` 은 하나의 코드 상태를 가리켜야 한다."""

    def test_one_fingerprint_is_not_a_conflict(self) -> None:
        from cpp_search.runner import _run_block

        block = _run_block(
            {"1": {"code_fingerprint": "aaaa"}, "2": {"code_fingerprint": "aaaa"}},
            [],
            {},
        )
        self.assertFalse(block["fingerprint_conflict"])
        self.assertEqual(block["code_fingerprint"], "aaaa")

    def test_split_fingerprints_are_reported_with_their_chapters(self) -> None:
        from cpp_search.runner import _run_block

        block = _run_block(
            {
                "1": {"code_fingerprint": "aaaa"},
                "2": {"code_fingerprint": "bbbb"},
                "3": {"code_fingerprint": "aaaa"},
            },
            [],
            {},
        )
        self.assertTrue(block["fingerprint_conflict"])
        self.assertIsNone(block["code_fingerprint"])
        self.assertEqual(block["fingerprints"], {"aaaa": ["1", "3"], "bbbb": ["2"]})

    def test_artifact_failures_are_recorded(self) -> None:
        from cpp_search.runner import _artifact_error, _run_block

        self.assertIsNone(_artifact_error({"artifacts": {"files": ["a.png"]}}))
        self.assertEqual(
            _artifact_error({"artifacts": {"error": "NameError: kpi", "files": []}}),
            "NameError: kpi",
        )
        block = _run_block({"8": {"code_fingerprint": "aaaa"}}, [], {"8": "NameError"})
        self.assertEqual(block["artifact_failures"], {"8": "NameError"})

    def test_the_run_block_is_not_printed_as_a_chapter(self) -> None:
        from io import StringIO
        from unittest.mock import patch

        from cpp_search.runner import _print_summary

        headlines = {
            "1": {"runtime_s": 1.0, "status": "ok", "conditions": {}},
            "_run": {"fingerprint_conflict": False},
        }
        with patch("sys.stdout", new=StringIO()) as stream:
            _print_summary(headlines)
        self.assertNotIn("_run", stream.getvalue())


class ArtifactFailureExitTests(unittest.TestCase):
    """그림/CSV 실패를 성공으로 보고하지 않는다.

    결과 JSON 은 계산이 비싸므로 먼저 저장하고 예외는 삼킨다. 그러나 실행이
    종료코드 0 으로 끝나면 산출물이 빠진 채로 "정상 종료"한 것이 되고,
    실제로 Ch8 의 figure NameError 가 그렇게 넘어갔다.
    """

    def _run(self, directory, *extra):
        from unittest.mock import patch

        from cpp_search.runner import main

        argv = [
            "--chapter", "1",
            "--output", str(Path(directory) / "chapter-1-result.json"),
            *extra,
        ]
        with patch(
            "cpp_search.figures.write_chapter_artifacts",
            side_effect=RuntimeError("boom"),
        ):
            return main(argv)

    def test_a_failed_artifact_run_exits_non_zero(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            with self.assertRaises(SystemExit) as raised:
                self._run(directory)
            self.assertNotEqual(raised.exception.code, 0)
            self.assertIn("boom", str(raised.exception))

    def test_the_result_json_survives_the_failure(self) -> None:
        from json import loads
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            destination = Path(directory) / "chapter-1-result.json"
            with self.assertRaises(SystemExit):
                self._run(directory)
            saved = loads(destination.read_text(encoding="utf-8"))
            self.assertIn("RuntimeError: boom", saved["artifacts"]["error"])
            self.assertIn("provenance", saved)

    def test_a_clean_run_returns_normally(self) -> None:
        from tempfile import TemporaryDirectory

        from cpp_search.runner import main

        with TemporaryDirectory() as directory:
            main([
                "--chapter", "1",
                "--output", str(Path(directory) / "chapter-1-result.json"),
            ])


class FingerprintConflictExitTests(unittest.TestCase):
    """한 번의 ``--chapter all`` 은 하나의 코드 상태를 가리켜야 한다.

    저장된 결과 묶음이 서로 다른 지문 여섯 개로 쪼개져 있었는데, 그 사실이
    각 결과의 ``provenance`` 에 이미 적혀 있었는데도 아무도 대조하지 않아
    드러나지 않았다.
    """

    @staticmethod
    def _fake_runner(fingerprints):
        """챕터를 실제로 돌리지 않고 결과 파일만 쓴다.

        이 테스트가 검사하는 것은 실험이 아니라 **러너의 지문 대조 논리**다.
        실제 챕터를 돌리면 Chapter 2 의 MILP 때문에 단위 테스트 하나가 수 분
        걸린다 (실제로 10분이었다).
        """

        from json import dumps

        def run_one(chapter, arguments, *, output_path=None):
            destination = output_path or (
                Path(arguments.output) / f"chapter-{chapter}-result.json"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                dumps(
                    {
                        "chapter": chapter,
                        "provenance": {"code_fingerprint": next(fingerprints)},
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            return destination

        return run_one

    def test_a_split_fingerprint_run_exits_non_zero_and_is_recorded(self) -> None:
        from json import loads
        from tempfile import TemporaryDirectory
        from unittest.mock import patch

        from cpp_search import runner

        fingerprints = iter(("aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb"))
        with TemporaryDirectory() as directory:
            with patch.object(runner, "ALL_CHAPTERS", ("1", "2")), patch.object(
                runner, "_run_one", self._fake_runner(fingerprints)
            ):
                with self.assertRaises(SystemExit) as raised:
                    runner.main(["--chapter", "all", "--output", directory])
            self.assertNotEqual(raised.exception.code, 0)
            summary = loads(
                (Path(directory) / "summary.json").read_text(encoding="utf-8")
            )
        self.assertTrue(summary["_run"]["fingerprint_conflict"])
        self.assertIsNone(summary["_run"]["code_fingerprint"])
        self.assertEqual(
            summary["_run"]["fingerprints"],
            {"aaaaaaaaaaaaaaaa": ["1"], "bbbbbbbbbbbbbbbb": ["2"]},
        )

    def test_one_fingerprint_run_returns_normally(self) -> None:
        from itertools import repeat
        from json import loads
        from tempfile import TemporaryDirectory
        from unittest.mock import patch

        from cpp_search import runner

        with TemporaryDirectory() as directory:
            with patch.object(runner, "ALL_CHAPTERS", ("1", "2")), patch.object(
                runner, "_run_one", self._fake_runner(repeat("cccccccccccccccc"))
            ):
                runner.main(["--chapter", "all", "--output", directory])
            summary = loads(
                (Path(directory) / "summary.json").read_text(encoding="utf-8")
            )
        self.assertFalse(summary["_run"]["fingerprint_conflict"])
        self.assertEqual(summary["_run"]["code_fingerprint"], "cccccccccccccccc")
