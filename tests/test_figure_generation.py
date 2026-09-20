"""그림·CSV 산출이 실제 결과 트리에서 만들어지는지 검사한다.

v0.3 에서 Ch8 의 figure 단계가 ``NameError`` 로 죽었는데 종료코드 0 으로
넘어가서, 산출물이 빠진 채로 "정상 종료"한 일이 있었다. 그래서 두 가지를
같이 본다 — 파일이 만들어지는지, 그리고 실패했을 때 러너가 그것을 **실패로**
보고하는지.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cpp_search.chapters import CHAPTER_MODULES
from cpp_search.figures import write_chapter_artifacts
from cpp_search.runner import _artifact_error, _json_safe
from tests.test_experiments_smoke import _options, _reduced_config


def _artifacts(chapter: str, result: dict, directory: Path) -> dict:
    payload = _json_safe(result)
    destination = directory / f"chapter-{chapter}-result.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return write_chapter_artifacts(chapter, payload, destination)


class ArtifactTests(unittest.TestCase):
    def _run(self, chapter: str) -> None:
        config = (
            _reduced_config(chapter)
            if chapter != "1"
            else _reduced_config(chapter)
        )
        result = CHAPTER_MODULES[chapter].run(config, _options())
        with tempfile.TemporaryDirectory() as name:
            artifacts = _artifacts(chapter, result, Path(name))
            self.assertNotIn("error", artifacts)
            self.assertTrue(artifacts["files"], f"chapter {chapter} made no artifacts")
            for path in artifacts["files"]:
                self.assertTrue(Path(path).is_file())
                self.assertGreater(Path(path).stat().st_size, 0)

    def test_chapter1_writes_the_sweep_width_table(self) -> None:
        self._run("1")

    def test_chapter2_writes_the_certificate_figures(self) -> None:
        self._run("2")

    def test_chapter3_writes_the_comparison_figures(self) -> None:
        self._run("3")

    def test_chapter4_writes_the_flown_kpi_figures(self) -> None:
        self._run("4")

    def test_stale_artifacts_are_removed(self) -> None:
        result = CHAPTER_MODULES["1"].run(_reduced_config("1"), _options())
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            first = _artifacts("1", result, directory)
            stale = Path(first["directory"]) / "chapter-1-99-left-over.png"
            stale.write_bytes(b"stale")
            second = _artifacts("1", result, directory)
            self.assertIn(str(stale), second["removed_stale"])
            self.assertFalse(stale.exists())


class ArtifactFailureReportingTests(unittest.TestCase):
    def test_an_artifact_failure_is_reported_not_swallowed(self) -> None:
        """그림 실패는 결과 JSON 을 지키되 **성공으로 보고하지 않는다**."""

        self.assertIsNone(_artifact_error({"artifacts": {"files": []}}))
        self.assertEqual(
            _artifact_error({"artifacts": {"error": "NameError: boom"}}),
            "NameError: boom",
        )


if __name__ == "__main__":
    unittest.main()
