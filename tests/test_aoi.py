"""AOI 사이징 규칙이 well-posed 한지 지킨다.

가장 중요한 성질은 **고정점 안정성**이다. 포함률 규칙이 현재 AOI 반경에
의존하면 (비율 사전분포를 쓰면 그렇게 된다) 적용할 때마다 반경이 커져
답이 없어진다. 아래 첫 테스트가 그 회귀를 막는다.
"""

import unittest
from dataclasses import replace

from cpp_search.aoi import (
    containment_radius_m,
    describe,
    end_radius_samples,
    halt_boost_sweep,
    step_sensitivity,
)
from cpp_search.config import load_chapter_config
from cpp_search.truth import cue_prior


class AoiSizingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_chapter_config("1")
        self.mission = self.config.mission()
        self.kwargs = {"sample_count": 400, "horizon_s": 480.0}

    def test_required_radius_does_not_depend_on_the_current_aoi(self) -> None:
        """R을 바꿔도 답이 같아야 한다. 아니면 규칙이 발산한다."""

        answers = []
        for radius_m in (2000.0, 3333.3, 8000.0):
            candidate = replace(self.mission, search_radius_m=radius_m)
            answers.append(
                containment_radius_m(self.config, candidate, **self.kwargs)[
                    "required_search_radius_m"
                ]
            )
        for value in answers[1:]:
            self.assertAlmostEqual(value, answers[0], places=6)

    def test_cue_ring_is_declared_in_absolute_metres(self) -> None:
        prior, reference_m, cue = cue_prior(self.config)
        self.assertIn(cue["kind"], {"moving-ring", "tp-centered"})
        self.assertEqual(prior.kind, cue["kind"])
        # tp-centered 는 TP 중심이므로 평균반경이 0 이고, 오차(sigma)만 양수다.
        # moving-ring 은 둘 다 양수. 어느 쪽이든 값은 절대 미터로 선언된다.
        if cue["kind"] == "tp-centered":
            self.assertEqual(cue["mean_radius_m"], 0.0)
        else:
            self.assertGreater(cue["mean_radius_m"], 0.0)
        self.assertGreater(cue["sigma_m"], 0.0)
        # 기준반경은 AOI 반경과 분리돼 있어야 한다.
        self.assertEqual(reference_m, cue["reference_radius_m"])
        # 절대 미터 계약: 다른 AOI 로 다시 만들어도 sigma_m 은 그대로다.
        _, _, cue_other = cue_prior(self.config, 9_000.0)
        self.assertEqual(cue_other["sigma_m"], cue["sigma_m"])

    def test_truth_is_generated_with_an_open_boundary(self) -> None:
        """경계에서 끊으면 종료반경이 AOI 근처에 몰려 측정이 무의미해진다."""

        samples = end_radius_samples(self.config, self.mission, **self.kwargs)
        _, reference_m, _ = cue_prior(self.config)
        self.assertGreater(max(samples["end_radius_m"]), reference_m)

    def test_presence_radius_is_never_larger_than_the_terminal_radius(self) -> None:
        result = containment_radius_m(self.config, self.mission, **self.kwargs)
        self.assertLessEqual(
            result["presence_required_search_radius_m"],
            result["required_search_radius_m"],
        )

    def test_containment_trade_off_is_monotone(self) -> None:
        rows = containment_radius_m(self.config, self.mission, **self.kwargs)[
            "containment_trade_off"
        ]
        radii = [row["search_radius_m"] for row in rows]
        self.assertEqual(radii, sorted(radii))

    def test_tel_travels_further_than_tank(self) -> None:
        """거동 배정이 실제로 갈렸는지. 같으면 Tank/TEL 비교가 무의미해진다."""

        tank = describe(
            end_radius_samples(
                self.config, self.mission, profile_key="tank", **self.kwargs
            )["path_length_m"]
        )
        tel = describe(
            end_radius_samples(
                self.config, self.mission, profile_key="tel", **self.kwargs
            )["path_length_m"]
        )
        self.assertGreater(tel["mean"], tank["mean"] * 1.2)


class SensitivityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_chapter_config("1")
        self.mission = self.config.mission()
        self.kwargs = {"sample_count": 300, "horizon_s": 480.0}

    def test_step_sensitivity_reports_every_requested_step(self) -> None:
        result = step_sensitivity(
            self.config, self.mission, steps_s=(30.0, 10.0), **self.kwargs
        )
        self.assertEqual([row["step_s"] for row in result["rows"]], [30.0, 10.0])
        self.assertIn("vs_baseline", result["rows"][1])

    def test_halt_boost_sweep_is_anchored_on_the_declared_value(self) -> None:
        result = halt_boost_sweep(
            self.config, self.mission, boosts=(0.0, 0.35), **self.kwargs
        )
        declared = [
            row
            for row in result["rows"]
            if row["halt_probability_boost"] == result["declared_boost"]
        ]
        self.assertEqual(len(declared), 1)
        self.assertEqual(declared[0]["end_radius_p99_delta_m"], 0.0)

    def test_halting_more_shortens_the_path(self) -> None:
        result = halt_boost_sweep(
            self.config, self.mission, boosts=(0.0, 0.5), **self.kwargs
        )
        self.assertLess(
            result["rows"][1]["path_length_m"]["mean"],
            result["rows"][0]["path_length_m"]["mean"],
        )


if __name__ == "__main__":
    unittest.main()
