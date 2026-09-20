import unittest

import numpy as np

from cpp_search.core.models import MissionConfig, PathSegment, Point2D, Route, SensorSpec
from cpp_search.core.belief import LogOddsEvidenceGrid, SearchProbabilityMetrics
from cpp_search.core.profiles import NOMINAL_TARGET_PROFILES
from cpp_search.core.search_envelope import CompositeSearchEnvelope
from cpp_search.core.evaluation import evaluate_routes, probability_mass_coverage


class ResearchFoundationTests(unittest.TestCase):
    def test_400m_support_is_not_800m_sweep_width(self) -> None:
        envelope = CompositeSearchEnvelope.operational_400m()

        self.assertAlmostEqual(envelope.support_half_width_m, 400.0)
        self.assertAlmostEqual(envelope.sweep_width_m(), 400.0, places=6)
        self.assertLess(envelope.sweep_width_m(), 2.0 * envelope.support_half_width_m)

    def test_weave_increases_flown_distance_and_reduces_centerline_progress(self) -> None:
        envelope = CompositeSearchEnvelope.operational_400m()

        self.assertGreater(envelope.actual_distance_m(1_000.0), 1_000.0)
        self.assertLess(envelope.centerline_progress_speed_mps(30.0), 30.0)

    def test_sensor_off_and_on_ingress_can_share_identical_flight_timing(self) -> None:
        mission = MissionConfig()
        sensor = SensorSpec.sr_z50()
        start = Point2D(-5_000.0, 0.0)
        end = Point2D(0.0, 0.0)
        off = PathSegment(
            start,
            end,
            False,
            speed_mps=mission.transit_speed_mps,
            search_pattern=True,
        )
        on = PathSegment(
            start,
            end,
            True,
            speed_mps=mission.transit_speed_mps,
            search_pattern=True,
        )

        self.assertAlmostEqual(off.duration_s(mission, sensor), on.duration_s(mission, sensor))
        self.assertEqual(off.effective_detection_scale, 0.0)
        self.assertEqual(on.effective_detection_scale, 1.0)

    def test_log_odds_and_single_target_location_maps_are_distinct(self) -> None:
        grid = LogOddsEvidenceGrid.from_prior([0.2, 0.3, 0.5])
        before = grid.target_location_probability.copy()
        grid.update(
            np.array([False, False, True]),
            detected=False,
            detection_probability=0.8,
            false_alarm_probability=0.05,
        )
        after = grid.target_location_probability

        self.assertAlmostEqual(float(after.sum()), 1.0)
        self.assertLess(after[2], before[2])
        self.assertFalse(np.isclose(float(grid.occupancy_probability.sum()), 1.0))

    def test_pos_is_probability_weighted_detection_not_poc_times_mean_over_all(self) -> None:
        result = SearchProbabilityMetrics.from_cell_probabilities(
            [0.2, 0.3, 0.5],
            [0.0, 0.5, 0.8],
        )

        self.assertAlmostEqual(result.probability_of_containment, 0.8)
        self.assertAlmostEqual(result.probability_of_detection_given_containment, 0.6875)
        self.assertAlmostEqual(result.probability_of_success, 0.55)

    def test_tank_and_tel_profiles_bind_signatures_to_imm5(self) -> None:
        mission = MissionConfig()

        self.assertEqual([profile.name for profile in NOMINAL_TARGET_PROFILES], [
            "Tank-nominal",
            "TEL-nominal",
        ])
        self.assertNotEqual(
            NOMINAL_TARGET_PROFILES[0].signature.name,
            NOMINAL_TARGET_PROFILES[1].signature.name,
        )
        self.assertNotEqual(
            NOMINAL_TARGET_PROFILES[0].imm5_behavior.mode_probabilities,
            NOMINAL_TARGET_PROFILES[1].imm5_behavior.mode_probabilities,
        )
        for profile in NOMINAL_TARGET_PROFILES:
            self.assertEqual(
                profile.motion_spec(max_speed_mps=mission.target_max_speed_mps).motion_model,
                "imm5",
            )

    def test_route_metrics_report_unique_area_redundancy_and_actual_distance(self) -> None:
        mission = MissionConfig(search_radius_m=1_000.0, uav_count=1)
        sensor = SensorSpec.sr_z50()
        route = Route(
            "test",
            0,
            (PathSegment(Point2D(-500.0, 0.0), Point2D(500.0, 0.0), True),),
        )

        metrics = evaluate_routes([route], mission, sensor)

        self.assertGreater(metrics.total_distance_m, metrics.planned_centerline_distance_m)
        self.assertGreater(metrics.unique_area_coverage_ratio, 0.0)
        self.assertLessEqual(metrics.unique_area_coverage_ratio, 1.0)
        self.assertGreaterEqual(metrics.coverage_redundancy_ratio, 0.0)
        self.assertAlmostEqual(metrics.equivalent_sweep_width_m, 400.0)

    def test_probability_mass_coverage_is_distinct_from_area_coverage(self) -> None:
        mission = MissionConfig(search_radius_m=1_000.0, uav_count=1)
        sensor = SensorSpec()
        route = Route(
            "test",
            0,
            (PathSegment(Point2D(-500.0, 0.0), Point2D(500.0, 0.0), True),),
        )

        covered = probability_mass_coverage(
            [route],
            mission,
            sensor,
            [Point2D(0.0, 0.0), Point2D(0.0, 500.0)],
            [0.8, 0.2],
        )

        self.assertAlmostEqual(covered, 0.8)


if __name__ == "__main__":
    unittest.main()
