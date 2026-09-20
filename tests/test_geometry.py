from dataclasses import replace
import math
import unittest

from cpp_search.core.models import MissionConfig, SensorSpec


class SensorGeometryTests(unittest.TestCase):
    def test_default_sensor_geometry(self) -> None:
        sensor = SensorSpec()

        self.assertAlmostEqual(sensor.instantaneous_swath_m, 190.03, places=1)
        self.assertAlmostEqual(sensor.gimbal_envelope_m, 1200.0, places=6)
        self.assertAlmostEqual(sensor.track_spacing_m, 152.02, places=1)
        self.assertAlmostEqual(sensor.eo.horizontal_ifov_urad, 163.62, places=1)
        self.assertAlmostEqual(sensor.gimbal_scan_period_s, 3.2, places=6)
        self.assertAlmostEqual(sensor.reference_observation_interval_s, 3.2)

    def test_sr_z50_catalog_search_geometry(self) -> None:
        sensor = SensorSpec.sr_z50()

        self.assertEqual((sensor.eo.image_width_px, sensor.eo.image_height_px), (1280, 720))
        self.assertEqual((sensor.ir.image_width_px, sensor.ir.image_height_px), (1280, 720))
        self.assertAlmostEqual(sensor.eo.horizontal_ifov_urad, 245.44, places=1)
        self.assertAlmostEqual(sensor.ir.horizontal_ifov_urad, 245.44, places=1)
        self.assertAlmostEqual(sensor.instantaneous_swath_m, 190.03, places=1)
        self.assertAlmostEqual(sensor.gimbal_max_rate_dps, 60.0)
        self.assertAlmostEqual(sensor.gimbal_scan_rate_dps, 30.0)
        self.assertAlmostEqual(sensor.gimbal_scan_period_s, 6.2, places=6)
        self.assertIsNone(sensor.ground_scan_radius_m)
        self.assertIsNotNone(sensor.search_envelope)
        self.assertAlmostEqual(sensor.coverage_half_width_m, 400.0)
        self.assertAlmostEqual(sensor.effective_sweep_width_m, 400.0)
        self.assertAlmostEqual(sensor.search_envelope.seeker_half_width_m, 300.0)
        self.assertAlmostEqual(sensor.search_envelope.weave.half_amplitude_m, 100.0)
        self.assertAlmostEqual(
            sensor.effective_ground_scan_radius_m(18.0),
            400.0,
        )
        # seeker 반폭 300 m -> 지상 300 m, 경사각 26.57도, FOV 절반을 뺀
        # 보어사이트 왕복 17.57도. 4 x 17.57 / 30 dps + 2 x 0.1 s 정정.
        self.assertAlmostEqual(sensor.gimbal_scan_cycle_s(18.0), 2.5420068236)
        self.assertGreater(sensor.actual_search_distance_m(1_000.0), 1_000.0)
        legacy_geometry = replace(sensor, search_envelope=None)
        self.assertAlmostEqual(
            legacy_geometry.effective_observation_interval_s(18.0),
            3.2,
        )
        self.assertAlmostEqual(sensor.reference_observation_interval_s, 3.2)

        full_hd = SensorSpec.sr_z50(eo_full_hd=True)
        self.assertEqual(
            (full_hd.eo.image_width_px, full_hd.eo.image_height_px),
            (1920, 1080),
        )

    def test_default_mission_radius_and_area(self) -> None:
        mission = MissionConfig()

        self.assertAlmostEqual(mission.search_radius_m, 3333.33, places=1)
        self.assertAlmostEqual(
            mission.search_radius_m,
            mission.reachable_radius_at_search_start_m,
            places=6,
        )
        self.assertAlmostEqual(mission.total_area_m2 / 1_000_000.0, 34.91, places=1)
        self.assertAlmostEqual(mission.sector_angle_rad, math.pi / 3.0)
        self.assertAlmostEqual(mission.max_speed_mps * 3.6, 160.0)
        self.assertAlmostEqual(mission.search_speed_mps * 3.6, 100.0)
        self.assertAlmostEqual(mission.target_max_speed_mps * 3.6, 40.0)


if __name__ == "__main__":
    unittest.main()
