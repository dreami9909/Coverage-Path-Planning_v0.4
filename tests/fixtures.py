"""테스트가 공유하는 작은 Stone 인스턴스.

실제 config 를 거치지 않고 도메인 객체를 직접 만든다. config 로딩까지 엮으면
환경 테스트가 실패했을 때 원인이 환경인지 config 인지 구별되지 않는다.
"""

from __future__ import annotations

from cpp_search.core.models import MissionConfig, Point2D, SensorSpec
from cpp_search.core.probability import TargetPrior
from cpp_search.core.profiles import NOMINAL_TARGET_PROFILES
from cpp_search.planning.stone_spx import (
    StoneGridInstance,
    StoneSPXRouteConfig,
    StoneSPXSearcherClass,
    StoneSPXTargetSpec,
    build_stone_grid_instance,
)
from cpp_search.core.terrain import build_synthetic_terrain


def small_instance(
    *,
    uav_count: int = 3,
    grid_width: int = 4,
    grid_height: int = 4,
    time_slice_count: int = 3,
    seed: int = 20_260_913,
    reservation_separation_m: float = 0.0,
    terrain_weighting: bool = True,
    occupancy_limit: int = 1,
) -> StoneGridInstance:
    mission = MissionConfig(
        center=Point2D(0.0, 0.0), search_radius_m=5_000.0, uav_count=uav_count
    )
    sensor = SensorSpec.sr_z50()
    terrain = build_synthetic_terrain(mission.search_radius_m, seed=seed)
    profiles = list(NOMINAL_TARGET_PROFILES)[:2]
    positions = tuple(
        Point2D(-2_000.0 + 2_000.0 * index, -1_200.0 + 600.0 * index)
        for index in range(uav_count)
    )
    return build_stone_grid_instance(
        mission,
        sensor,
        prior=TargetPrior(kind="moving-ring"),
        terrain=terrain,
        targets=tuple(StoneSPXTargetSpec(profile, 1.0) for profile in profiles),
        searcher_classes=(
            StoneSPXSearcherClass("common-sensor", uav_count, 1.0),
        ),
        initial_positions=positions,
        planning_time_s=480.0,
        initial_belief_delay_s=0.0,
        seed=seed,
        config=StoneSPXRouteConfig(
            grid_width=grid_width,
            grid_height=grid_height,
            time_slice_count=time_slice_count,
            particle_count=300,
            hazard_calibration="effective-sweep-width",
            occupancy_limit=occupancy_limit,
            reservation_separation_m=reservation_separation_m,
            max_iterations=20,
            master_time_limit_s=20.0,
            terrain_weighting=terrain_weighting,
        ),
    )
