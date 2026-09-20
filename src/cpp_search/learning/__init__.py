"""학습 — MAPPO 로 Stone SPX 를 해석한다.

- ``nets``       : 의존성 없는 합성곱/선형 블록.
- ``mappo``      : CTDE MAPPO 학습규칙 (GAE, 중앙화 비평자, PPO 클리핑).
- ``spx_env``    : ``StoneGridInstance`` 를 다중에이전트 환경으로 노출.
                   행동 = 인접 셀, 에피소드 = SPX 계획지평, 보상 = minimax
                   성형. 점유·분리 복구로 **SPX 와 같은 실행가능집합** 유지.
- ``spx_policy`` : 학습 루프와 SPX 대비 비교. 최적성 주장은 하지 않는다.

이 패키지는 ``planning`` 을 읽는다 (해석 대상이 거기 있다). 반대는 금지다.
"""

from __future__ import annotations

from cpp_search.learning.mappo import MAPPORollout, MAPPOUpdateStats, NumpyMAPPO, collect_mappo_rollout
from cpp_search.learning.spx_env import (
    SPXEnvironmentConfig,
    SPXSearchMetrics,
    StoneSPXEnvironment,
)
from cpp_search.learning.spx_policy import (
    StoneMAPPOResult,
    StoneMAPPOSettings,
    compare_with_spx,
    mappo_path_solution,
    train_stone_mappo,
)

__all__ = [
    "MAPPORollout",
    "MAPPOUpdateStats",
    "NumpyMAPPO",
    "SPXEnvironmentConfig",
    "SPXSearchMetrics",
    "StoneMAPPOResult",
    "StoneMAPPOSettings",
    "StoneSPXEnvironment",
    "collect_mappo_rollout",
    "compare_with_spx",
    "mappo_path_solution",
    "train_stone_mappo",
]
