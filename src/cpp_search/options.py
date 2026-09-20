"""해석이 끝난 실행 옵션.

``runner``가 `명령행 > config > 기본값` 해석을 마친 뒤 만드는 값 객체다.
챕터 모듈은 명령행을 읽지 않고 이 객체만 본다. 그래서 챕터 실험을
파이썬에서 직접 호출해도(테스트가 그렇게 한다) 동작이 동일하다.

의존: 표준 라이브러리만. ``runner``와 ``chapters/*``가 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RunOptions:
    sample_count: int
    particle_count: int
    episode_count: int
    target_profile: str
    seed: int
    map_error: bool
    communication_loss_probability: float
    communication_latency_slices: int
    mission_time_s: float
    fast: bool = False

    def __post_init__(self) -> None:
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if self.particle_count <= 0:
            raise ValueError("particle_count must be positive")
        if self.episode_count <= 0:
            raise ValueError("episode_count must be positive")
        if self.target_profile not in {"tank", "tel", "both"}:
            raise ValueError("target_profile must be tank, tel, or both")
        if not 0.0 <= self.communication_loss_probability <= 1.0:
            raise ValueError("communication_loss_probability must be in [0, 1]")
        if self.communication_latency_slices < 0:
            raise ValueError("communication_latency_slices must not be negative")
        if self.mission_time_s <= 0.0:
            raise ValueError("mission_time_s must be positive")
