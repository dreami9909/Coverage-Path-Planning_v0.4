"""챕터 하나에 모듈 하나. v0.4 는 핵심 연구 네 개다.

각 모듈은 ``run(config, options)`` 하나를 노출하고 JSON 으로 직렬화 가능한
결과 dict 를 돌려준다. 실험의 주인은 챕터 모듈이고,
``cpp_search.runner`` 는 명령행 해석 · config 로딩 · 결과 저장만 한다.

    1  평가도 정의 / 탐색기 스펙 정의        ch1_evaluation
    2  Stone(2016) SPX + 지형 가중          ch2_stone_spx
    3  MAPPO 로 Stone 해석                  ch3_mappo_stone
    4  합성지형 대입 결과                    ch4_synthetic_terrain

Chapter 3·4 는 Chapter 2 의 인스턴스 구축 함수를 **그대로 import 해서** 쓴다.
복사하면 조건이 소리 없이 갈리기 때문이다.
"""

from __future__ import annotations

from cpp_search.chapters import (
    ch1_evaluation,
    ch2_stone_spx,
    ch3_mappo_stone,
    ch4_synthetic_terrain,
)

CHAPTER_MODULES = {
    "1": ch1_evaluation,
    "2": ch2_stone_spx,
    "3": ch3_mappo_stone,
    "4": ch4_synthetic_terrain,
}

__all__ = ["CHAPTER_MODULES"]
