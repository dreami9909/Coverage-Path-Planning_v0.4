# Chapter 2 유효 결과

조건: 이송 480 s(750 kph, 100 km) + 탐색 600 s(160 kph + weave) · UAV 6대 ·
케이스별 minimax(시그니처 1종 × IMM5 거동 3종) · AOI 외접 정사각 격자 ·
T = H*vc/c · 인접은 도달반경 제한 · hazard 는 이동 경로에 분배.

재현: `src/cpp_search/chapters/ch2_studies.py`
조건 선언: `config/chapter2_stone_spx_terrain.json`

## 인증된 결과

| 케이스 | 격자 | 셀 | PD | 최악 gap |
|---|---|---|---|---|
| TEL | 10x10 | 1,990 m | 0.7328 | 0.903% |
| Tank | 8x8 | 2,221 m | 0.7465 | 0.977% |

seed 3개 전부 사전 선언 기준 1% 이내. **인용할 때 seed 수를 밝힐 것** —
최악 gap 이 기준에 바짝 붙어 있다.

## 파일별

- `50x50-unbounded-attempt.json` — 셀=탐지폭 격자, 24시간 미반환 기록
- `case-grid-sweep-large.json` — 16x16 (60s×60). 인증 붕괴 쪽 끝
- `case-grid-sweep-small.json` — TEL 8x8 (60s×60)
- `certificate-wall.json` — 10x10·12x12, master 60s×60. 배분 효과의 대조군
- `ch2-final-two.json` — Tank 8x8 (60s×60) + 50x50 시도
- `ch2-seed-replication.json` — 인증 주장의 근거. TEL 10x10 / Tank 8x8 × seed 3개, master 300s×20
- `particle-noise.json` — 입자 8배 증가에도 PD 무감 — 추정잡음 배제
- `per-case-aoi.json` — 케이스별 AOI 사이징
- `tank-8x8-300s.json` — Tank 8x8 예산 배분 대조 (300s×20). 같은 fingerprint 의 60s 실행과 짝
- `wall-10x10-bigger-master.json` — 10x10 을 300s×20 로. 60s 실행과 짝
