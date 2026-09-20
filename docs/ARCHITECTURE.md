# 코드 구조

수식별 구현 위치와 자동 추출한 의존 그래프는 [`MATH.md`](MATH.md)에 있다
(`python3 tools/gen_math_doc.py`로 재생성). 각 모듈의 첫 docstring에도 그 파일이
담당하는 수식과 위/아래 의존이 적혀 있다.

## 두 계층

새 코드를 어디에 둘지 헷갈리면 아래 한 줄로 판단한다.

> **"Chapter"나 "계획법"이라는 말이 필요 없으면 도메인 계층, 필요하면 연구 계층.**

```
src/cpp_search/                 ← 도메인 계층: 세계를 기술한다
│
├── models.py                   Point2D, SensorSpec, MissionConfig, PathSegment, Route
├── geometry.py                 극좌표 <-> 월드좌표
├── search_envelope.py          LM weave + EO/IR 주사 = 지지반폭 400 m, W = ∫PD(x)dx
├── motion.py                   IMM5 표적 운동(HALT/저속CV/고속CV/CTRV/기동), 진리 궤적
├── probability.py              표적 사전확률, 극좌표 확률지도
├── belief.py                   로그오즈 증거지도 / 정규화 위치지도 / POC·POD·POS
├── particle_filter.py          SIR PF: 예측·음성관측·재표본추출·roughening·격자투영
├── sensor_observation.py       Tank/TEL 시그니처, EO/IR 성능, 공간 탐지모델(P_D, LOS)
├── terrain.py                  합성 지형: 기동성·은폐·관측성 가중치
├── profiles.py                 Tank/TEL 운용 프로파일 = 시그니처 + IMM5
├── evaluation.py               경로 기하 KPI
├── simulation.py               Monte Carlo 진리 생성 + 탐지시간 -> PerformanceMetrics
│
└── (src/cpp_search 바로 아래)  ← 연구 계층: 연구를 수행한다
    ├── config.py               실험 조건 로딩 (명령행 > config 파일 > 코드 기본값)
    ├── options.py              해석이 끝난 실행 옵션
    ├── truth.py                진리 표적 앙상블 — 거동배정·지형결합·경계처리
    ├── aoi.py                  AOI 포함률 사이징 + step_s / halt boost 민감도
    ├── kpi.py                  공통 KPI 블록 (5개 순위지표 + 이탈률·면적당탐지)
    ├── reliability.py          부트스트랩 구간, 짝지은 비교, MC 설계 충분성
    ├── figures.py              CSV·PNG 산출
    ├── runner.py               CLI -> config -> 챕터 디스패치 -> JSON
    │
    ├── theory/                 경로제약 아래에서 무엇이 최적인가
    │   ├── path_constrained.py 미탐지 질량 재귀(기준 평가함수), hazard 합,
    │   │                       team H1/H2 후퇴지평 (**SPX warm start 전용**)
    │   ├── stone_path.py       Stone(2016) SP1/SPX 절단평면 + 최적성 인증
    │   └── markov.py           입자 belief -> 시공간 Markov 모델, 셀 도달가능성
    │
    ├── planning/               belief 를 축소 문제로, 그 답을 비행경로로
    │   ├── terrain_belief.py   지형결합 IMM5 입자 belief (**공통 입력**)
    │   ├── stone_spx.py        인스턴스 구축 / 절단평면 해법 / 진단 포장
    │   └── routes.py           셀 경로 -> 시간예산이 맞는 평행소인 (**공통 변환**)
    │
    ├── learning/               MAPPO 로 Stone 을 해석한다
    │   ├── nets.py             의존성 없는 합성곱/선형 블록
    │   ├── mappo.py            CTDE MAPPO 학습규칙 (GAE, 중앙비평자, PPO 클리핑)
    │   ├── spx_env.py          StoneGridInstance -> 다중에이전트 환경
    │   └── spx_policy.py       학습 루프 + SPX 대비 비교
    │
    └── chapters/               챕터 하나에 모듈 하나
        ├── ch1_evaluation.py               평가도 / 탐색기 스펙 정의
        ├── ch2_stone_spx.py                SPX + 지형 가중 + 최적성 인증
        ├── ch2_studies.py                  Ch2 연구 측정 (챕터와 같은 코드 경유)
        ├── ch3_mappo_stone.py              MAPPO 로 Stone 해석
        └── ch4_synthetic_terrain.py        합성지형 실비행 결과
```

## 의존 방향

```
chapters  ->  learning  ->  planning  ->  theory  ->  core(도메인)
```

- 도메인 계층(`core/`)은 위 계층을 **절대** import 하지 않는다.
- 이 규칙은 `tests/test_architecture.py` 가 코드로 강제한다.
- `theory`는 위를 모른다.
- **`learning`은 `planning`을 읽는다 — 의도된 것이다.** MAPPO가 해석하는 대상이
  `planning`이 만든 Stone 인스턴스다. 이 import가 없으면 "같은 문제를 푼다"는
  주장이 성립하지 않는다. 그래서 `tests/test_architecture.py`는 이것을 금지가
  아니라 **존재 확인**으로 검사한다.
- 반대 방향은 금지다. `planning`이 `learning`을 읽기 시작하면 계획법이 학습을
  알게 되고, 두 계획법을 같은 인스턴스로 비교한다는 전제가 무너진다.
- `runner`만 예외적으로 전 계층을 본다 (챕터를 디스패치해야 하므로).

`tests/test_architecture.py`가 이 규칙을 코드로 검사한다.

## 비교가 성립하는 지점

v0.4의 핵심 주장은 "Stone SPX와 MAPPO를 같은 자로 쟀다"는 것이다. 그것을
보장하는 장치는 코드 세 군데뿐이다.

| 장치 | 위치 | 무엇을 보장하나 |
|---|---|---|
| `StoneGridInstance` | `planning/stone_spx.py` | 격자·슬라이스·belief·hazard·출발셀·예약반경이 하나의 객체다. 두 계획법이 **같은 객체**를 받는다 |
| `common_input_fingerprint` | 같은 파일 | 그 객체의 SHA-256. 두 결과에 같은 값이 박히지 않으면 비교 주장은 무효다. Chapter 4는 어긋나면 `AssertionError`로 **실패한다** |
| `instance.score_paths` | 같은 파일 | 채점 함수가 하나다. SPX든 MAPPO든 이 함수만 통과한다 |

경로변환(`planning/routes.local_sweep`)도 공통이라 KPI 차이가 소인 패턴 차이로
새지 않는다.

## 어디에 무엇을 추가하나

| 하고 싶은 일 | 손댈 곳 |
|---|---|
| 실험 조건(격자, seed, 학습 하이퍼파라미터) 변경 | `config/chapter*.json` — 코드는 건드리지 않는다 |
| 초기 belief 종류/오차 변경 | `config/common_experiment.json`의 `aoi.cue_ring.kind`(moving-ring/tp-centered)·`sigma_m` — `truth.cue_prior`가 읽는다 |
| 표적별 계획 hazard 켜기/끄기 | `stone_spx.target_hazard_from_signature` — 켜면 `chapter2.signature_hazard_multipliers`가 Ch1 소인폭 표에서 유도 |
| 비행/표적 시간(현재 480 s) 변경 | `config/common_experiment.json`의 `mission.mission_time_s` + `tools/size_aoi.py --horizon-s <s> --apply`로 AOI 재측정 (둘을 같이 바꾸지 않으면 포함률 99% 주장이 깨진다) |
| 새 격자 조건 추가 | `config`의 `grid_conditions` 배열 |
| 지형 가중이 들어가는 지점 변경 | `planning/terrain_belief.py`의 `BeliefTerrainCoupling` (사전·전이. hazard 관측성은 합성지형에서 항상 1.0) |
| SPX 제약 추가 (점유·분리·양립불가) | `theory/stone_path.py`의 `stone_spx_cutting_plane` |
| MAPPO 관측·보상 변경 | `learning/spx_env.py` |
| 새 계획법 추가 | `StoneGridInstance`를 받아 `StoneGridPathSolution`을 돌려주는 함수 하나 + `route_plan_from_solution` |
| 지형 대조군의 solver 한도만 바꾸기 | `config`의 `terrain_unweighted_overrides` |
| 새 KPI 추가 | `src/cpp_search/kpi.py` 의 `mission_kpi` + `KPI_KEYS` |
| 새 그림 추가 | `src/cpp_search/figures.py` 의 `_chapterN_artifacts` |

**새 계획법을 추가할 때 `StoneGridInstance`를 우회하면 안 된다.** 우회하면 그
계획법은 다른 문제를 푼 것이고, 결과표에 나란히 실을 수 없다.

## v0.3에서 무엇을 뺐나

v0.4는 v0.3(`Coverage-Path-Planning_v0.3-main`)에서 핵심 네 가지만 남긴 것이다.
v0.3은 그대로 보존되어 있다.

| 뺀 것 | 이유 |
|---|---|
| Koopman/Stone 정적 노력배분, FAB 시간축 최적화 | 경로제약 SPX가 같은 질문에 더 강한 답을 준다. 배분 -> 경로 변환 단계가 필요 없어짐 |
| SAROPS accordion 기준선, Dell H1/H2 비교군 | 비교군을 SPX와 MAPPO 둘로 좁혔다. H1/H2는 SPX warm start로만 남았다 |
| ASOC 페로몬 협동, MAPPO 논문 도메인 재현 | MAPPO를 Stone 인스턴스에 붙이는 쪽으로 옮겼다 |
| 실제 지형 (Copernicus DEM / ESA WorldCover) | 합성지형으로 범위를 좁혔다. 지형 원자료 수집·교정이 별도 과제 |
| 개발/확인 seed 프로토콜, 챕터 6a-1/6a-2/6b 3단 확인실험 | 모델 선택 단계가 없어졌다 (비교군이 둘뿐) |

v0.3에서 **고친 것** (2026-09-13 정합성 검토, 전부 독립 수치검증으로 확정):

| 결함 | 위치 | 증상 | 검사 |
|---|---|---|---|
| 양립불가 이동(4.54)이 도달 불가 arc 를 오류로 올림 | `theory/stone_path.py` | v0.3 테스트 6건 실패 | `test_stone_path.IncompatibleMoveValidationTests` |
| 국소개선 켜면 master 자기 점에 접선이 없어 하한 정체 | `theory/stone_path.py` | passes=2 에서 200회 반복 후 gap 6% | `test_review_regressions.CuttingPlaneLocalImprovementTests` |
| dual bound 비유한 시 incumbent 를 하한으로 사용 | `theory/stone_path.py` | 시간제한 master 에서 거짓 인증 가능 | (경로 방어) |
| 정사각 격자 `cell_of` 의 `int()` 절단 | `planning/stone_spx.py` | 서·남쪽 바깥 한 셀 폭이 가장자리 셀로 배정 | `test_review_regressions.SquareGridCellOfTests` |
| critic clip 이 물린 표본의 도함수를 0 이 아닌 값으로 | `learning/mappo.py` | 유한차분 대비 5~14배 | `test_review_regressions.MappoGradientTests` |
| 출력 가중치 갱신 **뒤** 그 가중치로 은닉층 역전파 | `learning/mappo.py` | 큰 스텝에서 0.4배 어긋남 | 같은 테스트 |

첫 항목 상세:
도달 불가 arc를 `KeyError -> ValueError`로 올려서 모든 셀을 source로 열거하는
호출부가 SPX를 아예 못 돌렸다 (v0.3 테스트 6건이 이 하나 때문에 실패한다).
도달 불가 arc는 공허한 제약이므로 건너뛰고, 호출자가 문제를 잘못 기술한 경우만
실패한다. 회귀 검사는 `tests/test_stone_path.py`의
`IncompatibleMoveValidationTests`에 있다.
