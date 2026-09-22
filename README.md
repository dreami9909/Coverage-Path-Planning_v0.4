# Cooperative Target Search — Stone SPX + MAPPO (v0.4)

실종자를 6대의 무인기로 찾는 문제에서, **경로제약 다중에이전트 탐색
최적화**를 두 방법으로 풀고 같은 자로 재는 연구 코드다.

    Stone, Royset & Washburn (2016) 4장 SPX     최적성 인증 O, 확장 X
    MAPPO 학습 정책                              최적성 인증 X, 확장 O

이 두 줄이 v0.4의 전부다. v0.3의 챕터 0-8 중 이 주장에 필요한 것만 남겼고,
v0.3은 `Coverage-Path-Planning_v0.3-main/`에 **그대로 보존되어 있다**.

## 연구 구성

| Chapter | 묻는 것 | 주요 산출물 |
|---|---|---|
| 1 | 두 계획법을 비교하기 전에 무엇을 동일하게 고정해야 하는가 | 소인폭 표 W, 탐색기 스펙, 표적 프로파일, P_D/P_F 증거 계약, MC 설계, KPI 정의, **자원 상한** |
| 2 | 지형 가중이 최적 경로를 바꾸는가 / 최적성을 어디까지 증명할 수 있는가 | 격자별 **최적성 인증폭**, 인증 관문 판정, 지형 가중 짝지은 효과 + 95% CI |
| 3 | 학습된 정책이 Stone 문제를 대신 풀 수 있는가 | 학습곡선, **학습효과**(학습-미학습), **도달률**(MAPPO/SPX), 같은 인스턴스 짝지은 비교, 실행시간 |
| 4 | 그 계획을 실제로 비행시키면 무엇이 남는가 | 합성지형·대조군 지형 실비행 **탐지율 + 평균탐지시간** + 95% CI, **계획모형 − 실비행 차이**(model − flown) |

## 시나리오 확정값 (2026-09-13)

| 항목 | 값 | 근거 |
|---|---|---|
| 비행·표적 이동 시간 | **480 s** | 운용 요구 |
| 비행속도 | **160 kph** (탐색 구간은 weave로 중심선 39.0 m/s) | 기체 성능 |
| 탐색자 | **정상기 6대** (동일 → 등급 집계로 순열 대칭 제거) | 운용 구성 |
| AOI | **TP 중심 반경 4,784 m** = 480 s 순변위 99% (TEL 결정) | `tools/size_aoi.py --rule net-displacement` |
| 초기 belief | **tp-centered, σ = 500 m** (TP 정보 오차) | 선언 가정; 이 오차로 실제 포함률 97~98%, 이탈은 `escape_rate`로 보고 |
| 계획 hazard | **표적별 W** (Tank 0.874 · TEL 0.867 × 기하 400 m) | 실비행과 같은 공간 탐지모델 |
| polar 격자 | 600 m × 방위 N구간, **24슬라이스(20 s)**, 입자 3,000 | 방위 N은 민감도 실험으로 확정 |

이전 기본값(300 s · 100 kph · 정상4+저하2 · cue ring 6,940 m · 기하 W 공통)으로 낸
결과와 **비교하지 않는다** — 조건이 다르다.

## 시간축과 평가지표 (2026-09-13 변경)

- **표적이 움직이는 시간 = 비행 시간 = 8분 (480 s).** AOI 반경은 TP 기준
  순변위(|x(T)−x(0)|)의 99% 분위수로 잰다 (`tools/size_aoi.py --rule
  net-displacement --apply`, 결과는 `config/common_experiment.json`의 `aoi.measured`).
- **헤드라인 지표는 둘이다.** 비행 전체에서 탐지된 표적 비율(**탐지율**,
  `detection_probability_within_limit` — 한계시간이 곧 비행 종료라 별도의
  5분 절단은 없다)과, 탐지된 표적만의 **평균 탐지시간**
  (`conditional_mean_detection_time_s`). 둘은 짝으로 읽는다 — 하나만 보면
  "빨리 조금"과 "늦게 많이"를 구별할 수 없다. 미탐지를 비행종료로 절단한
  평균(`restricted_mean_detection_time_s`)은 진단으로만 남긴다.

## 핵심 주장과 그 한계

**주장한다.**

- 축소 격자에서 Stone SPX는 최적해와 **인증**을 같이 준다. 절단평면의 접선이
  미탐지확률의 유효 하한을, 정수 master의 경로가 상한을 준다.
- 그 격자에서 MAPPO는 **증명된 최적해의 몇 %까지** 붙는지 말할 수 있다.
- 운용 해상도(240셀·5슬라이스·6기)에서 SPX의 전역 하한 증명은 무너진다.
  v0.3 Ch6 감사에서 13분 27초를 써도 상대 PND gap 15.18%로 사전선언 1%를
  실패했다. 그 격자에서도 MAPPO는 실행가능한 계획을 낸다.
- 지형 가중을 계획에 넣은 것과 뺀 것의 차이를 **같은 seed로 짝지어** 뺀 값.

**주장하지 않는다.**

- MAPPO의 최적성. 하한을 증명하지 않으므로 결과 JSON에 `null`로 남는다.
  SPX 인증이 열린 격자에서는 **두 값 모두** 진짜 최적해보다 낮을 수 있다.
- 심층 MAPPO 구현과의 절대 성능 비교. 이 구현은 numpy만으로 돌고 은닉층이
  하나(tanh)라, 재현하는 것은 MAPPO의 **학습 규칙**(GAE, 중앙화 비평자, PPO
  클리핑)이고 원 논문의 **신경망 용량**이 아니다. `policy_capacity` 블록이 이
  사실을 결과에 같이 적는다.
- 실제 지형에서의 성능. v0.4는 합성지형만 쓴다.
- **Chapter 2의 최적성 인증이 실비행 성능의 인증이라는 것.** 인증은 계획모형
  안에서의 인증이다. 실측된 `model_minus_flown`은 약 **−0.03 ~ −0.05**로,
  계획모형이 실제보다 **낮게** 잡는다 (셀 방문 하나를 hazard 하나로 세는
  이산화가 셀 안의 연속 소인이 덮는 면적을 과소평가한다). 부호가 음수라
  과대주장이 되지는 않지만, 계획모형에서 최적인 경로가 실비행 척도에서도
  최적이라는 보장은 없다.
- 지형 가중치의 실측 근거. `terrain.py`의 통행성·은폐·관측성 가중은 **미교정
  운용 가정**이다.

## 비교가 성립하는 근거

두 계획법을 나란히 싣는 것을 정당화하는 장치는 코드 세 군데뿐이다.

1. **`StoneGridInstance`** — 격자·시간 슬라이스·지형결합 belief·탐색자 hazard·
   출발셀·예약반경·점유한도가 **하나의 객체**다. SPX와 MAPPO가 같은 객체를
   받는다.
2. **`common_input_fingerprint`** — 그 객체의 SHA-256. 두 결과에 같은 값이
   박히지 않으면 비교 주장은 무효다. Chapter 4는 어긋나면 `AssertionError`로
   실패한다.
3. **`instance.score_paths`** — 채점 함수가 하나다. 어느 계획법이든 이 함수만
   통과한다. `tests/test_spx_env.py`가 MAPPO 환경의 온라인 미탐지 재귀와 이
   함수의 값이 같은지(소수 12자리) 검사한다.

경로변환(`planning/routes.local_sweep`)도 공통이라, Chapter 4의 KPI 차이가
소인 패턴 차이로 새지 않는다.

MAPPO는 SPX의 실행가능집합을 **넘지 않는다**. 행동 후보가 인접 셀이고, 점유·
분리 위반은 환경이 순서대로 복구한다. 고정 K개 가지치기 때문에 오히려
**부분집합**이고, 그 대가는 `action_candidate_coverage`로 보고된다. 인증이
닫힌 격자에서 MAPPO가 SPX를 넘으면 그것은 성과가 아니라 버그이므로,
`tests/test_spx_policy.py`가 그 경우를 실패로 잡는다.

## 실행

패키지는 `src/cpp_search/`에 있는 src-layout이다. `run_research.py`가 실행 시
`src`를 `sys.path`에 직접 넣으므로 **환경변수 없이 그냥 실행하면 된다.**
IDE의 실행/디버그 버튼도 그대로 동작한다.

```bash
python3 run_research.py --chapter 1
python3 run_research.py --chapter 2
python3 run_research.py --chapter 3
python3 run_research.py --chapter 4
```

전부 한 번에 돌리고 요약표 + `results/summary.json` 생성:

```bash
python3 run_research.py --chapter all
```

빠른 확인이 필요할 때만 명령행으로 조건을 낮춘다:

```bash
python3 run_research.py --chapter 3 --seed-count 1 --episodes 10
```

**조건을 낮춘 결과는 성능 주장에 쓸 수 없다.** 결과에
`reliability.status = "insufficient-design"`이 기록된다. Chapter 4의 본설정은
독립 seed 20개와 seed당 표적 궤적 1,000개다.

### 실행시간 (Apple Silicon, 단일 프로세스 기준)

| Chapter | 조건 | 대략 |
|---|---|---:|
| 1 | seed 없음 | 10초 |
| 2 | 20 seed × 3 격자 (대조군 포함) | 35분 |
| 3 | 20 seed × 2 격자 (SPX + MAPPO 학습) | 26분 |
| 4 | 20 seed × 2 지형계열 × 1,000 표본 | 13분 |
| all | | **약 1시간 15분** (300 s 기준 실측; 480 s 는 더 걸린다) |

시간을 쓰는 곳은 거의 전부 **Chapter 2·3의 정수 master**다. Monte Carlo
평가(Chapter 4)는 오히려 가장 싸다. 급하면 `--seed-count`로 계획 seed를
줄이는 것이 가장 효과가 크다.

해석 순서는 `명령행 > config 파일 > 코드 기본값`이다. CLI 인자의 argparse
기본값이 모두 `None`이라, 플래그를 생략하면 config가 반드시 이긴다. 즉 플래그
없이 실행하면 `config/chapter*.json`에 적힌 조건이 그대로 재현된다.

### 의존성

```bash
python3 -m pip install numpy scipy matplotlib
```

SPX의 정수 master는 기본적으로 `scipy.optimize.milp`(HiGHS)로 푼다. 선택
의존성 두 개는 대체 백엔드다 — 없어도 전부 돌아간다.

- `highspy` — 지속(persistent) master, 정확 생존 MILP
- `pyscipopt` — `master_backend: "scip"`

## 조건은 전부 config에 있다

```
config/
├── common_experiment.json              네 챕터가 공유하는 조건 (단일 출처)
├── chapter1_evaluation_contract.json   Chapter 1 이 무엇을 보고하는지
├── chapter2_stone_spx_terrain.json     격자 사다리, 인증 관문, 지형 대조군
├── chapter3_mappo_stone.json           학습 하이퍼파라미터, 환경 설정
└── chapter4_synthetic_terrain.json     평가 격자, 지형 계열, 표적 계층 가중
```

코드는 조건값을 박지 않는다. `config/`에 없는 값은 어느 챕터도 읽지 않는다
(`tests/test_config_loader.py`가 읽히지 않는 config 파일이 남아 있는 것도
실패로 잡는다).

### 격자 사다리

Chapter 2가 최적성 인증이 무너지는 지점을 찾는다.

| 격자 | 셀 | 슬라이스 | 인증 | 비고 |
|---|---:|---:|---|---|
| `reduced-4x4` | 16 | 3 | 닫힘 | Chapter 3의 도달률, Chapter 4의 실비행은 이 격자에서만 |
| `reduced-5x5` | 25 | 3 | 닫힘 | 격자를 올렸을 때 인증폭·반복수가 어떻게 움직이는지 |
| `operational-polar` | 240 | 5 | **열림** | 운용 해상도. 인증 붕괴를 재현하고, Chapter 3이 학습 정책으로 해석 |

발사 고리는 `0.45R`이다. 더 안쪽이면 6기가 축소격자의 중앙 셀 주변에 몰려
점유한도 1과 예약이 첫 슬라이스부터 과포화된다 (9셀 격자에서 실제로
`InfeasiblePathError`가 났다).

**인증 관문은 사전에 선언한다** (`certificate.required_relative_gap = 0.01`).
결과를 보고 이 값을 올리면 인증 주장이 무효가 된다.

### 지형 가중이 들어가는 지점은 둘이다

지형은 **표적 belief** 에만 들어간다 — 격자 사전질량(통행성·은폐)과 입자필터
전이커널(회랑 추종·off-road·정지확률). 셀별 hazard 에는 들어가지 않는다:
합성지형의 `observability_weight` 는 선언된 모델 범위상 항상 1.0 이다 (지형은
탐색자가 *무엇을 믿는가*를 바꾸고 *얼마나 잘 보는가*는 바꾸지 않는다). 그래서
`terrain_weighting=false` 대조군과의 차이는 순수하게 belief 차이다.

### 지형 대조군의 인증은 열린다 (실행 중 확인된 사실)

지형 가중을 끄면 belief 가 cue ring 사전분포 그대로 — **회전 대칭**이 된다.
대칭해가 폭발하고 정수 master 가 그 대칭을 깨지 못한다 — 16셀 격자에서조차
수 분 안에 끝나지 않았다. 가중을 켠 같은 격자는 1초 안에 gap 0 으로 닫힌다.

그래서 대조군에만 적용할 한도를 config 가 `terrain_unweighted_overrides` 로
따로 선언한다. 한도를 줄인 결과는 인증이 열린 채 나오고, **그 사실이 결과에
남는다**:

- `terrain_unweighted_optimality_certificate.passed = false`
- `terrain_weighting_effect.both_arms_certified = false`
- `terrain_weighting_effect.reading` — 이 차이는 두 최적해의 차이가 아니라
  **인증된 해와 미인증 해의 차이**이므로 지형 가중 효과의 **하한**으로만
  읽어야 한다는 문장

이것 자체가 결과다. "지형 정보가 성능을 올린다"보다 약하지만 더 구체적인
주장이 나온다 — **지형 가중은 문제를 쉽게 만든다.** belief 에 비대칭성을 넣어
정수 계획법이 붙잡을 구조를 준다.

### 계획모형이 두 표적을 어떻게 구별하는가

SPX 의 셀 hazard 는 **기하 등가 탐색폭 W = 400 m** 로 보정되고
(`hazard = W·v·Δt / A_cell`), 두 표적의 `hazard_multiplier` 는 1.0 이다.
즉 계획모형 안에서 Tank 와 TEL 은 **운동(IMM5 거동)으로만** 다르고 탐지
난이도로는 같다. 표적별 실측 W (Chapter 1: Tank 348 m 등)는 실비행 평가의
공간 탐지모델에만 들어간다. Chapter 4 의 `model_minus_flown` 은 이 간극도
포함한다.

## 검증

```bash
python3 -m pytest
```

주요 검사:

| 파일 | 무엇을 지키나 |
|---|---|
| `test_architecture.py` | 계층 의존 방향. `learning -> planning`은 **존재 확인**, 역방향은 금지 |
| `test_stone_path.py` | SP1/SPX가 완전열거 최적해와 일치, 인증폭 0, 양립불가 이동(4.54) 처리 회귀 |
| `test_spx_env.py` | MAPPO 환경의 온라인 재귀 = 공식 채점 (12자리), 경로의 인접성·점유·분리 |
| `test_spx_policy.py` | 학습이 실제로 일어남, MAPPO가 **인증된 최적해를 넘지 않음**, 하한을 주장하지 않음 |
| `test_config_loader.py` | 해석 순서, Chapter 2·3이 같은 격자를 선언, 고아 config 없음 |
| `test_experiments_smoke.py` | 네 챕터가 축소 조건으로 끝까지 돌고 선언한 키를 내보냄 |

## 문서

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — 계층 구조, 의존 규칙,
  "어디에 무엇을 추가하나", v0.3에서 뺀 것과 고친 것
- [`docs/MATH.md`](docs/MATH.md) — 수식 -> 구현 위치(파일:줄) 색인 77개 +
  자동 추출 의존 그래프. `python3 tools/gen_math_doc.py`로 재생성

## 문제 해결

**`ModuleNotFoundError: No module named 'cpp_search'`**

`src`가 `sys.path`에 없을 때 나온다.

- `run_research.py`를 실행했는데도 난다면 저장소의
  `src/cpp_search/__init__.py`가 없거나 파일이 저장소 밖으로 복사된 경우다.
  저장소 루트에서 실행한다.
- 다른 스크립트나 노트북에서 직접 `import cpp_search`를 하려면
  `pip install -e .`로 설치하거나, IDE 실행 구성의 작업 디렉터리를 저장소
  루트로 두고 `PYTHONPATH`에 `src`를 추가한다.

**`InfeasiblePathError: every reachable first move is blocked`**

예약(분리반경 + 점유한도)이 한 슬라이스의 도달 가능한 첫 수를 전부 막았다.
셀 수가 무인기 수에 비해 너무 적거나 발사 고리가 너무 안쪽이다. 격자를 키우거나
`initial_position_ring_ratio`를 올린다.

**SPX가 끝나지 않는다**

`master_time_limit_s`를 선언하지 않은 격자에서 정수 master가 멈출 수 있다.
`config`의 모든 격자 조건에 이 값을 넣는다 — 없으면 MILP가 무한정 돈다.
