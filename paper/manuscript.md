# 다중 무인기 협동탐색 경로계획의 최적성 인증

Optimality Certification for Cooperative Multi-UAV Search Path Planning

---

## ABSTRACT

Cooperative search planning for multiple unmanned aerial vehicles is normally
judged by the detection probability that a plan attains. A detection
probability alone, however, does not say how far the plan is from the best plan
the model admits, so an improvement of a few points cannot be separated from
the residual room that any planner still leaves. This paper reports a planning
framework that returns, with every plan, a certificate bounding its distance
from optimality. The planner states the multi-target, path-constrained search
allocation problem in the form given by Stone et al., and solves it by an
outer-approximation cutting-plane scheme whose master problem is a
mixed-integer linear program. Every iteration yields a valid lower bound on the
attainable non-detection probability and an implementable plan that furnishes
an upper bound; their relative difference is the optimality gap. The machinery
is verified against exhaustive enumeration on small instances, where the
certified bounds coincide with the true optimum, and against Monte-Carlo
simulation of the target motion, where the objective agrees within sampling
error. Three structural results follow. First, the discretization is not free:
once the searcher kinematics are fixed, the number of time slices is determined
by the cell size, and choosing the two independently makes the reachability
model inconsistent with the coverage model. Second, the computational budget
must be spent on master solve time rather than on cutting-plane iterations.
Third, the plan is bit-reproducible whereas the certificate tightness is not,
because a time-limited master returns a load-dependent dual bound; the
certificate is therefore conservative but never wrong.

**Key Words** : Cooperative Search, Search Theory, Path-Constrained Search
Allocation, Cutting-Plane Method, Mixed-Integer Programming, Optimality
Certificate, Unmanned Aerial Vehicle

---

## 1. 서론

다수 무인기를 이용한 협동탐색 경로계획은 통상 계획이 달성한 **탐지확률**로
평가된다. 그러나 탐지확률만으로는 그 계획이 모형이 허용하는 최선에서 얼마나
떨어져 있는지 알 수 없다. 어떤 계획법이 기존 대비 탐지확률을 몇 점 올렸다고
할 때, 그 개선이 남은 여지의 전부인지 일부인지 구분되지 않기 때문이다.
남은 여지를 모르면 계획법끼리의 비교는 상대적일 뿐이고, "이보다 나은 계획은
없다"는 진술은 불가능하다.

이 문제를 해결하는 방법은 계획과 함께 **최적성 인증(optimality
certificate)** 을 내는 것이다. 즉 달성한 값에 대한 상계와 도달 가능한 최선에
대한 하계를 동시에 제시하여, 둘 사이의 상대적 차이를 **최적성 간격
(optimality gap)** 으로 보고한다. 간격이 1 % 이내이면 그 계획은 최적해 대비
1 % 이내임이 증명된 것이고, 추가 탐색으로 얻을 수 있는 개선의 상한도 1 %
이다.

경로 제약이 있는 이동표적 탐색 배분 문제는 일반적으로 NP-hard 이며[5],
분지한계법 기반의 정확해법이 제시되어 왔다[4]. Stone 등[1]은 이 문제군을
체계적으로 정리하고, 다표적·이종 탐색자·경로 제약을 포괄하는 정식화와 그에
대한 절단평면 해법을 제시하였다. 본 연구는 이 정식화를 기반으로 하되,
운용상 필요한 두 가지를 추가한다. 첫째, 소인(掃引) 효과를 도착 셀 하나가
아니라 **이동 경로가 지나간 셀 전체에 분배**한다. 둘째, 표적의 기동 양상이
알려지지 않은 조건을 반영하여 표적군별 **최소최대(minimax)** 목적을 사용한다.

본 논문의 기여는 다음과 같다.

1) 경로 분배 소인 모형을 포함한 다표적 최소최대 탐색 배분 문제를 정식화하고,
   이를 외부 근사 절단평면법으로 해결하여 최적성 인증을 산출하는 계획
   프레임워크를 제시한다.
2) 구현의 정확성을 완전열거 및 몬테카를로 모의와 대조하여 검증한다.
3) 셀 크기·시간 슬라이스 수·계산예산·최적성 간격 사이의 관계를 실험으로
   규명하고, 이산화가 자유 변수가 아님을 보인다.

## 2. 문제 정식화

### 2.1 탐색 이론 배경

측방 탐지확률 곡선 `P_D(x)` 에 대한 **탐지폭(sweep width)** `W` 는 Koopman[2]
의 정의에 따라 다음과 같다.

    W = ∫ P_D(x) dx                                        (1)

면적 `A` 인 영역을 소인면적 `S` 만큼 훑었을 때, 표적 위치가 그 영역 안에서
균일하다는 무작위탐색 가정 하의 탐지확률은

    q = 1 - exp(-a),     a = S / A                          (2)

이다[3]. 여기서 `a` 는 hazard 이며 확률이 아니다. 같은 영역을 여러 탐색자가
훑는 경우 확률은 더할 수 없지만 hazard 는 더할 수 있다.

    a_total = Σ_j a_j                                       (3)

이 가산성이 이후 정식화에서 목적함수를 **결정변수에 대해 선형인 양**으로
유지하는 근거가 된다.

### 2.2 이산화와 경로 제약

관심구역(AOI)을 `N x N` 정사각 격자로 나누고 탐색 단계를 `T` 개의 시간
슬라이스로 나눈다. 격자는 AOI 원에 **외접**하도록 잡는다. 내접 격자는 원의
약 64 % 만 덮으므로 AOI 안에 있는 표적 질량의 일부가 격자 밖으로 흘러
목적함수가 조용히 달라진다.

슬라이스 길이를 `Δ = H / T` (`H` 는 탐색 단계 지속시간), 셀 한 변을 `c`,
탐색자의 순항속도를 `v` 라 할 때, 한 슬라이스에 도달 가능한 거리는 `v Δ` 다.
이 값이 `c` 보다 작으면 어떤 탐색자도 자기 셀을 떠날 수 없고, `√2 c` 보다
크면 대각 이동이 허용된다. 따라서 **셀 크기와 슬라이스 수를 독립적으로 고를
수 없다.** 본 연구는 "한 슬라이스에 셀 한 변을 전진한다"는 조건

    T = H · v_c / c                                         (4)

로 `T` 를 셀 크기에서 유도한다. 여기서 `v_c` 는 weave 비행을 감안한 중심선
전진속도다. 이 규칙 하에서 도달거리 대 셀 크기의 비는 1.09 ~ 1.12 로, 대각
이동에 필요한 1.414 에 미치지 못한다. 따라서 한 셀의 후속 셀은 **최대 5 개**
(제자리 및 상하좌우)이며, 격자 가장자리에서는 4 개, 모서리에서는 3 개다.

### 2.3 경로 분배 소인 모형

탐색자가 슬라이스 `t` 에 셀 `i` 에서 `k` 로 이동할 때, 센서는 이동 중에도
작동한다. 따라서 소인 효과를 도착 셀에만 부여하는 것은 실제와 다르다. 본
연구는 hazard 를 **지나간 셀 전체에 길이에 비례하여 분배**한다. 셀 `j` 안에서
지나간 구간 길이를 `ℓ_j(i→k)`, 이동 후 해당 셀에 남는 시간을 `τ(i→k)` 라 하면

    α_j(t, i→k) = W · [ ℓ_j(i→k) + v · τ(i→k) · 1{j=k} ] / A     (5)

이다. 단위시간에 지면에 축적되는 hazard 총량은

    ∫∫ γ(x, z) dx dz = W · v                                (6)

로 **경로의 모양과 무관**하다. 측방 적분 `W` 는 진행 방향에 대한 양이므로
경로가 휘어도 변하지 않기 때문이다. 따라서 이동 구간과 체류 구간 모두
실제 비행속도 `v` 를 쓴다. 두 구간에 서로 다른 속도를 쓰면 총 소인량이 경로
모양에 의존하게 되어, 제자리 대기가 이동보다 불리하게 값매겨지는 편향이
생긴다.

### 2.4 지형 가중

지형은 관측 가능성에 영향을 준다. 본 연구는 셀별 관측성 배율 `w_i ∈ (0, 1]`
를 도입하여 식 (5)에 곱한다. 즉 가시성이 낮은 셀은 같은 소인면적에 대해 더
낮은 hazard 를 얻는다. 배율은 계획 시드마다 생성되는 합성 지형에서 산출하며,
계획이 사용하는 지형과 평가에 쓰이는 진리 지형을 분리할 수 있도록 두었다.
지형 가중을 끄면 셀별 hazard 가 균일해져 대칭해가 급증하고 정수 master 가
그 대칭을 깨지 못한다. 이 대조군은 별도 조건으로 선언한다.

### 2.5 다표적 최소최대 목적

표적의 **종류**는 알려져 있으나 **기동 양상**은 알려져 있지 않은 조건을
반영하여, 하나의 표적군을 서로 다른 기동 모드 비중을 갖는 복수의 표적 모형
`k ∈ K` 로 표현하고, 계획을 그 중 **최악 표적**으로 평가한다.

표적 `k` 의 초기 신념분포를 `π^k`, 슬라이스 `t` 의 전이행렬을 `P^k_t`,
누적 hazard 장(場)을 `Y^k ∈ R^{T x n}` 이라 할 때, 미탐지확률은 다음
전방 재귀로 얻는다.

    u_1 = π^k
    u_{t+1} = ( u_t ⊙ exp(-Y^k_t) ) P^k_t                   (7)
    f_k(Y^k) = Σ_i u_T(i) exp(-Y^k_{T,i})

`X_{j,t,i,k} ∈ {0,1}` 을 탐색자 `j` 가 슬라이스 `t` 에 `i` 에서 `k` 로 이동
하는 결정변수라 하면, hazard 장은 `X` 에 대해 선형이다.

    Y^k_{t,j} = Σ_j Σ_{(i→k')} α_j(t, i→k') · m_k · X_{j,t,i,k'}   (8)

여기서 `m_k` 는 표적 시그니처에 따른 hazard 배율이다. 최종 문제는

    min_X  max_{k ∈ K}  f_k( Y^k(X) )                       (9)

이며, 제약은 (i) 탐색자별·슬라이스별 행동 1 개, (ii) 시간전개 흐름 보존,
(iii) 셀 점유 한도, (iv) 양립 불가 이동(반대 방향 교차 금지) 이다.

## 3. 최적성 인증 해법

### 3.1 외부 근사 절단평면

`f_k` 는 `Y` 에 대해 볼록하고 `Y` 는 `X` 에 대해 선형이므로, `f_k` 의
접평면은 `f_k` 를 아래에서 받친다. 이를 이용해 보조변수 `η` 를 도입하고
Kelley[6] · Duran-Grossmann[7] 형태의 외부 근사 문제를 푼다.

    min η
    s.t.  η ≥ f_k(Ŷ) + ∇f_k(Ŷ) · ( Y^k - Ŷ )   ∀ 생성된 절단   (10)
          (8), (i)-(iv),  X ∈ {0,1}

기울기는 전방·후방 재귀로 계산한다. `b_T = 1`,
`b_t = P_t ( exp(-Y_{t+1}) ⊙ b_{t+1} )` 라 할 때

    ∂f / ∂Y_{t,i} = - u_t(i) · exp(-Y_{t,i}) · b_t(i)       (11)

이다. 즉 한 번의 전방 재귀와 한 번의 후방 재귀로 `f` 와 `∇f` 를 동시에 얻으며,
표적 경로를 지수 개로 열거하지 않는다.

### 3.2 상계·하계와 최적성 간격

반복 `r` 에서 master 를 풀면 그 최적값 `η*` 는 원문제의 **하계** `L` 이 된다.
절단이 `f_k` 를 아래에서 받치므로 외부 근사 문제의 실행가능집합이 원문제를
포함하기 때문이다. 동시에 master 가 낸 정수해 `X*` 는 실제로 비행 가능한
계획이므로, 이를 식 (7)로 정확히 평가한 값은 **상계** `U` 가 된다.

    L ≤ OPT ≤ U,      gap = ( U - L ) / |U|                 (12)

`gap` 이 선언한 허용치 이하로 떨어지면 **인증 완료**로 판정한다. 인증된
계획의 탐지확률은 최적 계획 대비 최대 `gap` 만큼만 열등하다.

주의할 점은 `U` 와 `L` 의 성격이 다르다는 것이다. `U` 는 **실제로 달성한**
값이고, `L` 은 **아직 달성하지 못했지만 그 아래로는 내려갈 수 없다는** 증명
이다. 따라서 `gap` 은 계획의 품질이 아니라 **남은 불확실성의 크기**를 뜻한다.

### 3.3 master MILP 의 규모

각 반복의 master 는 혼합정수선형계획이다. 변수는 arc 이진변수
(탐색자 클래스 x 슬라이스 x 도달 가능한 arc), hazard 연속변수
(표적 x 슬라이스 x 셀), 그리고 `η` 하나다. 셀 수가 늘면 arc 수는 셀 수에
비례하고 슬라이스 수는 식 (4)에 의해 다시 셀 크기에 반비례하므로, master 규모는
격자 한 변의 세제곱에 가깝게 증가한다. 이것이 인증 가능한 해상도의 상한을
결정한다.

### 3.4 구현 검증

인증의 신뢰성은 구현의 정확성에 의존하므로, 다음 네 가지를 독립적인 방법으로
확인하였다.

1) **목적함수** : 전이행렬로 표적 궤적을 추출하고 셀별 생존확률로 탐지를
   판정하는 몬테카를로 모의(표본 40 만)와 식 (7)의 재귀를 대조하였다. 경로
   분배 소인을 포함한 8 개 사례 전부에서 최대 차이가 1.4 x 10^-3 으로 표본
   오차(3σ = 2.4 x 10^-3) 안이었다.
2) **기울기** : 식 (11)의 해석적 기울기를 중심차분과 대조하여 최대 오차
   2.4 x 10^-11 을 확인하였다.
3) **절단의 유효성** : 무작위로 생성한 hazard 장 2,400 개에 대하여
   `f(Y) ≥ f(Ŷ) + ∇f(Ŷ)·(Y-Ŷ)` 를 검사하여 **위반 0 건**을 확인하였다.
   하계가 하계임이 성립하는 근거다.
4) **인증 정합성** : 실행 가능한 경로 묶음을 **완전열거**하여 진짜 최적값을
   구하고 솔버의 `L`, `U` 와 대조하였다. 6 개 사례 전부에서
   `L = OPT = U` 가 정확히 일치하였고, 반환된 계획이 실행가능집합 안에 있으며
   보고된 상계를 실제로 달성함을 확인하였다.

기하 측면에서는 이동 구간이 셀별로 나뉜 길이의 합이 직선거리와 상대오차
1.4 x 10^-16 로 일치함을, 표적 전이행렬의 행합이 1.000000000000 이고 음수
성분이 없음을 확인하였다.

## 4. 실험 및 결과

### 4.1 연구 프레임워크

계획과 인증은 다음 순서로 진행된다.

    (a) 임무 조건 -> AOI 및 격자 확정 (외접, N x N)
    (b) 식 (4) 로 슬라이스 수 T 유도
    (c) 표적군별 입자 신념분포 전파 -> 격자 전이행렬 추출
    (d) 합성 지형 -> 셀별 관측성 배율
    (e) 식 (5) 로 arc 별 소인 hazard 표 구성
    (f) 휴리스틱 초기해 (warm start)
    (g) 절단평면 반복 : master MILP -> 정수해 -> 정확 평가 -> 절단 추가
    (h) gap ≤ 허용치 이면 인증, 아니면 예산 소진 보고

(f)의 초기해는 가속 장치이지 필수가 아니다. 초기해가 전부 실행 불가한
경우에도 master 가 스스로 해를 찾는지 회귀 시험으로 고정하였다.

### 4.2 이산화의 자기정합성

식 (4)의 규칙 하에서 도달거리 대 셀 크기 비는 1.09 ~ 1.12 이므로 대각 이동이
불가능하고, 모든 셀의 후속 셀은 정확히 5 개다. 이 사실은 초기 배치에 구조적인
제약을 만든다. 셀 점유 한도가 1 인 조건에서 `m` 대의 탐색자를 **한 점에서**
출발시키면, 첫 슬라이스에 서로 다른 목적지 `m` 개가 필요한데 후속 셀은 최대
5 개뿐이다. `m > 5` 이면 실행 가능한 계획이 존재하지 않는다. 실제로 탐색자 6 대를 한 셀에 두면 연속완화 단계에서 실행 불가로
판정되었으며, 이는 격자 해상도와 표적군에 무관하게 재현되었다. 따라서 초기
배치는 **주어지는 조건**으로 다루어야 하며, 계획법이 선택할 수 있는 변수가
아니다.

### 4.3 신념분포의 집중과 격자 밖 누출

이송 단계가 끝난 시점의 표적 신념분포는 AOI 전역에 퍼지지 않고 소수의 셀에
집중된다. 본 실험 조건에서 질량의 50 % 가 3 ~ 4 개 셀에, 90 % 가 6 ~ 9 개
셀에 놓였다. 이는 전체 셀 수의 7 ~ 14 % 에 해당한다. 따라서 이 문제는 AOI
전역을 고르게 덮는 문제가 아니라 **소수의 유력 셀을 얼마나 깊이 훑는가**의
문제이며, 격자를 세분하여 얻는 이득도 그 유력 셀들의 분해능에서 나온다.

외접 격자를 쓰더라도 탐색 단계 중 표적 질량의 일부는 AOI 밖으로 나간다.
측정된 누출은 표적 모형에 따라 0 ~ 0.74 % 였다. 격자 밖은 흡수 상태로 두어
그 질량이 목적함수에서 미탐지로 계상되도록 하였다. 내접 격자를 쓰면 이
누출이 훨씬 커져 최소최대의 최악 표적이 바뀔 수 있다.

### 4.4 풀이 깊이와 반복 횟수

계산시간이 정해져 있을 때, 한 번을 **깊게** 풀 것인가 여러 번을 **얕게**
풀 것인가를 골라야 한다. 절단평면법에서 이 선택은 (i) master 1 회당 허용
시간과 (ii) 반복 횟수 두 축으로 나타난다.

두 축은 서로 다른 경계를 움직인다. **상계는 반복이 움직이고, 하계는 풀이
깊이가 움직인다.** 절단이 추가되어야 외부 근사가 조여져 상계가 내려가지만,
하계는 정수 master 가 얼마나 깊이 풀렸는지에 달려 있기 때문이다.

동일 인스턴스에 대해 두 선택을 비교한 결과, 얕게 여러 번 푼 경우 일정 반복
이후 상계·하계가 모두 정체하여 인증에 실패한 반면, 깊게 적게 푼 경우 더 짧은
총 시간 안에 인증이 닫혔다. 절단을 아무리 많이 쌓아도 정수 master 가 얕게
풀리면 쌍대 한계가 따라오지 못한다.

### 4.5 재현성의 경계

동일 시드·동일 조건에서 두 번 실행한 결과, **탐지확률은 부동소수점 16 자리
까지 완전히 일치**하였으나 최적성 간격과 반복 횟수는 일치하지 않았다. 원인은
master 의 시간 한도다. 한도에 걸린 MILP 는 그 시점까지의 약한 쌍대 한계를
반환하며, 한도에 걸리는지 여부는 실행 시점의 계산 부하에 의존한다.

이는 결함이 아니라 **인증의 성격**이다. 약한 하계도 유효한 하계이므로 인증은
보수적일 뿐 틀리지 않는다. 다만 보고 시에는 **계획과 그 탐지확률은 재현 가능한
수치**로, **최적성 간격은 실행 조건에 딸린 수치**로 구분하여 다루어야 한다.

### 4.6 인증 결과

(앵커 재실행 완료 후 기입)

## 5. 결론

본 연구는 다중 무인기 협동탐색 경로계획에 대하여 계획과 **최적성 인증**을
함께 산출하는 프레임워크를 제시하였다. 경로 제약이 있는 다표적 최소최대 탐색
배분 문제를 정식화하고, 이동 중 소인을 경로에 분배하는 hazard 모형을
도입하였으며, 외부 근사 절단평면법으로 상계·하계를 동시에 갱신하여 최적성
간격을 보고하였다.

구현은 완전열거·몬테카를로 모의·유한차분·접평면 검사로 독립 검증하였고, 작은
사례에서 인증된 경계가 진짜 최적값과 정확히 일치함을 확인하였다.

실험을 통해 세 가지 구조적 성질을 확인하였다. 첫째, 이산화는 자유 변수가
아니며 셀 크기가 슬라이스 수를 결정한다. 그 결과 후속 셀이 5 개로 고정되어
초기 배치에 구조적 제약이 생긴다. 둘째, 계산예산은 반복이 아니라 master
시간에 배분해야 인증이 닫힌다. 셋째, 계획과 그 탐지확률은 재현 가능하지만
최적성 간격은 계산 부하에 의존하므로, 인증은 보수적일 뿐 틀리지 않는다.

향후 과제는 학습 기반 계획법(다중 에이전트 강화학습)과의 비교이며, 본
프레임워크가 제공하는 인증은 그 비교에서 **두 계획법이 같은 문제를 풀었는가**
와 **최적 대비 얼마나 떨어져 있는가**를 동시에 판정하는 기준으로 쓰인다.

## 후 기

(기입 필요)

## References

[1] L. D. Stone, J. O. Royset and A. R. Washburn, *Optimal Search for Moving
    Targets*, Springer, 2016.
[2] B. O. Koopman, *Search and Screening: General Principles with Historical
    Applications*, Pergamon Press, 1980.
[3] A. R. Washburn, *Search and Detection*, 4th ed., INFORMS, 2002.
[4] J. N. Eagle and J. R. Yee, "An Optimal Branch-and-Bound Procedure for the
    Constrained Path, Moving Target Search Problem," *Operations Research*,
    Vol. 38, No. 1, pp. 110-114, 1990.
[5] K. E. Trummel and J. R. Weisinger, "The Complexity of the Optimal Searcher
    Path Problem," *Operations Research*, Vol. 34, No. 2, pp. 324-327, 1986.
[6] J. E. Kelley, "The Cutting-Plane Method for Solving Convex Programs,"
    *Journal of the SIAM*, Vol. 8, No. 4, pp. 703-712, 1960.
[7] M. A. Duran and I. E. Grossmann, "An Outer-Approximation Algorithm for a
    Class of Mixed-Integer Nonlinear Programs," *Mathematical Programming*,
    Vol. 36, pp. 307-339, 1986.
[8] Q. Huangfu and J. A. J. Hall, "Parallelizing the Dual Revised Simplex
    Method," *Mathematical Programming Computation*, Vol. 10, pp. 119-142,
    2018.
