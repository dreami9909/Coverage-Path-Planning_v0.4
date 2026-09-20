"""상계 U 가 만들어지는 다섯 단계를 3x3 장난감 문제로 보인다.

평가는 논문 코드의 진짜 함수(nondetection_trace)를 그대로 호출한다.
"""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
import sys; sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from cpp_search.theory.path_constrained import (
    SearcherModel, PathConstrainedProblem, nondetection_trace)

N, T, A = 3, 3, 0.7          # 3x3 격자, 3단계, 방문 1회당 hazard
S = N * N
rc = lambda i: (i // N, i % N)
idx = lambda r, c: r * N + c

# 인접: 상하좌우 + 제자리
adj = np.zeros((S, S), dtype=bool)
for i in range(S):
    r, c = rc(i); adj[i, i] = True
    for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
        if 0 <= r+dr < N and 0 <= c+dc < N: adj[i, idx(r+dr, c+dc)] = True

pi0 = np.array([0, .2, 0, .2, .4, .2, 0, 0, 0], float); pi0 /= pi0.sum()

def transition(stay):
    """stay 확률로 제자리, 나머지는 오른쪽 한 칸(막히면 제자리)."""
    P = np.zeros((S, S))
    for i in range(S):
        r, c = rc(i)
        j = idx(r, c+1) if c+1 < N else i
        P[i, i] += stay; P[i, j] += 1 - stay
    return P

HYP = {"가설 A · 정지 성향": 0.85, "가설 B · 이동 성향": 0.30}
paths = ((0, 1, 4), (8, 5, 5))          # master 해를 분해한 기체별 셀 순서

rate = np.full((T, S), A)
searchers = tuple(SearcherModel(start_state=p[0], detection_rate=rate, adjacency=adj)
                  for p in paths)

out = {}
for name, stay in HYP.items():
    P = transition(stay)
    prob = PathConstrainedProblem(pi0, np.stack([P] * (T - 1)), searchers)
    tr = nondetection_trace(prob, paths)
    out[name] = tr

# hazard 장 Y  (셀 점유로부터)
Y = np.zeros((T, S))
for p in paths:
    for t, cell in enumerate(p): Y[t, cell] += A

print("hazard 장 Y (단계 x 셀), 방문 1회 = %.1f" % A)
print(Y, "\n")
for name, tr in out.items():
    f = tr.probability_of_no_detection
    print(f"{name}:  미탐지 f = {f:.4f}   탐지 P_D = {1-f:.4f}")
    print("   단계별 남은 질량 합:", np.round(tr.mass.sum(1), 4),
          "→ 소인 후:", np.round(tr.surviving_mass.sum(1), 4))
fs = {k: v.probability_of_no_detection for k, v in out.items()}
worst = max(fs, key=fs.get)
print(f"\nU = max_k f_k = {fs[worst]:.4f}   (최악 = {worst})")
print(f"보고되는 탐지확률 = 1 - U = {1-fs[worst]:.4f}")
