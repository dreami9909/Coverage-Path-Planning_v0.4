"""전방재귀 네 줄을 한 장에 — 깎고, 옮기고, 반복."""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
import sys; sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.font_manager
from matplotlib.patches import FancyArrowPatch
from cpp_search.theory.path_constrained import (
    SearcherModel, PathConstrainedProblem, nondetection_trace)
for c in ("Apple SD Gothic Neo","AppleGothic","NanumGothic","Malgun Gothic"):
    if c in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = c; break
plt.rcParams.update({"axes.unicode_minus": False, "font.size": 9})

N, T, A = 3, 3, 0.7
S = N*N; rc = lambda i: (i//N, i%N); idx = lambda r,c: r*N+c
adj = np.zeros((S,S), bool)
for i in range(S):
    r,c = rc(i); adj[i,i] = True
    for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
        if 0<=r+dr<N and 0<=c+dc<N: adj[i, idx(r+dr,c+dc)] = True
pi0 = np.array([0,.2,0,.2,.4,.2,0,0,0], float); pi0 /= pi0.sum()
STAY = .85
P = np.zeros((S,S))
for i in range(S):
    r,c = rc(i); j = idx(r,c+1) if c+1<N else i
    P[i,i] += STAY; P[i,j] += 1-STAY
paths = ((0,1,4),(8,5,5))
rate = np.full((T,S), A)
prob = PathConstrainedProblem(pi0, np.stack([P]*(T-1)),
        tuple(SearcherModel(start_state=p[0], detection_rate=rate, adjacency=adj) for p in paths))
tr = nondetection_trace(prob, paths)
Y = np.zeros((T,S))
for p in paths:
    for t,cell in enumerate(p): Y[t,cell] += A
surv = np.exp(-Y)

fig = plt.figure(figsize=(12.6, 7.4))
gs = fig.add_gridspec(3, 4, width_ratios=[1,1,1,.78],
                      hspace=.30, wspace=.30, left=.11, right=.975, top=.84, bottom=.08)
ROWS = [("① u$_t$  —  표적이 여기 있을 확률", tr.mass, "Blues"),
        ("② exp(-Y$_t$)  —  살아남을 확률\n(훑은 셀일수록 어둡다 = 많이 깎인다)", surv, "Reds_r"),
        ("③ u$_t^+$ = u$_t$ ⊙ exp(-Y$_t$)  —  깎인 뒤", tr.surviving_mass, "Blues")]
axes = {}
for r,(label, M, cmap) in enumerate(ROWS):
    vmax = float(np.max(M))
    for t in range(T):
        ax = fig.add_subplot(gs[r,t]); axes[(r,t)] = ax
        ax.imshow(M[t].reshape(N,N), cmap=cmap, vmin=0, vmax=vmax)
        for i in range(S):
            rr, cc = rc(i); v = M[t, i]
            if v > 1e-4:
                ax.text(cc, rr, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                        color="w" if v > .55*vmax else "0.2")
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values(): s_.set_color("0.75")
        if r == 0: ax.set_title(f"t = {t}", fontsize=10.5, pad=6)
        if r == 1:      # 생존확률의 "합" 은 뜻이 없다
            ax.set_xlabel(f"훑은 셀 {int((M[t] < .999).sum())}개", fontsize=8.5,
                          labelpad=2, color="0.35")
        else:
            ax.set_xlabel(f"합 {M[t].sum():.3f}", fontsize=8.5, labelpad=2,
                          color=("crimson" if r == 2 else "0.35"))
    fig.text(.10, {0:.755, 1:.505, 2:.235}[r], label, fontsize=9.5, ha="right",
             va="center", linespacing=1.5)

# 세로 화살표 : 깎기
for t in range(T):
    for r0, r1, txt in ((0,1,"⊙"), (1,2,"=")):
        a = axes[(r0,t)].get_position(); b = axes[(r1,t)].get_position()
        fig.add_artist(FancyArrowPatch(((a.x0+a.x1)/2, a.y0-.012), ((b.x0+b.x1)/2, b.y1+.012),
            transform=fig.transFigure, arrowstyle="-|>", mutation_scale=11, lw=1.1, color="0.45"))
        fig.text((a.x0+a.x1)/2+.014, (a.y0+b.y1)/2, txt, fontsize=11, color="0.35", va="center")
# 가로 화살표 : 이동
for t in range(T-1):
    a = axes[(2,t)].get_position(); b = axes[(0,t+1)].get_position()
    fig.add_artist(FancyArrowPatch((a.x1+.004, a.y0+.02), (b.x0-.004, b.y0+.02),
        transform=fig.transFigure, arrowstyle="-|>", mutation_scale=12, lw=1.4,
        color="#176a73", connectionstyle="arc3,rad=-0.42"))
    fig.text((a.x1+b.x0)/2 + .012, (a.y0+b.y1)/2 + .10, f"× P$_{t}$\n표적이 이동",
             fontsize=9.5, color="#176a73", ha="center", va="center", linespacing=1.4,
             bbox=dict(fc="white", ec="none", pad=1.5))

# 마지막 열 : 결과
ax = fig.add_subplot(gs[:,3])
f_nd = tr.probability_of_no_detection
inc = tr.detection_increment
bottom = 0.0
for t in range(T):
    ax.bar(0, inc[t], bottom=bottom, width=.5, color=plt.cm.Oranges(.35+.22*t),
           ec="w", lw=1.2)
    ax.text(0, bottom+inc[t]/2, f"t={t} 에 탐지\n{inc[t]:.3f}", ha="center", va="center",
            fontsize=8.5, color="0.15")
    bottom += inc[t]
ax.bar(0, f_nd, bottom=bottom, width=.5, color="#2b5c8a", ec="w", lw=1.2)
ax.text(0, bottom+f_nd/2, f"끝까지 살아남음\nf = {f_nd:.4f}", ha="center", va="center",
        fontsize=9.5, color="w", fontweight="bold")
ax.set_ylim(0, 1.02); ax.set_xlim(-.45, .45); ax.set_xticks([])
ax.set_title("④ 질량 1 이 어디로 갔나", fontsize=10.5, pad=6)
ax.set_ylabel("표적 존재확률 질량", fontsize=9)
ax.spines[["top","right","bottom"]].set_visible(False); ax.tick_params(labelsize=8)
ax.text(0, -.055, f"f = Σ$_i$ u$_2^+$(i) = {f_nd:.4f}\n탐지확률 = 1 - f = {1-f_nd:.4f}",
        ha="center", va="top", fontsize=9, color="0.25", linespacing=1.5)

fig.suptitle("전방재귀 — 깎고(⊙ exp(-Y)), 옮기고(× P), 반복.   격자 안 숫자는 그 셀의 확률질량",
             fontsize=11.5, y=.945)
out = str(ROOT / "paper" / "fig" / "fig10_forward_recursion.png")
fig.savefig(out, dpi=185, bbox_inches="tight", facecolor="white")
print("저장:", out)
print("단계별 탐지 증분:", np.round(inc,4), " 합", round(inc.sum(),4), " f =", round(f_nd,4))
