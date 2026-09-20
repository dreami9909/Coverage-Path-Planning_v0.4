
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

for cand in ("Apple SD Gothic Neo", "AppleGothic", "NanumGothic", "Malgun Gothic"):
    if cand in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = cand; break
plt.rcParams.update({"axes.unicode_minus": False, "mathtext.fontset": "dejavusans",
                     "font.size": 8.5})

N, T, A = 3, 3, 0.7
S = N * N
rc = lambda i: (i // N, i % N)
idx = lambda r, c: r * N + c
adj = np.zeros((S, S), bool)
for i in range(S):
    r, c = rc(i); adj[i, i] = True
    for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
        if 0 <= r+dr < N and 0 <= c+dc < N: adj[i, idx(r+dr, c+dc)] = True
pi0 = np.array([0,.2,0,.2,.4,.2,0,0,0], float); pi0 /= pi0.sum()
def transition(stay):
    P = np.zeros((S, S))
    for i in range(S):
        r, c = rc(i); j = idx(r, c+1) if c+1 < N else i
        P[i, i] += stay; P[i, j] += 1 - stay
    return P
HYP = [("가설 A · 정지 성향", .85, "#a8542a"), ("가설 B · 이동 성향", .30, "#176a73")]
paths = ((0,1,4), (8,5,5))
rate = np.full((T, S), A)
searchers = tuple(SearcherModel(start_state=p[0], detection_rate=rate, adjacency=adj) for p in paths)
traces = {}
for name, stay, _ in HYP:
    prob = PathConstrainedProblem(pi0, np.stack([transition(stay)]*(T-1)), searchers)
    traces[name] = nondetection_trace(prob, paths)
Y = np.zeros((T, S))
for p in paths:
    for t, cell in enumerate(p): Y[t, cell] += A

def grid_ax(ax, title):
    ax.set_xlim(-.05, N+.05); ax.set_ylim(-.05, N+.05); ax.set_aspect("equal")
    for k in range(N+1):
        ax.axhline(k, color="0.82", lw=.7); ax.axvline(k, color="0.82", lw=.7)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=8.5, pad=4)
    for s in ax.spines.values(): s.set_visible(False)
xy = lambda i: (rc(i)[1] + .5, N - rc(i)[0] - .5)

def strip(ax, M, title, cmap, vmax=None, labels=None):
    """3단계를 가로로 이어 붙인 미니 격자."""
    pad = np.full((N, 1), np.nan)
    tiles, sep = [], []
    for t in range(T):
        tiles.append(M[t].reshape(N, N))
        if t < T-1: tiles.append(pad)
    img = np.hstack(tiles)
    ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax if vmax else np.nanmax(img))
    ax.set_anchor("N")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=8.5, pad=4)
    for t in range(T):
        ax.text(t*(N+1) + 1, N - .3, labels[t] if labels else f"t={t}",
                ha="center", va="top", fontsize=7, color="0.25")
    for s in ax.spines.values(): s.set_visible(False)

fig = plt.figure(figsize=(11.2, 6.2))
gs = fig.add_gridspec(2, 3, hspace=.58, wspace=.30,
                      left=.04, right=.97, top=.87, bottom=.10)

# ① master 가 준 호 유량
ax = fig.add_subplot(gs[0,0]); grid_ax(ax, "① master MILP 가 내놓은 것 — 호 유량")
cols = ["#1f6feb", "#c2410c", "#15803d"]
for t in range(T-1):
    for p in paths:
        a, b = xy(p[t]), xy(p[t+1])
        if p[t] == p[t+1]:
            ax.add_patch(plt.Circle(a, .17, fill=False, ec=cols[t], lw=1.6))
        else:
            ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11,
                                         lw=1.6, color=cols[t], shrinkA=9, shrinkB=9))
for i in range(S):
    ax.text(*xy(i), str(i), ha="center", va="center", fontsize=7, color="0.55")
ax.text(.5, -.13, "호마다 “몇 대가 지나갔는가”라는 정수만 있다\n"
                  "파랑 t=0→1,  주황 t=1→2,  원 = 제자리",
        transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color="0.3")

# ② 경로 분해
ax = fig.add_subplot(gs[0,1]); grid_ax(ax, "② _extract_group_paths — 기체별 셀 순서로 분해")
pc = ["#1f6feb", "#c2410c"]
for j, p in enumerate(paths):
    off = (-.14, .14)[j]
    for t in range(T-1):
        a = (xy(p[t])[0]+off, xy(p[t])[1]); b = (xy(p[t+1])[0]+off, xy(p[t+1])[1])
        if p[t] != p[t+1]:
            ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10,
                                         lw=2.0, color=pc[j], shrinkA=8, shrinkB=8))
    for t, cell in enumerate(p):
        x, y = xy(cell)
        ax.scatter(x+off, y, s=150, color=pc[j], ec="k", lw=.6, zorder=4, alpha=.35+.3*t)
        ax.text(x+off, y, str(t), color="w", fontsize=6.5, ha="center", va="center",
                fontweight="bold", zorder=5)
ax.text(.5, -.13, "기체 1: 0 → 1 → 4      기체 2: 8 → 5 → 5\n숫자는 단계, 실제로 비행 가능한 계획",
        transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color="0.3")

# ③ hazard 장
ax = fig.add_subplot(gs[0,2])
strip(ax, Y, "③ _target_path_hazard — 경로가 쌓은 hazard 장 Y", "Purples", vmax=A)
ax.text(.5, -.42, "방문한 셀에만 a = 0.7 이 쌓인다\n"
                  "같은 셀에 두 대면 더한다 (확률이 아니라 hazard 라서)",
        transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color="0.3")

# ④⑤ 전방재귀
for col, (name, stay, color) in enumerate(HYP):
    tr = traces[name]
    ax = fig.add_subplot(gs[1, col])
    strip(ax, tr.surviving_mass, f"{'④' if col==0 else '⑤'} 전방재귀 — {name}",
          "Greys", vmax=float(tr.mass.max()),
          labels=[f"t={t}\n남은 질량 {tr.surviving_mass[t].sum():.3f}" for t in range(T)])
    ax.text(.5, -.52, f"소인으로 깎고(exp(-Y)) 전이로 옮기고(P) 반복\n"
                      f"→  f = {tr.probability_of_no_detection:.4f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color=color)

# ⑥ 최악 가설
ax = fig.add_subplot(gs[1,2])
names = [h[0] for h in HYP]; fs = [traces[n].probability_of_no_detection for n in names]
bars = ax.barh(range(2), fs, color=[h[2] for h in HYP], height=.42)
U = max(fs); wi = int(np.argmax(fs))
ax.axvline(U, color="crimson", ls="--", lw=1.3)
ax.set_ylim(-.75, 1.75)
ax.text(U, 1.42, f"U = {U:.4f}", color="crimson", fontsize=9.5, ha="center", fontweight="bold")
ax.annotate("", xy=(U, 1.28), xytext=(U-.09, 1.28),
            arrowprops=dict(arrowstyle="-|>", color="crimson", lw=1.1))
for i, v in enumerate(fs):
    ax.text(v - .02, i, f"{v:.4f}", va="center", ha="right", color="w", fontsize=8.5, fontweight="bold")
ax.set_yticks(range(2)); ax.set_yticklabels(["A · 정지", "B · 이동"], fontsize=8.5)
ax.set_xlim(0, .62); ax.set_xlabel("미탐지확률 f", fontsize=8)
ax.set_title("⑥ max over k — 최악 기동가설이 상계", fontsize=8.5, pad=4)
ax.spines[["top","right"]].set_visible(False); ax.tick_params(labelsize=7.5)
ax.text(.5, -.22, f"최악은 «{names[wi].split('·')[1].strip()}»  →  U = {U:.4f}\n"
                  f"보고되는 탐지확률 = 1 - U = {1-U:.4f}",
        transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color="0.3")

fig.suptitle("상계 U 는 어떻게 만들어지는가 — 3×3 격자·2대·3단계 예제 (평가는 논문 코드의 nondetection_trace 그대로)",
             fontsize=10.5, y=.965)
out = str(ROOT / "paper" / "fig" / "fig12_upper_bound.png")
fig.savefig(out, dpi=190, bbox_inches="tight", facecolor="white")
print("저장:", out)
