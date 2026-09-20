"""hazard 장 Y 가 어떻게 만들어지는가 — 실제 _segment_cell_lengths / _swept_hazard 호출."""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
import sys; sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.font_manager
from matplotlib.patches import FancyArrowPatch
from cpp_search.core.models import Point2D
from cpp_search.planning.stone_spx import _SquareGrid, _segment_cell_lengths, _swept_hazard
for c in ("Apple SD Gothic Neo","AppleGothic","NanumGothic","Malgun Gothic"):
    if c in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = c; break
plt.rcParams.update({"axes.unicode_minus": False, "font.size": 9})

W = H = 5; SIDE = 1000.0; S = W*H
SWEEP, V, SLICE = 400.0, 25.0, 120.0
grid = _SquareGrid(Point2D(0.0, 0.0), SIDE*W/2, W, H)
AREA = grid.cell_area_m2
adjall = np.ones((S+1, S+1), bool)
haz = _swept_hazard(grid, adjall, sweep_width_m=SWEEP, transit_speed_mps=V,
                    search_speed_mps=V, slice_s=SLICE, cell_scale=np.ones(S))
ARCS = [(6, 13, "#1f6feb", "기체 1"), (2, 12, "#c2410c", "기체 2")]
xy  = lambda cell: (cell % W + .5, cell // W + .5)
TOT = SWEEP * V * SLICE / AREA

def grid_ax(ax, title, fs=10):
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal")
    for k in range(W+1): ax.axvline(k, color="0.8", lw=.7)
    for k in range(H+1): ax.axhline(k, color="0.8", lw=.7)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=fs, pad=5)
    for s_ in ax.spines.values(): s_.set_visible(False)

fig = plt.figure(figsize=(13.0, 8.4))
gs = fig.add_gridspec(2, 3, height_ratios=[1.12, 1], hspace=.52, wspace=.26,
                      left=.045, right=.975, top=.885, bottom=.055)

# ───────── A : 호 하나가 어느 셀을 얼마나 지나는가
ax = fig.add_subplot(gs[0, :2])
grid_ax(ax, "① 호 하나가 지나간 셀과 그 길이  —  _segment_cell_lengths()", 10.5)
src, dst, col, _ = ARCS[0]
a, b = grid.center_of(src), grid.center_of(dst)
L = _segment_cell_lengths(grid, a, b)
pa, pb = xy(src), xy(dst)
dxp, dyp = xy(dst)[0]-xy(src)[0], xy(dst)[1]-xy(src)[1]
_cuts = sorted({0.0, 1.0}
    | {(k-xy(src)[0])/dxp for k in range(W+1) if 0 < (k-xy(src)[0])/dxp < 1}
    | {(k-xy(src)[1])/dyp for k in range(H+1) if 0 < (k-xy(src)[1])/dyp < 1})
for n, (lo, hi) in enumerate(zip(_cuts, _cuts[1:])):
    mt = (lo+hi)/2
    mx, my = xy(src)[0]+dxp*mt, xy(src)[1]+dyp*mt
    cell = int(my)*W + int(mx)
    cx, cy = xy(cell)
    ax.add_patch(plt.Rectangle((cx-.5, cy-.5), 1, 1, color=col, alpha=.13))
    off = .34 if n % 2 == 0 else -.40
    ax.text(mx, my+off, f"ℓ = {L[cell]:.0f} m", ha="center", fontsize=8.8,
            color=col, fontweight="bold",
            bbox=dict(fc="white", ec="none", pad=1.2))
ax.add_patch(FancyArrowPatch(pa, pb, arrowstyle="-|>", mutation_scale=16, lw=2.6,
                             color=col, shrinkA=0, shrinkB=0, zorder=5))
# 격자선 교차점
dx, dy = pb[0]-pa[0], pb[1]-pa[1]
cuts = sorted({0.0, 1.0} | {(k-pa[0])/dx for k in range(W+1) if 0 < (k-pa[0])/dx < 1}
                         | {(k-pa[1])/dy for k in range(H+1) if 0 < (k-pa[1])/dy < 1})
for t in cuts[1:-1]:
    ax.scatter([pa[0]+dx*t], [pa[1]+dy*t], s=48, color="w", ec=col, lw=1.8, zorder=6)
ax.scatter(*pa, s=90, color=col, zorder=7); ax.scatter(*pb, s=90, color=col, zorder=7)
ax.text(pa[0]-.62, pa[1]-.22, f"출발\n셀 {src}", ha="center", va="top", fontsize=8.8,
        color=col, linespacing=1.4)
ax.text(pb[0]+.62, pb[1]+.22, f"도착\n셀 {dst}", ha="center", va="bottom", fontsize=8.8,
        color=col, linespacing=1.4)
ax.text(.5, -.075, f"직선을 격자선에서 잘라 셀마다 지나간 길이 ℓ 를 잰다  ·  전체 {a.distance_to(b):.0f} m"
                  f"\n이동에 {a.distance_to(b)/V:.1f} s, 도착 셀에 남는 체류 {SLICE - a.distance_to(b)/V:.1f} s",
        transform=ax.transAxes, ha="center", va="top", fontsize=9, color="0.3", linespacing=1.6)

# ───────── B : 그 호의 hazard 계수
ax = fig.add_subplot(gs[0, 2])
cells = sorted(haz[(src, dst)])
mv = [SWEEP*L.get(c, 0.0)/AREA for c in cells]
st = [haz[(src, dst)][c] - m for c, m in zip(cells, mv)]
xpos = np.arange(len(cells))
ax.bar(xpos, mv, .52, color=col, label="이동 중  W·ℓ/면적")
ax.bar(xpos, st, .52, bottom=mv, color=col, alpha=.42, hatch="//",
       label="도착 후  W·v·잔여시간/면적")
ax.bar(xpos + .0, 0, 0)
old = [0]*len(cells); old[cells.index(dst)] = TOT
ax.plot(xpos, old, "o--", color="0.55", ms=5, lw=1.1, label="옛 모형: 도착 셀에 몰아주기")
ax.axhline(TOT, color="crimson", ls=":", lw=1.2)
ax.text(len(cells)-.55, TOT+.03, f"합 {TOT:.2f}", color="crimson", fontsize=9, ha="right")
ax.set_xticks(xpos); ax.set_xticklabels([f"셀 {c}" for c in cells], fontsize=8.5)
ax.set_ylabel("hazard 계수", fontsize=9)
ax.set_title("② 호가 각 셀에 남기는 hazard", fontsize=10.5, pad=5)
ax.legend(fontsize=7.6, frameon=False, loc="upper left", bbox_to_anchor=(0, .92))
ax.set_ylim(0, TOT*1.22); ax.spines[["top","right"]].set_visible(False); ax.tick_params(labelsize=8)
ax.text(.5, -.16, f"어느 쪽이든 합은 W·v·Δ/면적 = {TOT:.2f} 로 같다\n"
                   "바뀌는 것은 총량이 아니라 «어디에» 쌓이는가",
        transform=ax.transAxes, ha="center", va="top", fontsize=8.8, color="0.3", linespacing=1.6)

# ───────── C : 여러 호를 더해 Y[t] 완성
fields = []
for src2, dst2, c2, name in ARCS:
    v = np.zeros(S)
    for cell, val in haz[(src2, dst2)].items(): v[cell] = val
    fields.append((v, c2, f"{name} 의 호  {src2}→{dst2}"))
Ysum = fields[0][0] + fields[1][0]
panels = [(fields[0][0], fields[0][2]), (fields[1][0], fields[1][2]),
          (Ysum, "Y[t, ·]  =  두 호의 합")]
vmax = float(Ysum.max())
for j, (field, title) in enumerate(panels):
    ax = fig.add_subplot(gs[1, j])
    ax.imshow(field.reshape(H, W), origin="lower", cmap="Purples", vmin=0, vmax=vmax)
    for i in range(S):
        if field[i] > 1e-6:
            ax.text(i % W, i // W, f"{field[i]:.2f}", ha="center", va="center",
                    fontsize=8.5, color="w" if field[i] > .55*vmax else "0.25")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(("③ " if j == 0 else "") + title, fontsize=10, pad=5,
                 color=("k" if j == 2 else "0.3"))
    for s_ in ax.spines.values(): s_.set_color("0.75")
    if j < 2:
        ax.text(1.06, .5, "+" if j == 0 else "=", transform=ax.transAxes,
                fontsize=22, color="0.4", ha="center", va="center")
fig.text(.5, .012, "같은 셀을 두 호가 지나가면 그냥 더한다 — 확률이 아니라 hazard 라서 가능하다.   "
                   "Y[t, i] = Σ(호 계수) × (그 호를 썼는가) 이므로 Y 는 결정변수 X 에 선형이다.",
         ha="center", fontsize=9.5, color="0.25")
fig.suptitle("hazard 장 Y 는 어떻게 만들어지는가  —  호 하나 → 셀별 계수 → 여러 호의 합\n"
             f"(예시값: 셀 한 변 {SIDE:.0f} m, 탐지폭 {SWEEP:.0f} m, 비행속도 {V:.0f} m/s, 한 단계 {SLICE:.0f} s)",
             fontsize=11.5, y=.975, linespacing=1.5)
out = str(ROOT / "paper" / "fig" / "fig9_hazard_field.png")
fig.savefig(out, dpi=185, bbox_inches="tight", facecolor="white")
print("저장:", out)
print("호1 합", round(sum(haz[(6,13)].values()),4), " 호2 합", round(sum(haz[(2,12)].values()),4),
      " 이론", round(TOT,4))
