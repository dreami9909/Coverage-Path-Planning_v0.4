"""학회 2쪽 요약본의 그림 3 — 셀별 가중(위)과 단계별 계획 점유(아래).

cellfig.py 와 stepfig.py 가 각각 내는 두 장을 단 폭 한 장으로 합친다.
2쪽 양식에서는 그림을 세 장까지만 싣기 때문이다. 자료 경로는 두 원본과 같다.
"""
import sys, json, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.font_manager
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter

for cand in ("Apple SD Gothic Neo", "AppleGothic", "NanumGothic", "Malgun Gothic"):
    if cand in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = cand; break
plt.rcParams.update({"axes.unicode_minus": False, "mathtext.fontset": "dejavusans",
                     "font.size": 6.0})

from cpp_search.config import load_chapter_config
from cpp_search.options import RunOptions
from cpp_search.chapters.ch2_stone_spx import (
    build_instance, case_mission, case_specs, case_target_specs)
from cpp_search.theory.path_constrained import search_free_marginals, PathConstrainedProblem
from cpp_search.theory import transitions as top

CASE, N, SEED = "Tank", 8, 20260201
cfg = load_chapter_config("2"); m0, sensor = cfg.mission(), cfg.sensor()
base = next(g for g in cfg.get("stone_spx.grid_conditions", []) if g.get("grid_kind") == "square")
block = next(c for c in case_specs(cfg) if c["name"] == CASE)
mission = case_mission(m0, block)
inst = build_instance(cfg, RunOptions(sample_count=10, particle_count=5000, episode_count=1,
        target_profile="both", seed=1, map_error=False, communication_loss_probability=0.0,
        communication_latency_slices=0, mission_time_s=600.0),
    mission, sensor, grid={**base, "grid_width": N, "grid_height": N},
    seed=SEED, terrain_weighting=True, targets=case_target_specs(cfg, block))

G, terr = inst.grid, inst.terrain
side = float(G.cell_area_m2) ** 0.5
cx = np.array([G.center_of(i).x for i in range(G.cell_count)])
cy = np.array([G.center_of(i).y for i in range(G.cell_count)])
g2 = lambda v: np.asarray(v[:G.cell_count], float).reshape(N, N)
names = [t.split("/")[-1] for t in inst.target_names]
tm = inst.target_models[names.index("STOP_HEAVY")]
u0 = np.asarray(tm.initial_mass, float)[:G.cell_count]
mob = np.array([terr.mobility_weight(x, y) for x, y in zip(cx, cy)])
con = np.array([terr.concealment_weight(x, y) for x, y in zip(cx, cy)])

paths = [[int(c) for c in p] for p in
         json.load(open(ROOT / "results/studies/occupancy-1.json"))["assignments"]]
T, J = len(paths[0]), len(paths)
groups = []
for t in range(T):
    if groups and all(paths[j][t] == paths[j][groups[-1][-1]] for j in range(J)):
        groups[-1].append(t)
    else:
        groups.append([t])
keep = {0, len(groups) - 1}
for idx in sorted(range(1, len(groups) - 1), key=lambda i: (-len(groups[i]), i)):
    if len(keep) >= 3: break
    keep.add(idx)
shown = [groups[i] for i in sorted(keep)]

fig = plt.figure(figsize=(3.23, 2.46))
gs = fig.add_gridspec(2, 3, hspace=.46, wspace=.30,
                      left=.015, right=.955, top=.90, bottom=.045)
half = N * side / 2
ext = [-half, half, -half, half]

def layer(ax, Z, title, cmap, log=False):
    Zp = np.flipud(Z.T)
    if log:
        v = Zp[Zp > 0]
        im = ax.imshow(np.where(Zp > 0, Zp, np.nan), extent=ext, cmap=cmap,
                       norm=LogNorm(vmin=max(v.min(), 1e-6), vmax=v.max()))
        zy, zx = np.where(~(Zp > 0))
        ax.scatter(ext[0] + (zx + .5) * side, ext[3] - (zy + .5) * side,
                   marker="x", s=1.2, c="0.45", lw=.25)
    else:
        im = ax.imshow(Zp, extent=ext, cmap=cmap)
    th = np.linspace(0, 2 * np.pi, 160)
    ax.plot(mission.search_radius_m * np.cos(th), mission.search_radius_m * np.sin(th),
            "k--", lw=.55, alpha=.55)
    ax.set_title(title, fontsize=5.6, pad=2.5)
    ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=.046, pad=.02)
    if log:
        cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        cb.ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    cb.ax.tick_params(labelsize=4.0, length=1.2, width=.35)
    cb.outline.set_linewidth(.35)

layer(fig.add_subplot(gs[0, 0]), g2(u0), "(가) 표적 존재확률", "viridis", log=True)
layer(fig.add_subplot(gs[0, 1]), g2(mob), "(나) 지형 통행성 가중", "YlOrBr")
layer(fig.add_subplot(gs[0, 2]), g2(con), "(다) 지형 은폐 가중", "Greens")

bel = g2(u0)
cols = [plt.cm.tab10(j) for j in range(J)]
for k, g in enumerate(shown):
    ax = fig.add_subplot(gs[1, k]); t = g[0]
    Z = np.flipud(bel.T)
    ax.imshow(np.where(Z > 0, Z, np.nan), extent=[0, N, 0, N], cmap="Greys",
              norm=LogNorm(vmin=max(bel[bel > 0].min(), 1e-6), vmax=bel.max()), alpha=.45)
    for x in range(N + 1): ax.axvline(x, color="0.86", lw=.3)
    for y in range(N + 1): ax.axhline(y, color="0.86", lw=.3)
    seen = {}
    for j in range(J):
        c = paths[j][t]; ccx, ccy = c % N, c // N
        d = seen.get(c, 0); seen[c] = d + 1
        ax.scatter(ccx + .5 + .16 * d, ccy + .5, s=22, color=cols[j], ec="k", lw=.35, zorder=4)
        ax.text(ccx + .5 + .16 * d, ccy + .5, str(j + 1), fontsize=3.4, ha="center",
                va="center", color="w", fontweight="bold", zorder=5)
    lbl = f"단계 {g[0]}" if len(g) == 1 else f"단계 {g[0]}–{g[-1]}"
    ax.set_title(lbl, fontsize=5.6, pad=2.5)
    ax.set_xlim(0, N); ax.set_ylim(0, N)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")

fig.text(.49, .955, "셀별 가중", ha="center", fontsize=6.4, fontweight="bold")
fig.text(.49, .452, "단계별 계획 점유", ha="center", fontsize=6.4, fontweight="bold")
out = ROOT / "paper/fig/fig13_cells_and_steps.png"
fig.savefig(out, dpi=470, bbox_inches="tight")
print("저장:", out)
print("수록 단계:", [f"{g[0]}" if len(g) == 1 else f"{g[0]}-{g[-1]}" for g in shown])
