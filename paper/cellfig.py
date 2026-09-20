"""Fig. 7 — 인증된 계획이 보고 결정한 셀 단위 입력.

경로는 그리지 않는다. 이 그림의 목적은 "무엇을 보았는가"(확률과 지형 가중)이고,
"어디로 갔는가"는 Fig. 4·5 가 맡는다. 한 장에 둘을 겹치면 어느 쪽도 읽히지 않는다.
"""
import sys, json, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter

WIDE = "--wide" in sys.argv          # 단독 열람용 큰 그림. 기본은 2단 조판용 단 폭.

for cand in ("Apple SD Gothic Neo", "AppleGothic", "NanumGothic", "Malgun Gothic"):
    if cand in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = cand
        plt.rcParams["axes.unicode_minus"] = False
        break
# 컬러바의 로그 눈금은 mathtext 로 그려진다. 한글 폰트에는 U+2212(−) 가 없어
# 10⁻¹ 의 빼기 기호가 두부 글자로 나왔다. mathtext 만 DejaVu 로 돌린다.
plt.rcParams["mathtext.fontset"] = "dejavusans"

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
tg = case_target_specs(cfg, block); mission = case_mission(m0, block)
inst = build_instance(cfg, RunOptions(sample_count=10, particle_count=5000, episode_count=1,
        target_profile="both", seed=1, map_error=False, communication_loss_probability=0.0,
        communication_latency_slices=0, mission_time_s=600.0),
    mission, sensor, grid={**base, "grid_width": N, "grid_height": N},
    seed=SEED, terrain_weighting=True, targets=tg)

G, terr = inst.grid, inst.terrain
side = float(G.cell_area_m2) ** 0.5
cx = np.array([G.center_of(i).x for i in range(G.cell_count)])
cy = np.array([G.center_of(i).y for i in range(G.cell_count)])
g2 = lambda v: np.asarray(v[:G.cell_count], float).reshape(N, N)

names = [t.split("/")[-1] for t in inst.target_names]
tm = inst.target_models[names.index("STOP_HEAVY")]
u0 = np.asarray(tm.initial_mass, float)[:G.cell_count]
prob = PathConstrainedProblem(np.asarray(tm.initial_mass, float),
                              top.as_sequence(tm.transitions), inst.problem.searchers)
uT = np.asarray(search_free_marginals(prob)[-1], float)[:G.cell_count]
mob = np.array([terr.mobility_weight(x, y) for x, y in zip(cx, cy)])
con = np.array([terr.concealment_weight(x, y) for x, y in zip(cx, cy)])

# 본문은 2단이라 그림이 단 폭(약 3.2 in)으로 줄어든다. 10.6 in 로 그려 두면
# 11 pt 제목이 3 pt 로 앉아 읽히지 않는다. 처음부터 단 폭으로 그린다.
FS = (10.6, 10.0) if WIDE else (3.23, 3.02)
TS, CS = (11, 9) if WIDE else (5.6, 4.6)      # 소제목 / 컬러바 눈금
fig, axes = plt.subplots(2, 2, figsize=FS)
half = N * side / 2
ext = [-half, half, -half, half]

def draw(ax, Z, title, cmap, log=False):
    Zp = np.flipud(Z.T)
    if log:
        v = Zp[Zp > 0]
        im = ax.imshow(np.where(Zp > 0, Zp, np.nan), extent=ext, cmap=cmap,
                       norm=LogNorm(vmin=max(v.min(), 1e-6), vmax=v.max()))
        zy, zx = np.where(~(Zp > 0))
        # 표식 크기는 포인트 단위라 화폭을 줄여도 그대로다. 단 폭에서는 칸보다
        # 커져 격자를 덮었다. 칸 대비 비율이 WIDE 와 같도록 화폭에 맞춰 줄인다.
        ax.scatter(ext[0] + (zx + .5) * side, ext[3] - (zy + .5) * side,
                   marker="x", s=15 if WIDE else 1.6, c="0.45",
                   lw=.9 if WIDE else .3)
    else:
        im = ax.imshow(Zp, extent=ext, cmap=cmap)
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(mission.search_radius_m * np.cos(th), mission.search_radius_m * np.sin(th),
            "k--", lw=1, alpha=.6)
    ax.set_title(title, fontsize=TS); ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=.045, pad=.02)
    if log:
        # 로그 눈금의 기본 표기는 mathtext(10^-1)라 한글 폰트에 없는 U+2212 를 불러
        # 빼기 기호가 두부로 찍혔다. 소수로 적으면 글리프 문제가 없고 더 짧다.
        cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        cb.ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    cb.ax.tick_params(labelsize=CS, length=2, width=.5)
    cb.outline.set_linewidth(.5)

draw(axes[0, 0], g2(u0), "(가) 탐색 시작 시점의 표적 존재확률   (× = 확률 0)" if WIDE else "(가) 시작 시점 존재확률 (× = 0)", "viridis", log=True)
draw(axes[0, 1], g2(uT), "(나) 탐색 종료 시점의 표적 존재확률   (탐색을 가하지 않은 경우)" if WIDE else "(나) 종료 시점 존재확률 (탐색 없음)", "viridis", log=True)
draw(axes[1, 0], g2(mob), "(다) 지형 통행성 가중", "YlOrBr")
draw(axes[1, 1], g2(con), "(라) 지형 은폐 가중", "Greens")
# 단 폭에서는 큰 제목이 지면만 먹는다. 같은 내용이 캡션에 있으므로 뺀다.
if WIDE:
    fig.suptitle(f"{CASE} {N}×{N} 격자의 계획 입력 — 셀 {side:.0f} m, 탐지폭 400 m (셀 한 변의 18 %)\n"
                 f"점선 = 관심구역 경계", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, .94 if WIDE else 1.0], pad=.35, w_pad=.5, h_pad=.7)
out = ROOT / "paper/fig/fig7_cell_layers.png"
fig.savefig(out, dpi=155 if WIDE else 460); print("저장:", out, FS)
print(f"확률 0 셀 {(u0 <= 0).sum()}/{G.cell_count} (시작) -> {(uT <= 0).sum()} (종료)")
