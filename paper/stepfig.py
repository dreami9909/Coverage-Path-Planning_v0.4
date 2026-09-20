"""Fig. 5 — 단계별 점유. 동일한 연속 단계는 하나로 묶는다.

경로를 한 장에 겹쳐 그리면 서로 다른 시각의 통과가 동시 점유처럼 보인다.
단계마다 격자를 하나씩 두면 셀당 점이 몇 개인지 바로 보인다. 제자리 호가
70 % 이므로 연속 단계가 완전히 같은 구간이 많고, 그런 구간은 한 칸으로 묶는다.
"""
import sys, json, pathlib, collections
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

for cand in ("Apple SD Gothic Neo", "AppleGothic", "NanumGothic", "Malgun Gothic"):
    if cand in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = cand
        plt.rcParams["axes.unicode_minus"] = False
        break

R = ROOT / "results/studies"
N = 8
paths = [[int(c) for c in p] for p in json.load(open(R / "occupancy-1.json"))["assignments"]]
T, J = len(paths[0]), len(paths)

# 표적 존재확률을 옅게 깔아 배경으로 쓴다
bel = None
try:
    from cpp_search.config import load_chapter_config
    from cpp_search.options import RunOptions
    from cpp_search.chapters.ch2_stone_spx import (
        build_instance, case_mission, case_specs, case_target_specs)
    cfg = load_chapter_config("2"); m0, sensor = cfg.mission(), cfg.sensor()
    base = next(g for g in cfg.get("stone_spx.grid_conditions", []) if g.get("grid_kind") == "square")
    block = next(c for c in case_specs(cfg) if c["name"] == "Tank")
    inst = build_instance(cfg, RunOptions(sample_count=10, particle_count=5000, episode_count=1,
            target_profile="both", seed=1, map_error=False, communication_loss_probability=0.0,
            communication_latency_slices=0, mission_time_s=600.0),
        case_mission(m0, block), sensor, grid={**base, "grid_width": N, "grid_height": N},
        seed=20260201, terrain_weighting=True, targets=case_target_specs(cfg, block))
    names = [t.split("/")[-1] for t in inst.target_names]
    bel = np.asarray(inst.target_models[names.index("STOP_HEAVY")].initial_mass,
                     float)[:N * N].reshape(N, N)
except Exception as exc:                      # 그림은 belief 없이도 뜻이 통한다
    print("belief 생략:", type(exc).__name__, exc)

# 연속 단계 중 전원 위치가 같은 구간을 묶는다
groups = []
for t in range(T):
    if groups and all(paths[j][t] == paths[j][groups[-1][-1]] for j in range(J)):
        groups[-1].append(t)
    else:
        groups.append([t])

cols = [plt.cm.tab10(j) for j in range(J)]

# 본문은 2단이라 그림이 단 폭(약 3.2 in)까지 줄어든다. 묶음 7 개를 4 열로 늘어놓으면
# 한 칸이 0.6 in 이 되어 기체 번호가 읽히지 않는다. 단 폭에서는 2x2 로 짜고
# 대표 묶음만 고른다 — 첫 단계, 마지막 단계, 그리고 가장 긴 정지 구간들.
WIDE = "--wide" in sys.argv
if not WIDE and len(groups) > 4:
    keep = {0, len(groups) - 1}
    for idx in sorted(range(1, len(groups) - 1),
                      key=lambda i: (-len(groups[i]), i)):
        if len(keep) >= 4:
            break
        keep.add(idx)
    shown = [groups[i] for i in sorted(keep)]
else:
    shown = groups

ncol = min(4 if WIDE else 2, len(shown))
nrow = -(-len(shown) // ncol)
FS = (3.15 * ncol, 3.2 * nrow) if WIDE else (3.23, 3.23 / ncol * nrow + 0.30)
TS, BS, NS = (10.5, 120, 7.5) if WIDE else (6.0, 46, 4.4)
fig, axes = plt.subplots(nrow, ncol, figsize=FS, squeeze=False)
for k, g in enumerate(shown):
    ax = axes[k // ncol][k % ncol]
    t = g[0]
    if bel is not None:
        Z = np.flipud(bel.T)
        ax.imshow(np.where(Z > 0, Z, np.nan), extent=[0, N, 0, N], cmap="Greys",
                  norm=LogNorm(vmin=max(bel[bel > 0].min(), 1e-6), vmax=bel.max()), alpha=.45)
    for x in range(N + 1): ax.axvline(x, color="0.85", lw=.5)
    for y in range(N + 1): ax.axhline(y, color="0.85", lw=.5)
    seen = {}
    for j in range(J):
        c = paths[j][t]; cx, cy = c % N, c // N
        d = seen.get(c, 0); seen[c] = d + 1
        ax.scatter(cx + .5 + .16 * d, cy + .5, s=BS, color=cols[j], ec="k", lw=.5, zorder=4)
        ax.text(cx + .5 + .16 * d, cy + .5, str(j + 1), fontsize=NS, ha="center", va="center",
                color="w", fontweight="bold", zorder=5)
    dup = sum(v - 1 for v in seen.values())
    label = f"단계 {g[0]}" if len(g) == 1 else f"단계 {g[0]}–{g[-1]}" + ("  (위치 동일)" if WIDE else "")
    ax.set_title(label + ("   중복 발생" if dup else ""), fontsize=TS,
                 color=("crimson" if dup else "k"))
    ax.set_xlim(0, N); ax.set_ylim(0, N); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
for k in range(len(shown), nrow * ncol):
    axes[k // ncol][k % ncol].axis("off")
# 단 폭에서는 큰 제목이 지면만 먹는다. 같은 내용을 캡션이 진다.
if WIDE:
    fig.suptitle(f"단계별 점유 — 전 단계에서 한 셀에 두 대 이상이 놓이지 않는다\n"
                 f"숫자 = 무인기 번호 · 회색 = 표적 존재확률 · 전원 위치가 같은 연속 단계는 한 칸으로 묶음",
                 fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, .93 if WIDE else 1.0], pad=.3, w_pad=.6, h_pad=.8)
out = ROOT / "paper/fig/fig5_occupancy_steps.png"
fig.savefig(out, dpi=150 if WIDE else 460); print("저장:", out, FS)
print("전체 묶음:", [f"{g[0]}" if len(g) == 1 else f"{g[0]}-{g[-1]}" for g in groups])
print("수록 묶음:", [f"{g[0]}" if len(g) == 1 else f"{g[0]}-{g[-1]}" for g in shown])
