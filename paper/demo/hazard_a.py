
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
import sys; sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.font_manager
for c in ("Apple SD Gothic Neo","AppleGothic","NanumGothic","Malgun Gothic"):
    if c in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = c; break
plt.rcParams.update({"axes.unicode_minus": False, "font.size": 9.5})

a = np.linspace(0, 3, 400)
q = 1 - np.exp(-a)
PTS = [(.2236, "이동 중 한 셀", "#1f6feb", 1),
       (.5292, "도착 셀 (이동+체류)", "#c2410c", -1),
       (1.20,  "한 단계 총소인량을\n한 셀에 다 쏟으면", "#6b21a8", 1),
       (2.40,  "같은 셀 두 단계", "#15803d", -1)]

fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5), gridspec_kw={"width_ratios":[1.25,1]})
ax = axes[0]
ax.plot(a, a, color="0.7", ls="--", lw=1.2, label="a 를 그대로 확률로 읽으면 (틀림)")
ax.plot(a, q, color="k", lw=2.4, label="q = 1 - exp(-a)   실제 탐지확률")
ax.fill_between(a, q, np.minimum(a,1), color="crimson", alpha=.07)
for v, lbl, col, side in PTS:
    y = 1-np.exp(-v)
    ax.scatter([v],[y], s=62, color=col, zorder=5, ec="w", lw=1.2)
    ax.annotate(f"{lbl}\na={v:.3f} → q={y:.3f}", (v,y),
                xytext=(v + (.18 if side>0 else -.18), y + (-.17 if side>0 else .13)),
                ha="left" if side>0 else "right", fontsize=8.5, color=col, linespacing=1.4,
                arrowprops=dict(arrowstyle="-", color=col, lw=.8))
ax.axhline(1, color="0.85", lw=.8)
ax.set_xlim(0,3); ax.set_ylim(0,1.28)
ax.set_xlabel("hazard  a = 소인면적 S / 셀 면적 A", fontsize=10)
ax.set_ylabel("그 셀에 있는 표적의 탐지확률 q", fontsize=10)
ax.set_title("a 는 확률이 아니다 — 확률로 바뀌는 건 마지막 한 번뿐", fontsize=11, pad=8)
ax.legend(fontsize=8.6, frameon=False, loc="lower right")
ax.spines[["top","right"]].set_visible(False); ax.tick_params(labelsize=8.5)
ax.text(2.55, .40, "이 틈이\n‘포화’", color="crimson", fontsize=9, ha="center", linespacing=1.4)

ax = axes[1]
rows = [("한 번 훑음",      [.7],        "#1f6feb"),
        ("두 번 훑음",      [.7,.7],     "#c2410c"),
        ("두 대가 동시에",   [.7,.7],     "#6b21a8")]
for i,(lbl, parts, col) in enumerate(rows):
    tot = sum(parts); left = 0
    for j,pp in enumerate(parts):
        ax.barh(i, pp, left=left, height=.44, color=col, alpha=.85 if j==0 else .5,
                ec="w", lw=1.4)
        ax.text(left+pp/2, i, f"a={pp}", ha="center", va="center", color="w", fontsize=9,
                fontweight="bold")
        left += pp
    ax.text(tot+.06, i, f"합 a={tot:.1f}  →  q = {1-np.exp(-tot):.3f}",
            va="center", fontsize=9.5, color=col, fontweight="bold")
ax.set_yticks(range(3)); ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
ax.set_xlim(0, 2.45); ax.set_xlabel("누적 hazard a", fontsize=10)
ax.set_title("hazard 는 더하면 된다 — 확률은 못 더한다", fontsize=11, pad=8)
ax.spines[["top","right"]].set_visible(False); ax.tick_params(labelsize=8.5)
ax.invert_yaxis()
ax.text(.02, -.24, "확률로 더했다면 0.503 + 0.503 = 1.006 > 1  (불가능)\n"
                   "hazard 로 더하면 0.7 + 0.7 = 1.4  →  q = 0.753  (정상)",
        transform=ax.transAxes, fontsize=9, color="0.3", linespacing=1.6, va="top")
fig.suptitle("“각 스텝 한 번이 a 인가?” — 그렇다. a = S/A 이고, 단계마다·탐색자마다 더해진다",
             fontsize=12, y=1.0)
fig.tight_layout(rect=[0,.03,1,.94])
out = str(ROOT / "paper" / "fig" / "fig8_hazard_to_probability.png")
fig.savefig(out, dpi=185, bbox_inches="tight", facecolor="white"); print("저장:", out)
