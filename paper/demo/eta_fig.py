"""eta* 가 어떻게 정해지는가 — 절단이 쌓이며 하계가 올라가는 그림 (1차원 단면)."""
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
plt.rcParams.update({"axes.unicode_minus": False, "font.size": 9})

f  = lambda y: (y - 2.0) ** 2 / 8.0 + 0.25          # 볼록한 미탐지확률 (단면)
df = lambda y: (y - 2.0) / 4.0
FEAS = np.arange(0, 6)                               # 정수해가 만드는 이산 후보
ys = np.linspace(-0.2, 5.4, 400)

STEPS = [ [0.4], [0.4, 5.0], [0.4, 5.0, 2.9, 1.3] ]
TITLES = ["반복 1 — 절단 1개", "반복 2 — 절단 2개", "반복 4 — 절단 4개"]
UP, LO, CUT = "#a8542a", "#176a73", "#8b8b8b"

fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.3), sharey=True)
for ax, cuts, title in zip(axes, STEPS, TITLES):
    env = np.max([f(c) + df(c) * (ys - c) for c in cuts], axis=0)      # 접평면들의 상포락
    env_feas = np.max([f(c) + df(c) * (FEAS - c) for c in cuts], axis=0)
    ax.plot(ys, f(ys), color="k", lw=2.2, zorder=4, label="참 목적함수  max$_k$ f$_k$")
    for c in cuts:
        ax.plot(ys, f(c) + df(c) * (ys - c), color=CUT, lw=.9, ls="--", zorder=2)
        ax.scatter([c], [f(c)], s=16, color=CUT, zorder=5)
    ax.plot(ys, env, color=LO, lw=1.8, zorder=3, label="절단의 상포락 = master 가 보는 것")
    ax.fill_between(ys, env, f(ys), color=LO, alpha=.10, zorder=1)

    i = int(np.argmin(env_feas)); y_star = FEAS[i]
    L = env_feas[i]; U = f(y_star)
    ax.scatter(FEAS, [f(v) for v in FEAS], s=26, facecolor="w", ec="k", lw=.9, zorder=6)
    ax.scatter([y_star], [L], s=90, marker="v", color=LO, zorder=7)
    ax.scatter([y_star], [U], s=90, marker="^", color=UP, zorder=7)
    ax.annotate("", xy=(y_star, U), xytext=(y_star, L),
                arrowprops=dict(arrowstyle="<->", color="crimson", lw=1.5), zorder=7)
    ax.text(y_star - .25, (U + L) / 2, f"gap\n{U-L:.3f}", color="crimson",
            fontsize=8.5, va="center", ha="right", fontweight="bold")
    ax.axhline(f(2.0), color=UP, lw=.8, ls=":", zorder=1)
    ax.text(5.5, f(2.0) + .05, "f*", color=UP, fontsize=9, ha="right")
    ax.set_title(f"{title}\n"
                 f"η* = L = {L:.3f}   U = {U:.3f}", fontsize=9.5, pad=6)
    ax.set_xlim(-.3, 5.6); ax.set_ylim(-1.45, 1.62)
    ax.set_xlabel("결정변수가 만드는 hazard 장 Y  (1차원 단면)", fontsize=8.5)
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(labelsize=8)
axes[0].set_ylabel("미탐지확률 f", fontsize=9)
axes[0].legend(fontsize=8, frameon=False, loc="upper center", ncol=1)
axes[0].text(.02, .02, "흰 점 = 정수해가 만드는 후보", transform=axes[0].transAxes,
             fontsize=8, color="0.35")
fig.suptitle("η* 는 어떻게 정해지는가 — 접평면의 상포락을 실행가능집합 위에서 최소화한 값", fontsize=11.5, y=1.0)
fig.tight_layout(rect=[0, 0, 1, .93])
out = str(ROOT / "paper" / "fig" / "fig11_eta_star.png")
fig.savefig(out, dpi=190, bbox_inches="tight", facecolor="white")
print("저장:", out)
for cuts, title in zip(STEPS, TITLES):
    env_feas = np.max([f(c) + df(c) * (FEAS - c) for c in cuts], axis=0)
    i = int(np.argmin(env_feas))
    print(f"  {title}: 고른 해 Y={FEAS[i]}  L={env_feas[i]:.4f}  U={f(FEAS[i]):.4f}  gap={f(FEAS[i])-env_feas[i]:.4f}")
