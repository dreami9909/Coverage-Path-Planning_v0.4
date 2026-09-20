"""논문 그림 5점을 결과 json 에서 다시 만든다.

새 인증 결과가 results/studies/ 에 떨어질 때마다 이 스크립트를 다시 돌리면
Fig. 4(인증 계획)와 Fig. 5(해상도 스윕)가 갱신된다. Fig. 1~3 은 개념도라 불변이다.
2단 조판이므로 단 폭(약 8.2 cm)에 맞춘다. 그림 안 글자는 한글로 둔다 —
본문이 한글이므로 영문 라벨은 읽는 흐름을 끊는다.
"""
import json, glob, os
import numpy as np
import matplotlib
import matplotlib.font_manager
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.ticker import FuncFormatter

def _korean_font():
    """설치된 한글 폰트를 고른다. 없으면 기본 폰트로 떨어진다(라벨이 깨진다)."""
    have = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    for c in ("Apple SD Gothic Neo", "AppleGothic", "NanumGothic", "Malgun Gothic"):
        if c in have:
            return c
    return "DejaVu Sans"


plt.rcParams.update({
    "font.family": _korean_font(), "font.size": 7.2, "axes.unicode_minus": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.labelsize": 6.6, "ytick.labelsize": 6.6, "axes.labelsize": 7.2,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    # 한글 폰트에는 U+2212(−) 가 없어 mathtext 눈금의 빼기 기호가 두부로 찍혔다.
    "mathtext.fontset": "dejavusans",
})
W = 3.23          # 단 폭 [inch] ~ 8.2 cm
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]   # 저장소 최상위
OUT = str(ROOT / "paper" / "fig")
INK, UP, LO = "#1a1a1a", "#a8542a", "#176a73"
GREY = "#5b6472"


# ------------------------------------------------- Fig 1
fig, axes = plt.subplots(1, 2, figsize=(W, 1.35))
for ax, (title, shade, lab) in zip(axes, [
        ("Destination only", (0.0, 1.0), ("0 %", "100 %")),
        ("Path-distributed", (0.46, 0.54), ("46 %", "54 %"))]):
    for k in (0, 1):
        ax.add_patch(Rectangle((k, 0), 1, 1, facecolor=LO, alpha=0.10 + 0.5*shade[k],
                               edgecolor=INK, linewidth=0.6))
        ax.text(k + 0.5, -0.22, lab[k], ha="center", va="top", fontsize=6.4, color=INK)
    ax.annotate("", xy=(1.5, 0.5), xytext=(0.5, 0.5),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color=INK))
    ax.plot(0.5, 0.5, "o", ms=3, color=INK)
    ax.set_title(title, fontsize=6.9, pad=3)
    ax.set_xlim(-0.05, 2.05); ax.set_ylim(-0.5, 1.05); ax.axis("off")
fig.savefig(f"{OUT}/fig1_sweep_allocation.png"); plt.close(fig)


# ------------------------------------------------- Fig 2
fig, ax = plt.subplots(figsize=(W, 1.55))
it = np.arange(1, 8)
u = np.array([0.47, 0.40, 0.372, 0.360, 0.355, 0.3531, 0.3530])
l = np.array([0.300, 0.318, 0.331, 0.339, 0.346, 0.3505, 0.3521])
ax.plot(it, u, "-o", ms=2.6, lw=1.2, color=UP, label="Upper bound $U$")
ax.plot(it, l, "-o", ms=2.6, lw=1.2, color=LO, label="Lower bound $L$")
ax.fill_between(it, l, u, color=LO, alpha=0.10)
ax.annotate("gap", xy=(6.05, (u[5]+l[5])/2), fontsize=6.4, color=INK)
ax.set_xlabel("Cutting-plane iteration"); ax.set_ylabel("Non-detection probability")
ax.legend(frameon=False, fontsize=6.4, loc="upper right", handlelength=1.4)
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(f"{OUT}/fig2_bounds.png"); plt.close(fig)


# ------------------------------------------------- Fig 3
# 세로로 길면 2단 조판에서 한 단의 3분의 1을 먹는다. 상자 배치는 그대로 두고
# 화폭만 낮춰 여백을 줄인다 — 상자 높이는 글자의 두 배 이상으로 남는다.
fig, ax = plt.subplots(figsize=(W, 2.52))
def box(x,y,w,h,text,sub=None,ec=INK):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.004,rounding_size=0.008",
                 linewidth=0.6,edgecolor=ec,facecolor="#f7f8fa"))
    ax.text(x+w/2, y+h*(0.63 if sub else 0.5), text, ha="center", va="center", fontsize=5.8)
    if sub: ax.text(x+w/2, y+h*0.27, sub, ha="center", va="center", fontsize=5.0, color=GREY)
def arr(x0,y0,x1,y1,c=GREY,rad=0.0):
    ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle="-|>",mutation_scale=5,
                 lw=0.65,color=c,connectionstyle=f"arc3,rad={rad}"))
box(0.10,0.910,0.80,0.072,"Mission conditions and search area")
arr(0.50,0.910,0.50,0.878)
box(0.06,0.762,0.42,0.104,"Discretize the space","grid and time steps")
box(0.52,0.762,0.42,0.104,"Model target motion","propagate the belief")
arr(0.27,0.762,0.27,0.730); arr(0.73,0.762,0.73,0.730)
box(0.06,0.614,0.88,0.104,"Evaluate the detection effect","per-move sweep")
arr(0.50,0.614,0.50,0.582)
box(0.10,0.494,0.80,0.072,"Generate an initial plan")
arr(0.50,0.494,0.50,0.462)
ax.add_patch(Rectangle((0.02,0.170),0.96,0.292,fill=False,lw=0.7,edgecolor=LO,linestyle=(0,(3,2))))
ax.text(0.50,0.444,"iterate",ha="center",fontsize=5.4,color=LO,
        bbox=dict(fc="white",ec="none",pad=0.6))
box(0.06,0.330,0.42,0.098,"Optimize approximation","lower bound $L$")
box(0.52,0.330,0.42,0.098,"Evaluate the plan","upper bound $U$")
arr(0.48,0.379,0.52,0.379,LO)
arr(0.73,0.330,0.27,0.330,LO,rad=-0.34)
ax.text(0.50,0.196,"add a tangent, refine the approximation",ha="center",fontsize=5.2,color=GREY)
arr(0.50,0.170,0.50,0.138)
box(0.06,0.020,0.88,0.104,"Check the optimality gap","certify, or report the budget exhausted")
ax.set_xlim(0,1); ax.set_ylim(0.0,1.0); ax.axis("off")
fig.savefig(f"{OUT}/fig3_workflow.png"); plt.close(fig)



# ------------------------------------------------- Fig 4
d=json.load(open(ROOT / "results/studies/fig4-trend-data.json"))
n=d["n"]; belief=np.array(d["belief"]); paths=d["paths"]; starts=d["starts"]
fig,ax=plt.subplots(figsize=(W,W*1.04))
ax.imshow(np.sqrt(belief/belief.max()),cmap="Greys",vmin=0,vmax=1.25,
          extent=[0,n,n,0],interpolation="nearest")
for i in range(n+1):
    ax.axhline(i,color="#c8ced6",lw=0.35); ax.axvline(i,color="#c8ced6",lw=0.35)
cols=["#1f6f7a","#a8542a","#3a5a9c","#7a6220","#8c3f5e","#2f7a4e"]
offs=[(-.16,-.16),(.16,-.16),(-.16,.16),(.16,.16),(0,-.22),(0,.22)]
for s,path in enumerate(paths):
    ox,oy=offs[s%6]
    xy=[(c%n+.5+ox,c//n+.5+oy) for c in [starts[s]]+list(path)]
    ax.plot([p[0] for p in xy],[p[1] for p in xy],"-",lw=1.0,color=cols[s],zorder=3)
    ax.plot(xy[-1][0],xy[-1][1],"o",ms=2.6,color=cols[s],zorder=4)
    ax.plot(xy[0][0],xy[0][1],"o",ms=6.2,mfc="white",mec=cols[s],mew=1.0,zorder=5)
    ax.text(xy[0][0],xy[0][1],str(s+1),ha="center",va="center",fontsize=4.9,color=cols[s],zorder=6)
ax.plot(n/2,n/2,"+",ms=9,mew=1.2,color=INK,zorder=7)
ax.plot(n/2,n/2,"o",ms=7,mfc="none",mec=INK,mew=1.0,zorder=7)
ax.annotate("TP",xy=(n/2,n/2),xytext=(n/2+1.15,n/2+1.1),fontsize=6.4,color=INK,zorder=8,
            bbox=dict(fc="white",ec="none",pad=0.8),
            arrowprops=dict(arrowstyle="-",lw=0.6,color=INK,shrinkA=1,shrinkB=6))
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel("darker cells carry more target location probability",fontsize=6.2)
fig.savefig(f"{OUT}/fig4_certified_plan.png"); plt.close(fig)

# ------------------------------------------------- Fig 6 (해상도 스윕, 두 케이스)
SWEEP = {
    "Tank": [(4, 0.3786, 0.011), (6, 0.5981, 0.028), (8, 0.7474, 0.810),
             (10, 0.7952, 2.709), (12, 0.8315, 11.068), (14, 0.8401, 31.109)],
    "TEL":  [(6, 0.5121, 0.069), (8, 0.6691, 0.413), (10, 0.7283, 0.807),
             (12, 0.7761, 7.035), (14, 0.7981, 17.994)],
}
fig, (a1, a2) = plt.subplots(2, 1, figsize=(W, 2.7), sharex=True,
                             gridspec_kw={"hspace": 0.14})
for name, mark, col in (("Tank", "o", UP), ("TEL", "s", LO)):
    n = [r[0] for r in SWEEP[name]]
    a1.plot(n, [r[1] for r in SWEEP[name]], "-", marker=mark, ms=3, lw=1.2,
            color=col, label=f"사례 {'A' if name == 'Tank' else 'B'}")
    a2.plot(n, [r[2] for r in SWEEP[name]], "-", marker=mark, ms=3, lw=1.2, color=col)
a1.set_ylabel("탐지확률"); a1.set_yticklabels([])
a1.legend(frameon=False, fontsize=5.8, loc="lower right")
a1.spines[["top", "right"]].set_visible(False)
a1.text(0.035, 0.86, "(가)", transform=a1.transAxes, fontsize=6.8)
a2.axhline(1.0, color="0.45", lw=0.9, ls=(0, (4, 2)))
a2.text(0.98, 0.56, "인증 기준 1 %", transform=a2.transAxes,
        ha="right", va="bottom", fontsize=5.8, color="0.35")
a2.set_yscale("log"); a2.set_ylabel("최적성 간격 [%]")
# 로그 눈금의 기본 표기는 mathtext(10^-1)이고 그 빼기 기호는 U+2212 다.
# 한글 폰트에 그 글리프가 없어 두부로 찍혔다. 소수로 적으면 문제가 없다.
a2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
a2.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
a2.set_xlabel("격자 해상도 $N$  ($N \\times N$)")
a2.set_xticks([4, 6, 8, 10, 12, 14]); a2.spines[["top", "right"]].set_visible(False)
a2.text(0.035, 0.86, "(나)", transform=a2.transAxes, fontsize=6.8)
fig.savefig(f"{OUT}/fig6_resolution.png"); plt.close(fig)


print("그림 5점 생성 완료")
