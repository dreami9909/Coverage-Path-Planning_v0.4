"""학회 발표논문 2쪽 요약본 docx 조립.

전체 원고(build_docx.py)와는 목적이 다르다. 저쪽은 HTML 원고를 그대로 옮기고,
이쪽은 템플릿이 요구하는 2쪽·2단 분량에 맞추어 서론/본론/결론 세 장으로 압축한다.
따라서 본문은 이 파일이 들고 있다. 수치는 전부 전체 원고와 같은 값이어야 하므로
바꿀 일이 생기면 양쪽을 함께 고친다.

템플릿 규약: 캡션과 참고문헌은 영문, 캡션 첫 글자만 대문자, 마침표 없음.
"""
import pathlib, re, shutil
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = "/Users/otaegyeong/Downloads/워드템플릿.docx"
DST = str(ROOT / "paper/다중무인기_이동표적_경로계획_2쪽.docx")
FIG = ROOT / "paper/fig"

COL_TW = 4620          # 단 폭 [twips] ≈ 3.21 in
FRAME_W = 9639         # 표제부 프레임 폭 (2단 가로지르기)
BODY_LINE = 240        # 본문 줄 간격 [twips] = 12 pt

shutil.copyfile(SRC, DST)
doc = Document(DST)
body = doc.element.body
sectPr = body.find(qn("w:sectPr"))
for child in list(body):
    if child is not sectPr:
        body.remove(child)

# ---------------------------------------------------------------- 분량 집계
# 2쪽을 넘기면 학회가 반려한다. LibreOffice 가 없어 실제 조판을 볼 수 없으므로
# 쌓아 올린 블록의 높이를 직접 더해 둔다. 어림이지만 넘치는 원고는 잡아낸다.
COL_IN = 3.21                      # 단 폭
PAGE_COL_IN = 9.72                 # 본문 높이
BUDGET_IN = 2 * 2 * PAGE_COL_IN    # 2쪽 x 2단
_used = {"본문": 0.0, "그림": 0.0, "표": 0.0, "수식": 0.0,
         "표제부": 0.0, "초록": 0.0, "참고문헌": 0.0, "제목줄": 0.0}


def _em(text):
    """글자 폭 합 [em]. 한글·전각기호는 1, 나머지는 0.5 로 센다."""
    return sum(1.0 if ord(c) > 0x2E80 else 0.5 for c in text)


def _lines(text, size, *, latin=False, span=1):
    """줄 수를 어림한다. span 은 문단이 걸치는 단 수다."""
    return max(1, -(-_em(text) // (span * COL_IN * 72 / size)))


def _tally(bucket, inches):
    _used[bucket] += inches


def _set(el, tag, **attrs):
    e = OxmlElement(tag)
    for k, v in attrs.items():
        e.set(qn(f"w:{k}"), str(v))
    el.append(e)
    return e


def para(text="", *, font="바탕", size=9, bold=False, align=None,
         indent=False, frame=False, line=BODY_LINE, tabs=None, bucket=None,
         latin=False):
    p = doc.add_paragraph()
    pPr = p._p.get_or_add_pPr()
    if frame:
        f = OxmlElement("w:framePr")
        for k, v in (("w", FRAME_W), ("h", 301), ("wrap", "around"),
                     ("hAnchor", "page"), ("x", 1135), ("yAlign", "top")):
            f.set(qn(f"w:{k}"), str(v))
        pPr.append(f)
    _set(pPr, "w:spacing", line=line, lineRule="atLeast", after=0)
    if indent:
        _set(pPr, "w:ind", firstLineChars=100, firstLine=180)
    if align is not None:
        p.alignment = align
    else:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if tabs:
        t = OxmlElement("w:tabs")
        for pos, kind in tabs:
            tab = OxmlElement("w:tab")
            tab.set(qn("w:val"), kind); tab.set(qn("w:pos"), str(pos))
            t.append(tab)
        pPr.append(t)
    if text:
        r = p.add_run(text)
        r.font.name = font
        r.font.size = Pt(size)
        r.bold = bold
        rPr = r._element.get_or_add_rPr()
        rf = rPr.find(qn("w:rFonts"))
        if rf is None:
            rf = OxmlElement("w:rFonts"); rPr.insert(0, rf)
        for a in ("ascii", "eastAsia", "hAnsi"):
            rf.set(qn(f"w:{a}"), font)
    if bucket:
        span = 2 if frame else 1        # 프레임은 2단을 가로지른다
        n = _lines(text, size, latin=latin, span=span) if text else 1
        _tally(bucket, n * (line / 20 / 72) * span)
    return p


def head(text, *, bucket="제목줄"):
    para(text, font="돋움", size=9, bold=True, line=260,
         align=WD_ALIGN_PARAGRAPH.LEFT, bucket=bucket)


def eq(lines, number):
    """중앙 수식 + 우측 번호. `_{...}` 는 워드의 진짜 아래첨자로 조판한다.

    유니코드 아래첨자로 적으면 대문자 첨자(T, X, K)에 해당 글자가 없어
    ᴛ(작은 대문자) 같은 엉뚱한 글자가 들어간다. 첨자는 서식으로 준다.
    """
    token = re.compile(r"_\{([^}]*)\}")
    for i, line in enumerate(lines):
        p = para(align=WD_ALIGN_PARAGRAPH.CENTER, line=260,
                 tabs=[(COL_TW, "right")] if i == len(lines) - 1 else None)
        pos = 0
        for m in token.finditer(line):
            for text, sub in ((line[pos:m.start()], False), (m.group(1), True)):
                if not text:
                    continue
                r = p.add_run(text)
                r.font.name = "바탕"; r.font.size = Pt(9); r.font.subscript = sub
            pos = m.end()
        if line[pos:]:
            r = p.add_run(line[pos:]); r.font.name = "바탕"; r.font.size = Pt(9)
        if i == len(lines) - 1:
            r2 = p.add_run(f"\t({number})")
            r2.font.name = "바탕"; r2.font.size = Pt(9)
    _tally("수식", (len(lines) + 0.5) * 260 / 20 / 72)


def figure(stem, caption, width_in):
    hits = sorted(FIG.glob(f"{stem}_*.png"))
    if not hits:
        raise SystemExit(f"그림 없음: {stem}")
    png = hits[0]
    from PIL import Image
    w, h = Image.open(png).size
    para("", line=80)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(png), width=Inches(width_in))
    para(caption, font="돋움", size=8, align=WD_ALIGN_PARAGRAPH.CENTER,
         line=220, latin=True)
    para("", line=100)
    cap_in = _lines(caption, 8, latin=True) * 220 / 20 / 72
    _tally("그림", width_in * h / w + cap_in + 2 * (100 / 20 / 72))


def table(caption, widths, header, rows):
    para("", line=100)
    para(caption, font="돋움", size=8, align=WD_ALIGN_PARAGRAPH.CENTER,
         line=220, latin=True)
    t = doc.add_table(rows=1, cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    # 템플릿에 Table Grid 가 없다. 없으면 테두리를 직접 넣는다.
    try:
        t.style = "Table Grid"
    except KeyError:
        borders = OxmlElement("w:tblBorders")
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
            e = OxmlElement(f"w:{edge}")
            e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "4")
            e.set(qn("w:color"), "808080")
            borders.append(e)
        t._tbl.tblPr.append(borders)

    def fill(cells, values, bold):
        for c, v, wd in zip(cells, values, widths):
            c.width = Inches(wd)
            c.text = ""
            pr = c.paragraphs[0]
            pr.alignment = WD_ALIGN_PARAGRAPH.CENTER
            _set(pr._p.get_or_add_pPr(), "w:spacing", line=200,
                 lineRule="atLeast", after=0)
            name = "돋움" if bold else "바탕"
            r = pr.add_run(v)
            r.font.name = name
            r.font.size = Pt(7); r.bold = bold
            rPr = r._element.get_or_add_rPr()
            rf = OxmlElement("w:rFonts"); rPr.insert(0, rf)
            for a in ("ascii", "eastAsia", "hAnsi"):
                rf.set(qn(f"w:{a}"), name)

    fill(t.rows[0].cells, header, True)
    for row in rows:
        fill(t.add_row().cells, row, False)
    para("", line=100)
    _tally("표", (len(rows) + 1) * 200 / 20 / 72 + 2 * (220 / 20 / 72))


# ================================================================ 표제부
TITLE_KO = "다중 무인기의 이동표적 탐지 경로계획: 확률적 탐색이론 기반 접근"
TITLE_EN = ("Path Planning for Multi-UAV Moving-Target Detection: "
            "A Probabilistic Search-Theoretic Approach")
FIG_W = 3.12          # 그림 폭 [inch] — 단 폭(3.21)에 거의 꽉 채운다.
                      # 그림 원본이 단 폭 기준으로 그려져 있으므로 이 폭에서 글자가 제 크기로 앉는다.

para(TITLE_KO, font="돋움", size=13, bold=True,
     align=WD_ALIGN_PARAGRAPH.CENTER, frame=True, line=260, bucket="표제부")
para(TITLE_EN, font="돋움", size=9.5, align=WD_ALIGN_PARAGRAPH.CENTER,
     frame=True, line=240, bucket="표제부", latin=True)
para("홍길동*·성명2**", font="돋움", size=9, align=WD_ALIGN_PARAGRAPH.CENTER,
     frame=True, line=240, bucket="표제부")
para("* 소속1   ** 소속2   (발표자 이메일 주소)", font="돋움", size=8,
     align=WD_ALIGN_PARAGRAPH.CENTER, frame=True, line=220, bucket="표제부")
para("", frame=True, line=200, bucket="표제부")

# ================================================================ 초록
head("ABSTRACT")
para("A team of small unmanned aircraft sweeps only a fraction of an area of interest "
     "within its endurance and its motion is confined to a path, so search effort cannot "
     "be apportioned freely across cells. Paths are chosen by a mixed-integer program "
     "a grid-reduced target distribution, in which the sweep contribution of a move is "
     "distributed over every cell the flight path traverses. A cutting-plane scheme "
     "bounds each plan's distance from the best plan the model admits, and a cell-free "
     "evaluator measures the distance between that model and continuous space. The two "
     "are shown to be independent.",
     size=8, line=195, bucket="초록", latin=True)
para("Key Words : Cooperative Search, Search Theory, Path-Constrained Search, "
     "Cutting-Plane Method, Mixed-Integer Programming, Unmanned Aerial Vehicle",
     size=8, bold=True, line=195, bucket="초록", latin=True)

# ================================================================ 1. 서론
head("1. 서 론")
for t in [
    "확률에 기반한 탐색이론은 제2차 세계대전기 대잠전에서 출발하였다. Koopman의 탐지폭(sweep "
    "width)과 무작위탐색 모형을 Stone이 최적탐색 배분 이론으로 체계화하였고[1], 유실 수소폭탄 "
    "회수와 잠수함 Scorpion 탐색, 해안경비대 수색구조 체계에 쓰였다. 탐색 자원은 표적이 있을 법한 "
    "영역에 비해 언제나 작다.",

    "탐색 수단이 무인기로 옮겨 오면서 이 문제는 다시 현안이 되었다. 무인기는 체공시간과 탐지폭이 "
    "제한되어 탐색 면적이 작고 기동이 경로로 묶이므로, 고전 이론이 전제하는 효과의 자유로운 배분이 "
    "성립하지 않는다. 표적이 이동하면 정지 표적의 결과도 쓸 수 없다.",

    "이 문제는 정지 표적에 대해서도 NP-complete이며[3], 이동표적 문제가 이를 포함한다. 단일 탐색자의 분지한계 해법[2]과 복수 탐색자로의 혼합정수 확장[6]이 제시되어 "
    "Stone 등[1]이 정리하였다. 본 논문은 소인 효과를 경로가 통과한 셀 전체에 배분하는 형태로 이를 "
    "확장하고, 계획이 모형의 최적해로부터 떨어진 거리와 모형이 연속 공간으로부터 떨어진 거리를 함께 "
    "측정한다.",
]:
    para(t, indent=True, bucket="본문")

# ================================================================ 2. 본론
head("2. 본 론")
head("2.1 문제 정식화")
para("센서 성능은 측방 탐지확률 곡선의 적분인 탐지폭 W로 요약한다[1]. 면적 A에 소인면적 S가 "
     "축적되었을 때 무작위탐색 가정 하의 탐지확률은 1 − exp(−a), a = S/A이다. a는 확률이 아니라 "
     "탐지 위험률(hazard)의 시간 적분이므로 탐색자에 대해 가산적이고, 이 성질이 목적함수의 입력을 "
     "결정변수에 선형으로 유지한다.", indent=True, bucket="본문")
para("관심구역은 원에 외접하는 정사각 격자로 나눈다. 한 단계의 도달거리가 셀 한 변보다 짧으면 "
     "셀을 벗어날 수 없고 √2배를 넘으면 대각 이동이 열리므로, 셀 크기와 단계 수는 따로 정할 수 "
     "없다. 한 단계에 셀 한 변을 전진한다는 조건으로 단계 수를 유도하면 그 비가 1.07~1.20이어서 대각 "
     "이동이 배제되고 후속 셀은 최대 5개다.", indent=True,
     bucket="본문")
para("이동 중에도 센서는 작동한다. 소인을 도착 셀에만 주면 이동 구간에서 실제로 탐색된 셀에 "
     "hazard가 쌓이지 않는다(Fig. 1). 본 논문은 한 호가 통과한 각 셀에 탐지폭과 그 셀 내부 통과 "
     "길이의 곱을, 도착 셀에는 잔여 체류시간과 비행속도의 곱을 더해 셀 면적으로 나눈 값을 hazard "
     "계수로 쓴다. 지형은 센서가 아니라 표적 belief에 작용한다(2.3절).", indent=True, bucket="본문")
figure("fig1", "Fig. 1. Sweep effort assigned to the arrival cell only versus "
                "distributed over every traversed cell", FIG_W)
para("표적 종류는 식별되나 기동 양상은 미지인 조건을 고려하여 한 표적 유형을 기동 가설 k ∈ K로 "
     "나누고 최악 가설로 평가한다. 초기 확률분포 π, 전이행렬 P, 누적 hazard 장 Y에 대한 "
     "목적함수는 식 (1)이다.", indent=True, bucket="본문")
eq(["u_{1} = π ,    u_{t+1} = ( u_{t} ⊙ exp(−Y_{t}) ) P_{t}",
    "f_{k}(Y) = Σ_{i} u_{T}(i) · exp(−Y_{T,i})",
    "min_{X}  max_{k∈K}  f_{k}( Y(X) )"], 1)
para("X는 탐색자가 지나는 호를 나타내는 이진 결정변수이며, Y는 그 셀을 지나는 모든 호의 계수 "
     "합이므로 X에 대해 선형이다. 제약은 탐색자별·단계별 행동 1개, 흐름 보존, 셀 점유 한도, 양립 "
     "불가 이동이다.", indent=True, bucket="본문")

head("2.2 최적성 인증 해법")
para("f_k는 Y에 볼록하고 Y는 X에 선형이므로 f_k의 접평면은 하계를 이룬다. 보조변수 η로 "
     "Kelley[4] 및 Duran–Grossmann[5] 형태의 외부 근사 문제를 구성하며, 이는 Stone 등[1]의 "
     "절단평면 알고리즘과 동형이다. 기울기는 식 (1)의 전방 재귀와 대응하는 후방 재귀에서 한 번의 "
     "통과로 나오므로 표적 경로를 열거하지 않는다[7].", indent=True, bucket="본문")
para("각 반복에서 주문제의 최적값은 하계 L, 그 정수해를 식 (1)로 평가한 값은 상계 U가 되며 최적성 "
     "간격을 (U − L)/|U|로 정의한다. 주문제는 HiGHS[8]로 푼다. 셀 수가 늘면 주문제 규모가 격자 한 "
     "변의 3차로 증가해 인증 가능한 해상도의 상한을 만든다(Fig. 2).",
     indent=True, bucket="본문")
figure("fig3", "Fig. 2. Workflow for path planning and evaluation", FIG_W)

head("2.3 구현 검증과 가정")
para("구현은 네 경로로 독립 검증하였다. 완전열거로 구한 참 최적값과 솔버 상·하계의 대조, 표적 "
     "궤적을 직접 표집하는 몬테카를로와의 대조(최대 절대차 5.3×10⁻⁵), 기울기의 중심차분 "
     "대조(최대 차 2.4×10⁻¹¹), 접평면의 하계성 검사(위반 0건)이다.", indent=True, bucket="본문")
para("가정은 셀 내부의 균일·독립을 전제하는 무작위탐색, 셀 간 Markov 전이로의 축약, 표적이 "
     "탐색자를 인지하지 않는다는 전제, 단일 값으로 요약된 탐지폭과 오탐 미고려, 통신·항법 오차 0, "
     "그리고 사전 일괄 계획이다.", indent=True, bucket="본문")

head("2.4 결과")
para("Fig. 3은 계획이 무엇을 보고 결정되었는지(위)와 그 결과 계획이 어떻게 움직이는지(아래)를 함께 "
     "제시한 것이다. 확률 질량은 소수의 셀에 집중되어 있고 두 지형층이 그 분포를 만든다 — 통행성은 "
     "확산 방향을, 은폐는 사전질량의 쏠림을 정한다. 다만 두 지형층은 표적 belief에만 작용하며 셀별 "
     "hazard는 바꾸지 않는다.", indent=True, bucket="본문")
para("Table 1은 인증된 계획을 두 방식으로 평가한 결과다. MILP는 계획이 전제한 셀 모형 위의 값, MC는 "
     "같은 계획을 이산화 없이 연속 공간에서 직접 표집하여 얻은 값이다(조건마다 8만 표본 이상).",
     indent=True, bucket="본문")
table("Table 1. MILP solutions versus Monte-Carlo evaluation",
      [0.78, 0.49, 0.49, 0.49, 0.49, 0.47],
      ["조건", "탐지폭/셀", "MILP P_D", "최적성 간격", "MC P_D", "비"],
      [["Tank 4×4",   "9.0 %",  "0.3786", "0.011 %", "0.1868", "0.493"],
       ["Tank 6×6",   "13.5 %", "0.5981", "0.028 %", "0.4209", "0.704"],
       ["Tank 8×8",   "18.0 %", "0.7474", "0.810 %", "0.5699", "0.763"],
       ["TEL 10×10",  "20.1 %", "0.7283", "0.807 %", "0.6009", "0.825"],
       ["Tank 10×10", "22.5 %", "0.7971", "0.962 %", "0.6459", "0.810"]])
para("차이는 모든 조건에서 같은 방향이며 셀 모형이 탐지확률을 높게 본다. 그 차이는 최적성 간격과 "
     "무관하다. 간격이 가장 작은 조건에서 차이가 가장 컸고 가장 큰 조건에서 오히려 작아 상관은 "
     "−0.69였고, 차이의 비는 탐지폭 대 셀 한 변의 비에 단조 증가하여 +0.94였다. 간격이 작다는 "
     "것은 모형을 잘 풀었다는 뜻이지 모형이 현실에 가깝다는 뜻이 아니다. 같은 인스턴스에서 서로 다른 "
     "계산 조건으로 얻은 다섯 계획을 공통 난수로 채점하였을 때, 셀 모형이 최선으로 지목한 계획은 연속 "
     "모형에서 3위였고 연속 모형의 최선은 셀 모형 기준 4위였다. 짝지은 표준오차의 20배를 넘는 차이라 "
     "표집 변동으로 설명되지 않는다.", indent=True, bucket="본문")
para("Fig. 3의 아래쪽은 같은 계획의 단계별 점유다. 경로를 겹쳐 그리면 서로 다른 시각의 통과가 동시 "
     "점유처럼 보이나, 단계로 분리하면 동일 시점 동일 셀·동일 간선·정면 교환의 위반이 모두 0건임이 "
     "드러난다. 확률 질량이 몇 셀에 몰린 조건에서는 이동보다 체류가 더 많은 소인을 쌓아 제자리 호의 "
     "비율이 70 %에 이른다.", indent=True, bucket="본문")
para("동시에 이 그림은 위 한계를 드러낸다. 탐지폭이 셀 한 변의 18 %뿐이므로 한 셀의 hazard가 셀 "
     "전체에 균일하다는 전제는 이 규모에서 성립하지 않는다. 인증에도 실행 의존적 변동이 있어 동일 "
     "조건 재실행에서 간격이 2.19 %p, 입자 표본 교체에서 탐지확률이 0.005~0.010 변동하였다.",
     indent=True, bucket="본문")
figure("fig13", "Fig. 3. Cell-level weights (top) and planned occupancy by step (bottom)",
       FIG_W)

# ================================================================ 3. 결론
head("3. 결 론")
para("본 논문에서는 경로 제약을 갖는 다중 무인기의 이동표적 탐지 경로계획을 확률적 탐색이론 위에 "
     "구성하고, 외부 근사 절단평면법으로 계획과 최적성 간격을 산출하였으며, 연속 공간 "
     "평가기로 모형과 현실의 거리를 따로 측정하였다. 이산화는 자유 변수가 아니며 셀 크기가 단계 "
     "수를 정한다. 최적성 간격은 인증 기준과 같은 크기의 변동을 가져 기준 부근의 판정을 단일 "
     "실행으로 확정할 수 없고, 간격과 모형 충실도는 서로 독립이다.",
     indent=True, bucket="본문")
para("향후 과제는 2.3절의 가정에 대응한다. 확률 질량이 몰린 영역만 세분하는 적응적 격자와, 셀 "
     "내부의 소인 기하를 목적함수에 반영하는 정식화가 후보다. 표적이 탐색자를 인지한다고 보면 "
     "문제는 이인 영합 게임이 되어 최소최대 구조를 확장해 다룰 수 있다. 정적 계획에는 belief "
     "갱신을 반영하는 재계획이 자연스러우나 매 시점 혼합정수계획을 푸는 방식은 실시간 제약에 맞지 "
     "않으므로, 다중 에이전트 강화학습으로 정책을 미리 학습하는 접근이 의미를 가진다. 그 잣대는 "
     "셀 모형이 아니라 연속 공간 평가여야 한다.",
     indent=True, bucket="본문")

# ================================================================ 참고문헌
head("References")
REFS = [
 "L. D. Stone, J. O. Royset and A. R. Washburn, “Optimal Search for Moving Targets,” "
 "Springer, Cham, 2016.",
 "J. N. Eagle and J. R. Yee, “An Optimal Branch-and-Bound Procedure for the Constrained "
 "Path, Moving Target Search Problem,” Oper. Res., Vol. 38, No. 1, pp. 110-114, 1990.",
 "K. E. Trummel and J. R. Weisinger, “The Complexity of the Optimal Searcher Path "
 "Problem,” Oper. Res., 34(2), pp. 324-327, 1986.",
 "J. E. Kelley, Jr., “The Cutting-Plane Method for Solving Convex Programs,” J. SIAM, "
 "Vol. 8, No. 4, pp. 703-712, 1960.",
 "M. A. Duran and I. E. Grossmann, “An Outer-Approximation Algorithm for a Class of "
 "Mixed-Integer Nonlinear Programs,” Math. Program., Vol. 36, No. 3, pp. 307-339, 1986.",
 "J. O. Royset and H. Sato, “Route Optimization for Multiple Searchers,” Naval Res. "
 "Logist., Vol. 57, No. 8, pp. 701-717, 2010.",
 "S. S. Brown, “Optimal Search for a Moving Target in Discrete Time and Space,” Oper. "
 "Res., Vol. 28, No. 6, pp. 1275-1289, 1980.",
 "Q. Huangfu and J. A. J. Hall, “Parallelizing the Dual Revised Simplex Method,” Math. "
 "Program. Comput., 10(1), pp. 119-142, 2018.",
]
for i, r in enumerate(REFS, 1):
    para(f"[{i}] {r}", size=7, line=160, bucket="참고문헌", latin=True)

doc.save(DST)

total = sum(_used.values())
print(f"저장: {DST}")
print(f"  2쪽 용량 {BUDGET_IN:.1f} 단·in 중 {total:.1f} 사용 ({total / BUDGET_IN * 100:.0f} %)")
for k, v in sorted(_used.items(), key=lambda kv: -kv[1]):
    print(f"    {k:5s} {v:5.2f}")
if total > BUDGET_IN:
    print(f"  ** 2쪽 초과 추정: {total - BUDGET_IN:.2f} 단·in")
