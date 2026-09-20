"""HTML 원고의 그림을 base64 로 박아 넣는다 — 한 파일로 떼어 놓아도 보이게.

그림 1~4 는 인라인 SVG 라 어디서 열어도 보였지만, 5 번부터는 ``fig/`` 상대경로
``<img>`` 였다. 원고를 폴더 밖으로 옮기거나 미리보기로 열면 전부 깨진다.
이 스크립트가 ``src`` 를 data URI 로 바꾸고 원본 파일명을 ``data-fig`` 에 남긴다.
남겨 둔 덕에 **여러 번 돌려도 안전하다** — 두 번째부터는 그 파일을 다시 읽어
갱신한다. 그림을 다시 그린 뒤 반드시 이것을 돌려야 원고에 반영된다.
"""
import base64, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
HTML = ROOT / "paper/협동탐색_최적성_인증_초고.html"
FIG = ROOT / "paper/fig"

html = HTML.read_text(encoding="utf-8")
IMG = re.compile(r'<img\b([^>]*?)>', re.S)
SRC = re.compile(r'\ssrc="([^"]*)"')
DATA = re.compile(r'\sdata-fig="([^"]*)"')

missing, embedded, total_bytes = [], [], 0


def rewrite(match: re.Match) -> str:
    global total_bytes
    attrs = match.group(1)
    named = DATA.search(attrs)
    src = SRC.search(attrs)
    if named:
        name = named.group(1)
    elif src and not src.group(1).startswith("data:"):
        name = src.group(1).split("/")[-1]
    else:
        return match.group(0)          # 이름을 모르면 건드리지 않는다
    path = FIG / name
    if not path.exists():
        missing.append(name)
        return match.group(0)
    raw = path.read_bytes()
    total_bytes += len(raw)
    uri = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
    attrs = SRC.sub("", attrs)
    attrs = DATA.sub("", attrs)
    embedded.append(name)
    return f'<img src="{uri}" data-fig="{name}"{attrs}>'


out = IMG.sub(rewrite, html)
HTML.write_text(out, encoding="utf-8")
print(f"내장 {len(embedded)}장, 원본 {total_bytes/1024:.0f} KB "
      f"-> 원고 {len(out.encode())/1024/1024:.2f} MB")
for n in embedded:
    print("   ", n)
if missing:
    print("※ 파일을 못 찾아 건너뜀:", ", ".join(missing))
    sys.exit(1)
