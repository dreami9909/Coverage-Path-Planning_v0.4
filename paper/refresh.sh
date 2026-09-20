#!/bin/zsh
# 새 인증 결과를 반영해 그림과 논문을 다시 만든다.
# 경로는 스크립트 위치 기준이다 — 저장소를 옮기거나 이름을 바꿔도 동작한다.
#
# 순서가 중요하다. 그림을 먼저 그리고, HTML 에 박아 넣고, 그다음 docx 를 만든다.
# embed_figs.py 를 건너뛰면 원고를 폴더 밖에서 열었을 때 그림이 깨진다.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 paper/make_figs.py        # 그림 1~6
python3 paper/stepfig.py          # 그림 5 (단계별 점유)
python3 paper/cellfig.py          # 그림 7 (셀 단위 입력)
python3 paper/combfig.py          # 그림 13 (2쪽 요약본 전용 합본)
python3 paper/embed_figs.py       # HTML 을 자체 포함 문서로

[ -f paper/build_docx.py ]  && python3 paper/build_docx.py
# 학회 2쪽 요약본. 전체 원고와 같은 그림·같은 수치를 쓰므로 함께 갱신한다.
[ -f paper/build_brief.py ] && python3 paper/build_brief.py
echo "갱신 완료: $ROOT/paper"
