# Coverage Path Planning — 연구 실행 진입점
#
# 전제: PYTHONPATH 를 따로 잡지 않아도 되도록 `make install` 로 editable 설치한다.
# 설치 없이 쓰려면 모든 타깃이 PYTHONPATH=src 를 스스로 붙이므로 그대로 동작한다.

PY      ?= python3
RUN      = PYTHONPATH=src $(PY) -m cpp_search
PYTEST   = PYTHONPATH=src $(PY) -m pytest
RESULTS  = results

.DEFAULT_GOAL := help
.PHONY: help install ch1 ch2 ch3 ch4 all test test-fast paper figures docs clean clean-results check

## ---------------------------------------------------------------- 도움말
help:   ## 타깃 목록
	@grep -hE '^[a-z][a-zA-Z0-9_-]*:.*?## ' $(MAKEFILE_LIST) \
	 | awk -F':.*?## ' '{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

## ---------------------------------------------------------------- 설치
install:  ## editable 설치 (이후 PYTHONPATH 불필요)
	$(PY) -m pip install -e .

## ---------------------------------------------------------------- 챕터 실행
# 산출: results/chN/chapter-N-result.json  +  results/chN/figures/
ch1:    ## Chapter 1 평가 계약 · 탐색기 스펙
	@mkdir -p $(RESULTS)/ch1
	$(RUN) --chapter 1

ch2:    ## Chapter 2 SPX 최적성 인증 (오래 걸린다)
	@mkdir -p $(RESULTS)/ch2
	$(RUN) --chapter 2

ch3:    ## Chapter 3 MAPPO 대 SPX (같은 인스턴스)
	@mkdir -p $(RESULTS)/ch3
	$(RUN) --chapter 3

ch4:    ## Chapter 4 합성지형 실비행
	@mkdir -p $(RESULTS)/ch4
	$(RUN) --chapter 4

all:    ## 1~4 순차 실행 + results/summary.json
	@mkdir -p $(RESULTS)
	$(RUN) --chapter all --output $(RESULTS)

## ---------------------------------------------------------------- 검증
test:       ## 전체 테스트 (약 5분)
	$(PYTEST) tests/ -q

test-fast:  ## 챕터를 실제로 돌리는 두 파일을 뺀 빠른 검사 (약 2분)
	$(PYTEST) tests/ -q \
	  --ignore=tests/test_experiments_smoke.py \
	  --ignore=tests/test_figure_generation.py

check:      ## 정합성 점검 — 저장소 밖 절대경로와 낡은 이름이 남았는지
	@echo "── 저장소 밖 절대경로"
	@! grep -rnE "/private/tmp|/Users/[a-z]+/Downloads/Coverage" \
	    --include="*.py" --include="*.sh" --include="*.json" --include="*.md" . \
	  || (echo "  ↑ 저장소 밖 경로가 남아 있다"; exit 1)
	@echo "  없음"
	@echo "── 낡은 모듈명 (chapterN_*)"
	@! grep -rn "chapters\.chapter[1-4]_" --include="*.py" . \
	  || (echo "  ↑ 개명 전 이름이 남아 있다"; exit 1)
	@echo "  없음"
	@echo "── MATH.md 가 코드와 붙어 있는가"
	@$(PY) tools/gen_math_doc.py 2>&1 | grep -c MISS | xargs -I{} sh -c '[ {} -eq 0 ] && echo "  MISS 0" || (echo "  MISS {} 건 — make docs 로 재생성"; exit 1)'

## ---------------------------------------------------------------- 문서·논문
docs:       ## docs/MATH.md 재생성 (수식 ↔ 코드 연결 검사 포함)
	$(PY) tools/gen_math_doc.py

figures:    ## 논문 그림 재생성 → paper/fig/
	$(PY) paper/make_figs.py

paper:      ## 그림 + 전체 원고 docx + 학회 2쪽 요약본 빌드
	./paper/refresh.sh

## ---------------------------------------------------------------- 정리
clean:          ## 캐시·OS 잡파일 제거 (결과는 남긴다)
	@find . -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
	@find . \( -name "*.pyc" -o -name ".DS_Store" \) -delete 2>/dev/null || true
	@rm -rf .pytest_cache
	@echo "캐시 제거 완료"

clean-results:  ## 실행 산출물 전부 제거 (studies 는 남긴다 — 재현에 수 시간)
	@rm -rf $(RESULTS)/ch1 $(RESULTS)/ch2 $(RESULTS)/ch3 $(RESULTS)/ch4 $(RESULTS)/summary.json
	@echo "챕터 산출물 제거 — results/studies/ 는 보존"
