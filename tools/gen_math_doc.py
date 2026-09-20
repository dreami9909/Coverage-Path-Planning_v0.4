"""docs/MATH.md 생성 — 수식 -> 구현 위치(파일:줄) 색인과 의존 그래프.

표는 손으로 선언한 (분류, 수식, 파일, 찾을 문자열) 목록에서 나오고, 줄 번호만
자동으로 찾는다. 의존 그래프는 전부 자동 추출이다.

찾을 문자열이 사라지면 MISS 로 보고한다. 수식이 코드에서 없어졌거나 구현이
옮겨갔다는 신호이므로, 조용히 표에서 빠지게 두지 않는다.

실행: 저장소 최상위에서 ``python3 tools/gen_math_doc.py``
"""

import ast
import pathlib

ROOT = pathlib.Path("src/cpp_search")

# (분류, 수식, 파일, 찾을 문자열)
ENTRIES = [
    # --- Chapter 1: 평가도 / 탐색기 스펙 ---
    ("Ch1 센서·탐색범위", "swath = 2 H tan(HFOV/2)", "core/models.py", "def instantaneous_swath_m"),
    ("Ch1 센서·탐색범위", "s = W (1 - overlap)", "core/models.py", "return self.effective_sweep_width_m * (1.0 - self.overlap_ratio)"),
    ("Ch1 센서·탐색범위", "PD(x) = e + (1-e)(1+cos(pi u))/2", "core/search_envelope.py", "relative = self.edge_relative_probability"),
    ("Ch1 센서·탐색범위", "W = ∫ PD(x) dx  (Simpson)", "core/search_envelope.py", "weighted_sum = values[0] + values[-1]"),
    ("Ch1 센서·탐색범위", "weave 호 길이 ∫ sqrt(1+(A k cos)^2) ds", "core/search_envelope.py", "def integrand(centerline_m: float) -> float:"),
    ("Ch1 센서·탐색범위", "v_centerline = v_actual / (L(lambda)/lambda)", "core/search_envelope.py", "def progress_speed_mps"),
    ("Ch1 확률 계약", "l = ln(p/(1-p)), p = sigmoid(l)", "core/belief.py", "def _logit"),
    ("Ch1 확률 계약", "l += ln(PD/PF) / ln((1-PD)/(1-PF))", "core/belief.py", "increment = np.log(pd / pf)"),
    ("Ch1 확률 계약", "POC / POS / POD", "core/belief.py", "poc = float(normalized[searched].sum())"),
    ("Ch1 자원 상한", "A_max = n v T W, ceiling = A_max/(pi R^2)", "chapters/ch1_evaluation.py", "def _coverage_feasibility"),
    ("Ch1 AOI 사이징", "포함률 기반 탐색반경", "aoi.py", "def containment_radius_m"),

    # --- 표적 운동 · 사전확률 · 입자필터 ---
    ("표적 운동", "P = alpha I + (1-alpha) 1 pi^T", "core/motion.py", "def transition_matrix"),
    ("표적 운동", "CTRV 전파 (v/omega)(sin psi' - sin psi)", "core/motion.py", "def propagate_kinematics"),
    ("표적 운동", "원 밖 흡수 처리", "core/motion.py", "def apply_circular_boundary"),
    ("사전확률·지도", "f(r) = exp(-(r-mu)^2 / 2 sigma^2)", "core/probability.py", "standardised = (radius_m - mean_radius_m) / sigma_m"),
    ("사전확률·지도", "A_cell = (r_out^2 - r_in^2) dtheta / 2", "core/probability.py", "cell_area = 0.5 * (outer_radius**2"),
    ("입자필터", "ESS = 1 / sum w_i^2", "core/particle_filter.py", "squared_weight_sum = sum(weight * weight"),
    ("입자필터", "systematic / stratified 재표본추출", "core/particle_filter.py", "def _resample(self) -> None:"),
    ("입자필터", "roughening sigma = K E N^(-1/2)", "core/particle_filter.py", "sigma = (\n            self.config.roughening_gain"),
    ("입자필터", "음성관측 w <- w exp(-Lambda)", "core/particle_filter.py", "def observe_no_detection_segments"),
    ("입자필터", "입자 -> 셀 질량 투영", "core/particle_filter.py", "def cell_masses"),

    # --- EO/IR 탐지 · 지형 · 경로 KPI ---
    ("EO/IR 탐지", "GSD = 2R tan(HFOV/2)/W_px, n_px", "core/sensor_observation.py", "ground_width_m = 2.0 * range_m * tan("),
    ("EO/IR 탐지", "SNR = SNR_ref (R_ref/R)^2 contrast", "core/sensor_observation.py", "snr = (\n            config.snr_at_reference"),
    ("EO/IR 탐지", "P_fused = 1 - (1-P_eo)(1-P_ir)", "core/sensor_observation.py", "fused_probability = 1.0 - ("),
    ("EO/IR 탐지", "lambda = -ln(1-PD)/t_scan", "core/sensor_observation.py", "return -log1p(-probability) / self.reference_scan_s"),
    ("EO/IR 탐지", "P(노출 t) = 1 - exp(-lambda t)", "core/sensor_observation.py", "def probability_for_exposure"),
    ("지형 가중", "통로 가중 exp(-d^2/2sigma^2)", "core/terrain.py", "def weight(self, x: float, y: float) -> float:"),
    ("지형 가중", "이동 편향 v <- (1-b)v + b grad(mobility)", "core/terrain.py", "def bias_step"),
    ("지형 가중", "관측성 가중치 (셀 hazard 배율)", "core/terrain.py", "def observability_weight"),
    ("지형 가중", "합성지형 생성 (회랑·장애물·은폐)", "core/terrain.py", "def build_synthetic_terrain"),
    ("지형 가중", "대조군 지형 (회랑 없음)", "core/terrain.py", "def build_control_terrain"),
    ("경로 KPI", "임무시간 클리핑 t = L_flown / v", "core/evaluation.py", "def _segments_within_window"),
    ("경로 KPI", "고유 면적률 / 중복률", "core/evaluation.py", "unique_area_ratio = min("),
    ("경로 KPI", "확률질량 탐색률", "core/evaluation.py", "def probability_mass_coverage"),
    ("탐지시간 평가", "상대운동 노출구간 (2차 방정식)", "core/simulation.py", "def _relative_motion_exposure"),
    ("탐지시간 평가", "누적 위험률 역변환 표집", "core/simulation.py", "def _detection_time_from_hazard_intervals"),
    ("탐지시간 평가", "Lambda(t) = threshold 선형 보간", "core/simulation.py", "detected = previous_time + ("),

    # --- Chapter 2: 경로제약 · Stone SP1/SPX ---
    ("Ch2 경로제약 기준식", "미탐지 재귀 u_{t+1} = (u_t e^{-a}) P_t", "theory/path_constrained.py", "def nondetection_trace"),
    ("Ch2 경로제약 기준식", "hazard 합 exp(-sum_j a_j)", "theory/path_constrained.py", "def joint_survival"),
    ("Ch2 경로제약 기준식", "유효 hazard a_eff = a(t,k) f[i,k]", "theory/path_constrained.py", "def effective_hazard"),
    ("Ch2 경로제약 기준식", "PD(경로묶음) = 1 - sum u_{T-1}^+", "theory/path_constrained.py", "def path_detection_probability"),
    ("Ch2 warm start", "team H1 후퇴지평 (SPX 초기해 전용)", "theory/path_constrained.py", "def team_receding_horizon"),
    ("Ch2 warm start", "team H2 후보 PD 비교", "theory/path_constrained.py", "def team_h2_receding_horizon"),
    ("Ch2 Markov 추정", "pi_0(x) = sum_{i in x} w_i", "theory/markov.py", "def estimate_space_time_markov"),
    ("Ch2 Markov 추정", "P_t(x,y) = N_t(x,y)/sum_y N_t(x,y)", "theory/markov.py", "row_sum = transitions[time_index].sum(axis=1)"),
    ("Ch2 Markov 추정", "도달가능 A(x,y) = [d <= reach]", "theory/markov.py", "def cell_adjacency"),
    ("Ch2 Stone SP1/SPX", "Markov PND f(Y) 와 gradient", "theory/stone_path.py", "def _nondetection_value_gradient"),
    ("Ch2 Stone SP1/SPX", "SP1 동질·단일표적 (4.23)-(4.28)", "theory/stone_path.py", "def stone_sp1_cutting_plane"),
    ("Ch2 Stone SP1/SPX", "SPX 이질·다중표적 minimax (4.48)-(4.56)", "theory/stone_path.py", "def stone_spx_cutting_plane"),
    ("Ch2 Stone SP1/SPX", "접평면 f(Yi)+grad^T(Y-Yi) <= eta (4.3.1)", "theory/stone_path.py", "cuts.append((target_index, value, gradient"),
    ("Ch2 Stone SP1/SPX", "셀 점유한도 (4.53)", "theory/stone_path.py", "# Cell occupancy deconfliction, equation (4.53)."),
    ("Ch2 Stone SP1/SPX", "양립불가 이동 (4.54)", "theory/stone_path.py", "# General incompatible-move deconfliction, equation (4.54)."),
    ("Ch2 Stone SP1/SPX", "분리반경 clique 절단", "theory/stone_path.py", "def _maximal_conflict_cliques"),
    ("Ch2 Stone SP1/SPX", "정확 생존 MILP (곱항 선형화)", "theory/stone_path.py", "def _solve_exact_survival_milp"),
    ("Ch2 지형결합 belief", "지형 3성분 주입 지점", "planning/terrain_belief.py", "terrain=terrain if terrain_coupling.prior else None,"),
    ("Ch2 인스턴스", "입자 -> 축약 다중표적 Markov 투영", "planning/stone_spx.py", "def _target_markov"),
    ("Ch2 인스턴스", "공통 입력 지문 (두 계획법 동일성 증거)", "planning/stone_spx.py", "def _common_input_fingerprint"),
    ("Ch2 인스턴스", "hazard = W v dt / A_cell x scale x observability", "planning/stone_spx.py", "full_slice_swept_area_m2 = ("),
    ("Ch2 경로변환", "n_track = round(sqrt(L/s)) 정사각 블록", "planning/routes.py", "track_count = max(1, int(round(sqrt(centerline_budget_m / spacing))))"),
    ("Ch2 경로변환", "시간예산 회계 (실제 소요시간으로 축소)", "planning/routes.py", "leg_length *= 0.9 * (time_budget_s / elapsed_s)"),

    # --- Chapter 3: MAPPO 로 Stone 해석 ---
    ("Ch3 SPX 환경", "행동 후보 = belief 점수 상위 K 인접셀", "learning/spx_env.py", "def _build_candidates"),
    ("Ch3 SPX 환경", "점유·분리 복구 (실행가능집합 유지)", "learning/spx_env.py", "def _resolve_destinations"),
    ("Ch3 SPX 환경", "온라인 미탐지 재귀 (= nondetection_trace)", "learning/spx_env.py", "survival = np.exp(-hazard * self._target_multiplier[index])"),
    ("Ch3 SPX 환경", "minimax 성형 w_i ∝ exp(-PD_i/tau)", "learning/spx_env.py", "def _worst_target_weights"),
    ("Ch3 SPX 환경", "종단 보상 = min_i PD_i", "learning/spx_env.py", "rewards += self.config.terminal_weight * float(self._detected.min())"),
    ("Ch3 MAPPO", "TD 오차 delta_t", "learning/mappo.py", "delta = rewards + gamma * nonterminal * next_values - episode_old_values"),
    ("Ch3 MAPPO", "GAE 역방향 누적 A_t = delta_t + gamma lambda A_{t+1}", "learning/mappo.py", "for time_index in range(episode_steps - 1, -1, -1):"),
    ("Ch3 MAPPO", "PPO clipped surrogate", "learning/mappo.py", "def update("),
    ("Ch3 비교", "학습효과 = 학습 - 미학습 (결정론적 평가)", "learning/spx_policy.py", "learning_effect=float(best_worst - untrained_worst)"),
    ("Ch3 비교", "SPX - MAPPO (같은 인스턴스)", "learning/spx_policy.py", "def compare_with_spx"),

    # --- Chapter 4: 합성지형 실비행 ---
    ("Ch4 실비행 평가", "계획모형 - 실비행 차이 (부호는 측정 결과)", "chapters/ch4_synthetic_terrain.py", "model_minus_flown="),
    ("Ch4 실비행 평가", "표적 계층 가중 합성", "chapters/ch4_synthetic_terrain.py", "def _combine_strata"),
    ("Ch4 통계", "짝지은 seed 군집 부트스트랩 백분위 구간", "reliability.py", "def bootstrap_weighted_mean_ci"),
    ("Ch4 통계", "짝지은 차이 + 부호 방향 명시", "reliability.py", "def paired_metric_report"),
    ("Ch4 통계", "Wilson 점수 구간", "reliability.py", "def wilson_score_interval"),
    ("Ch4 KPI", "공통 KPI 블록 (5 순위지표 + 진단)", "kpi.py", "def mission_kpi"),
    ("Ch4 KPI", "seed 군집 요약 + 신뢰성 판정", "kpi.py", "def summarise_kpi"),
]

def find_line(path: pathlib.Path, needle: str) -> int | None:
    text = path.read_text(encoding="utf-8")
    index = text.find(needle)
    if index < 0:
        return None
    return text.count("\n", 0, index) + 1


lines: list[str] = []
lines.append("# 수식 색인 — 어떤 식이 어디에 구현되어 있나\n")
lines.append(
    "각 행의 위치는 저장소 기준 `파일:줄`이다. 코드를 고치면 줄 번호는 바뀌므로,\n"
    "찾을 때는 줄 번호보다 **함수 이름**을 먼저 보는 편이 안전하다.\n"
    "이 표는 `python3 tools/gen_math_doc.py`로 다시 생성할 수 있다.\n"
)

current = None
missing: list[str] = []
for group, formula, filename, needle in ENTRIES:
    path = ROOT / filename
    line = find_line(path, needle) if path.is_file() else None
    if line is None:
        missing.append(f"{filename} :: {needle[:50]}")
        continue
    if group != current:
        lines.append(f"\n## {group}\n")
        lines.append("| 수식 / 개념 | 구현 위치 |")
        lines.append("|---|---|")
        current = group
    lines.append(f"| {formula} | `src/cpp_search/{filename}:{line}` |")

output = pathlib.Path("docs/MATH.md")
output.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {output} with {len(ENTRIES) - len(missing)} / {len(ENTRIES)} entries")
for item in missing:
    print("  MISS", item)


# --- 의존 그래프 (자동 추출) -------------------------------------------------

def module_name(path: pathlib.Path) -> str:
    relative = path.relative_to(ROOT.parent).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def internal_imports(path: pathlib.Path, name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    package = name if path.name == "__init__.py" else name.rsplit(".", 1)[0]
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 0:
                target = node.module or ""
            else:
                base = package.split(".")
                climb = node.level - 1
                if climb:
                    base = base[:-climb]
                target = ".".join(base + ([node.module] if node.module else []))
            if target.startswith("cpp_search"):
                found.add(target)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("cpp_search"):
                    found.add(alias.name)
    return found


graph: dict[str, set[str]] = {}
for path in sorted(ROOT.rglob("*.py")):
    name = module_name(path)
    graph[name] = internal_imports(path, name)

reverse: dict[str, set[str]] = {name: set() for name in graph}
for name, targets in graph.items():
    for target in targets:
        reverse.setdefault(target, set()).add(name)


def short(name: str) -> str:
    return name.replace("cpp_search.", "")


section = ["\n\n# 의존 그래프 (import 기준, 자동 추출)\n"]
section.append(
    "`쓰는 것`은 그 모듈이 import하는 저장소 내부 모듈, `쓰이는 곳`은 그 모듈을\n"
    "import하는 모듈이다. 패키지 `__init__.py`의 재수출은 제외하지 않았으므로,\n"
    "`theory`/`planning`/`learning` 같은 패키지 이름이 보이면 그 패키지의\n"
    "`__init__.py`를 통해 들어온 것이다.\n"
)
section.append("\n| 모듈 | 쓰는 것 | 쓰이는 곳 |")
section.append("|---|---|---|")
for name in sorted(graph):
    if name.endswith("__init__") or name == "cpp_search":
        pass
    uses = ", ".join(f"`{short(t)}`" for t in sorted(graph[name])) or "—"
    used_by = ", ".join(f"`{short(t)}`" for t in sorted(reverse.get(name, ()))) or "—"
    section.append(f"| `{short(name)}` | {uses} | {used_by} |")

with output.open("a", encoding="utf-8") as handle:
    handle.write("\n".join(section) + "\n")
print("appended dependency graph")
