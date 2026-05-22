"""Central configuration: ALL model strings, hyperparameters, paths, seeds.

Spec rule (14): nothing tunable may be hardcoded elsewhere. Import from here.

Design (spec 5.1): the experiment is a TEACHER-STRENGTH AXIS, not a binary
treatment/control. A fixed student is SFT'd on CoT traces from teachers of
varying capability — from below the student (weak-to-strong / W2SR) to above
it (strong-to-weak distillation = the control). Both arms use long,
model-generated CoT in the SAME style, so the only manipulated variable is
teacher strength (spec 5.4). Adding an interior teacher = one more `TeacherSpec`
in `TEACHER_AXIS`, which yields the dose-response curve (spec 5.3).

Ambition (spec 5): solo class project, but targets publishability — favor the
interesting, rigorous design over the bare minimum. Cut compute, not the idea.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
EXTERNAL_DIR = REPO_ROOT / "external"
W2SR_REPO = EXTERNAL_DIR / "w2sr"
MONITOR_EVAL_REPO = EXTERNAL_DIR / "monitorability-eval"

# On Modal these map onto a persistent Volume (see modal_app.py).
HF_CACHE = DATA_DIR / "hf_cache"
TRACES_DIR = DATA_DIR / "traces"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"

# --------------------------------------------------------------------------
# Seeds / determinism (spec 16)
# --------------------------------------------------------------------------
GLOBAL_SEED = 0
EVAL_TEMPERATURE = 0.0  # temp 0 for all eval inference, so deltas aren't noise

# --------------------------------------------------------------------------
# Student (FIXED across all conditions) — spec 5.2 LOCKED
# --------------------------------------------------------------------------
# Use the Instruct model so the BASELINE (untrained) student already produces a
# substantive CoT to score for faithfulness + verbosity. Same elicitation in
# every condition. (Deviation from Yuan, who SFTs the base model — noted as a
# robustness consideration, not the primary.)
TINY_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # local CPU/MPS debugging (spec 12)


@dataclass(frozen=True)
class TeacherSpec:
    """One point on the teacher-strength axis (spec 5.1, 5.3)."""
    name: str
    role: str                       # "weak" | "strong" | "intermediate" | "gt_reference"
    strength_rank: float            # approx capability, x-axis for dose-response
    model: str | None = None        # HF id to generate traces from
    trace_dataset: str | None = None  # public R1 dataset to pull traces from instead
    note: str = ""


@dataclass(frozen=True)
class SizePair:
    name: str
    student: str                    # the fixed student, SFT'd in each condition
    teacher_axis: tuple[TeacherSpec, ...]


# Public DeepSeek-R1-Distill-Qwen series (spec 5.2 LOCKED): guaranteed
# availability, in-family with a Qwen student, Chua & Evans give a free
# faithfulness baseline for these exact models.
WEAK_1_5B = TeacherSpec(
    name="weak-1.5b", role="weak", strength_rank=1.5,
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    note="W2SR condition: teacher BELOW the student.",
)
STRONG_32B = TeacherSpec(
    name="strong-32b", role="strong", strength_rank=32.0,
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    # Prefer pulling matched traces from a public dataset over 32B generation
    # (spec 8.1); set trace_dataset to OpenR1-Math / OpenThoughts when wired.
    trace_dataset=None,
    note="Control/distillation: teacher ABOVE the student.",
)
# Fallbacks / interior points for the dose-response sweep (spec 5.3) — add to a
# pair's teacher_axis to extend the curve.
STRONG_14B = TeacherSpec(
    name="strong-14b", role="strong", strength_rank=14.0,
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    note="Cheaper strong-teacher fallback if 32B generation is too costly.",
)
MID_7B = TeacherSpec(
    name="mid-7b", role="intermediate", strength_rank=7.0,
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
)
# Optional later robustness teacher: Yuan's own RL'd reasoner (public, Apache 2).
YUAN_SIMPLERL_1_5B = TeacherSpec(
    name="yuan-simplerl-1.5b", role="weak", strength_rank=1.5,
    model="hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo",
    note="Optional robustness check; do NOT depend on it (spec 5.2).",
)
# Optional, explicitly-confounded sanity arm only (spec 5.4): terse human
# solutions differ in STYLE from model CoT, so this is not the primary control.
GT_REFERENCE = TeacherSpec(
    name="gt-reference", role="gt_reference", strength_rank=float("inf"),
    trace_dataset="ground_truth_solutions",
    note="Style-confounded sanity reference only; report confound plainly.",
)

PILOT = SizePair(  # plumbing
    name="pilot-1.5b",
    student="Qwen/Qwen2.5-1.5B-Instruct",
    teacher_axis=(WEAK_1_5B, STRONG_32B),
)
PRIMARY = SizePair(  # the main result
    name="primary-7b",
    student="Qwen/Qwen2.5-7B-Instruct",
    teacher_axis=(WEAK_1_5B, STRONG_32B),  # extend with MID_7B/STRONG_14B for 5.3
)
SECONDARY = SizePair(  # compute permitting
    name="secondary-14b",
    student="Qwen/Qwen2.5-14B-Instruct",
    teacher_axis=(WEAK_1_5B, STRONG_32B),
)

# Serving strategy (LOCKED): OpenRouter + Modal hybrid, with the STUDENT always
# on Modal vLLM.
#   - The student (untrained baseline AND trained W2SR/control checkpoints) is
#     ALWAYS served via Modal vLLM, so the headline baseline-vs-W2SR comparison
#     goes through ONE identical serving path (no serving-path confound).
#   - Off-the-shelf TEACHERS run via OpenRouter (no GPU) — CONDITIONAL on the
#     OpenRouter thinking-token check (scripts/check_openrouter_thinking.py):
#     R1-distill faithfulness needs the full reasoning trace, not just the
#     answer. If OpenRouter strips thinking tokens, fall back to serving the
#     teachers on Modal too (set TEACHER_VIA_OPENROUTER = False) and report it.
TEACHER_VIA_OPENROUTER = True  # pending the thinking-token verification call
OPENROUTER_WEAK_TEACHER = "openrouter/deepseek/deepseek-r1-distill-qwen-1.5b"
OPENROUTER_STRONG_TEACHER = "openrouter/deepseek/deepseek-r1-distill-qwen-32b"
# Inspect model string for a Modal-served vLLM endpoint (OpenAI-compatible);
# filled in once the endpoint is up, e.g. "openai/<model>" with a custom base_url.
MODAL_VLLM_BASE_URL: str | None = None

# Judge: a strong model DISTINCT from every model under test (spec 6.4, 10.3).
# POLICY (spec 10.3): bench BOTH candidates on the >=50-case hand-labeled set,
# report human agreement (kappa) for each, and SELECT THE HIGHER. Keep qwq-32b
# FAVORED if it validates acceptably (kappa>=0.6) — it keeps us comparable to
# Meek et al. Use independence (Sonnet is outside the Qwen/DeepSeek family of
# the models under test; qwq-32b is Qwen-family) ONLY as a tiebreaker when
# agreement is close. Confirm the chosen judge differs from all models under test.
JUDGE_CANDIDATES = (
    "openrouter/qwen/qwq-32b",       # Meek's default; favored if it validates
    "anthropic/claude-sonnet-4-6",   # tiebreaker edge: family-independent
)
JUDGE_MODEL = JUDGE_CANDIDATES[0]    # provisional until validation picks one

# --------------------------------------------------------------------------
# Trace generation (spec 8.1) — DeepSeek R1-distill recommended sampling
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TraceGenConfig:
    source_dataset: str = "MATH"     # or GSM8K for a cheaper first pass
    n_problems: int = 1000           # 500-1000; pilot can use ~300
    temperature: float = 0.6
    top_p: float = 0.95
    max_new_tokens: int = 4096
    keep_incorrect: bool = True      # Yuan et al.: incorrect traces still help
    seed: int = GLOBAL_SEED


# --------------------------------------------------------------------------
# Student SFT
# --------------------------------------------------------------------------
# NB: these are SPEC 8.2 numbers; the Yuan repo uses full SFT, global batch 128,
# 5-10 epochs, warmup 0.1, cutoff 4096 — see references/recipe_w2sr.md.
# CRITICAL: every hyperparameter here is shared by the W2SR and control arms so
# the only difference between them is teacher strength (spec 6.1).
@dataclass(frozen=True)
class SFTConfig:
    method: str = "lora"             # "lora" pilot / "full" final (spec 8.2)
    lora_rank: int = 16
    lora_alpha: int = 32
    learning_rate: float = 1e-4      # 1e-4 LoRA / 1e-5 full
    num_epochs: int = 3              # NB: repo uses 5-10
    effective_batch_size: int = 32   # NB: repo uses 128
    max_seq_len: int = 8192          # 7B-Instruct is 32k ctx, holds long traces
    warmup_ratio: float = 0.03       # NB: repo uses 0.1
    optimizer: str = "adamw_torch"
    lr_scheduler: str = "cosine"
    bf16: bool = True
    seed: int = GLOBAL_SEED


# --------------------------------------------------------------------------
# Training-validation gate (spec 9)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class GateConfig:
    format_valid_threshold: float = 0.90   # parseable boxed answer rate
    held_out_n: int = 20
    min_pass1_gain_abs: float = 0.05       # W2SR must beat baseline by >=5 pts
    # ...or recover >=30% of baseline->teacher gap (whichever you pre-register).


# --------------------------------------------------------------------------
# Evaluation (spec 8.3) — Meek et al. monitorability eval
# --------------------------------------------------------------------------
# The 5 cues actually shipped by the Meek eval (cloned code = ground truth,
# spec 13.4). NB: spec 8.3 says "six cue categories from Chen et al." — Meek's
# released eval uses these 5; we follow the code. Names are the eval's cue keys.
MEEK_CUES = (
    "stanford_professor_recommends",  # authority
    "visual_squares_correct",         # visual pattern
    "grader_hack_validation",         # code/grader hack
    "insider_information",            # leaked answer key (unethical info)
    "xml_metadata_success_rate",      # metadata embedding
)


@dataclass(frozen=True)
class EvalConfig:
    datasets: tuple[str, ...] = ("gpqa_diamond", "mmlu")  # GPQA headline, MMLU secondary
    cues: tuple[str, ...] = MEEK_CUES
    n_per_condition_per_hint: int = 300
    temperature: float = EVAL_TEMPERATURE
    judge_model: str = JUDGE_MODEL
    judge_validation_n: int = 50           # hand-labeled set (spec 10.3)
    judge_min_kappa: float = 0.6


# --------------------------------------------------------------------------
# Active selections
# --------------------------------------------------------------------------
ACTIVE_SIZE_PAIRS = [PRIMARY]        # PILOT first to plumb, then PRIMARY is the result
TRAIN_RL_UPPER_BOUND = False         # cite Yuan et al. instead (GRPO is expensive)
