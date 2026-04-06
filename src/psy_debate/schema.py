from __future__ import annotations

from typing import Any, Literal

from typing_extensions import NotRequired, TypedDict


# ---------------------------------------------------------------------------
# Stage names
# ---------------------------------------------------------------------------

STAGES = Literal[
    "awaiting_student_id",
    "entry_detection",
    "active_listen",
    "structured_probe",
    "rapport_build",
    "somatic_bridge",
    "anchoring",
    "hypothesis_probe",
    "conclusion",
    "forced_closure",
    "crisis",
]

# ---------------------------------------------------------------------------
# Verbal style
# ---------------------------------------------------------------------------

VERBAL_STYLES = Literal[
    "unknown",
    "expressive",
    "silent",
    "resistant",
    "somatic",
    "hyperverbal",
]

# ---------------------------------------------------------------------------
# Symptom record (stored inside portrait["symptoms"])
# ---------------------------------------------------------------------------
# status: unmentioned → suspected → probable → confirmed | disputed

# ---------------------------------------------------------------------------
# Main state
# ---------------------------------------------------------------------------

class DebateState(TypedDict):
    # ── Session identity ──────────────────────────────────────────────────
    student_id: str           # 学号，空字符串表示尚未收集
    session_id: str           # UUID，由 session_init 创建
    is_returning_patient: bool
    turn_count: int

    # ── Input / output ────────────────────────────────────────────────────
    user_input: str
    assistant_output: str
    history: list[dict[str, str]]   # 最近 N 轮，仅用于上下文

    # ── Patient profile ───────────────────────────────────────────────────
    verbal_style: str          # expressive / silent / resistant / somatic / hyperverbal / unknown
    alliance_score: float      # 0-1，滚动均值

    # ── State machine ─────────────────────────────────────────────────────
    stage: str                 # 见 STAGES
    fallback_counter: int      # 连续无进展轮数，触发 FORCED_CLOSURE

    # ── Clinical portrait（独立于 history window）─────────────────────────
    portrait: dict[str, Any]
    # portrait 结构:
    # {
    #   "symptoms": {
    #     "depressed_mood": {
    #       "status": "confirmed",          # suspected/probable/confirmed/disputed
    #       "confidence": 0.82,
    #       "first_turn": 2,
    #       "last_consistent_turn": 7,
    #       "contradicted": false
    #     }, ...
    #   },
    #   "timeline": {
    #     "onset": "2mo_ago",
    #     "duration_confirmed": false,
    #     "trigger": "exam_stress",
    #     "anchor_established": true
    #   },
    #   "functional_impact": {
    #     "academic": "impaired",
    #     "social": "withdrawn",
    #     "daily_living": "unknown"
    #   },
    #   "hypotheses": [
    #     {
    #       "disorder": "MDD",
    #       "disorder_cn": "重性抑郁障碍",
    #       "confidence": 0.74,
    #       "missing_criteria": ["duration_2weeks"],
    #       "critical_criteria_covered": ["depressed_mood", "functional_impairment"]
    #     }
    #   ],
    #   "consistency_score": 0.78,
    #   "treatment_history": {
    #     "medications": [
    #       {"name": "氯氮平", "outcome": "无效", "side_effects": "神经痉挛", "source": "北大六院"}
    #     ],
    #     "procedures": [
    #       {"type": "ECT", "sessions": 3, "outcome": "首次有效，后两次无效"}
    #     ],
    #     "hospitalization": {"total_years": 3, "pattern": "阶段性"}
    #   }
    # }

    # ── Risk ──────────────────────────────────────────────────────────────
    risk_level: str            # low / medium / high / critical
    risk_factors: list[str]
    phase: str                 # normal / crisis

    # ── Safety nodes ──────────────────────────────────────────────────────
    input_safety: dict[str, Any]
    output_safety: dict[str, Any]

    # ── Handoff / report ──────────────────────────────────────────────────
    handoff_required: bool
    handoff_reason: str
    session_report: NotRequired[dict[str, Any]]

    # ── Debug ─────────────────────────────────────────────────────────────
    debug_notes: NotRequired[str]
