from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from typing import Any

from .db import (
    close_session_record,
    create_session_record,
    load_patient,
    save_portrait,
    upsert_patient,
)
from .models import DeepSeekHub
from .prompts import RISK_GUARD_SYSTEM, build_clinical_brain_prompt
from .schema import DebateState

HISTORY_WINDOW = max(1, int(os.getenv("HISTORY_WINDOW", "8")))

# 中文描述 → 标准英文 key 映射（兜底规范化）
_CN_TO_KEY: dict[str, str] = {
    "幻听": "hallucinations", "幻觉": "hallucinations",
    "妄想": "delusions", "被害妄想": "delusions", "关系妄想": "delusions",
    "思维紊乱": "disorganized_speech", "言语紊乱": "disorganized_speech",
    "行为紊乱": "disorganized_behavior",
    "阴性症状": "negative_symptoms", "情感淡漠": "emotional_blunting",
    "情绪低落": "depressed_mood", "情绪低沉": "depressed_mood",
    "兴趣减退": "anhedonia", "快感缺失": "anhedonia",
    "睡眠障碍": "sleep_disturbance", "失眠": "sleep_disturbance", "睡不着": "sleep_disturbance",
    "疲劳": "fatigue", "精力不足": "fatigue",
    "无价值感": "worthlessness_guilt", "内疚": "worthlessness_guilt",
    "注意力障碍": "concentration_difficulty", "思维混乱": "thought_disorganization",
    "自杀意念": "suicidal_ideation", "死亡想法": "suicidal_ideation",
    "体重变化": "weight_change", "食欲变化": "weight_change",
    "坐立不安": "restlessness", "易激惹": "irritability",
    "肌肉紧张": "muscle_tension", "心悸": "palpitations",
    "现实解体": "derealization", "人格解体": "derealization",
    "躯体不适": "somatic_complaints", "躯体症状": "somatic_complaints",
    "社交回避": "social_withdrawal", "回避行为": "social_withdrawal",
    "功能下降": "functional_decline", "学业受损": "functional_decline",
    "意志减退": "avolition", "言语贫乏": "alogia",
}

_VALID_KEYS = set(_CN_TO_KEY.values())


def _normalize_portrait(portrait: dict) -> dict:
    """
    规范化 LLM 返回的 portrait：
    1. 将 symptoms 中 'consistency' 字段重命名为 'status'
    2. 将中文 key 映射为标准英文 key（未知 key 保留但警告）
    """
    symptoms = portrait.get("symptoms", {})
    normalized: dict = {}
    for key, val in symptoms.items():
        if not isinstance(val, dict):
            continue
        # 字段名修正：consistency → status
        if "status" not in val and "consistency" in val:
            val = {**val, "status": val.pop("consistency")}
        elif "status" not in val:
            # 根据 confidence 推断 status
            conf = val.get("confidence", 0.3)
            if conf >= 0.8:
                val = {**val, "status": "confirmed"}
            elif conf >= 0.5:
                val = {**val, "status": "probable"}
            else:
                val = {**val, "status": "suspected"}
        # key 规范化
        std_key = _CN_TO_KEY.get(key, key)
        normalized[std_key] = val
    portrait = {**portrait, "symptoms": normalized}
    return portrait

# ---------------------------------------------------------------------------
# Stage transition thresholds
# ---------------------------------------------------------------------------

# 症状达到 probable 或以上才算有效
def _count_symptoms(portrait: dict[str, Any], min_status: str = "probable") -> int:
    order = {"suspected": 1, "probable": 2, "confirmed": 3, "disputed": 0}
    threshold = order.get(min_status, 2)
    symptoms = portrait.get("symptoms", {})
    return sum(
        1 for s in symptoms.values()
        if order.get(s.get("status", "suspected"), 0) >= threshold
    )


def _anchor_established(portrait: dict[str, Any]) -> bool:
    return portrait.get("timeline", {}).get("anchor_established", False)


def _leading_hypothesis(portrait: dict[str, Any]) -> dict[str, Any] | None:
    hypotheses = portrait.get("hypotheses", [])
    if not hypotheses:
        return None
    return max(hypotheses, key=lambda h: h.get("confidence", 0))


def _should_force_closure(state: DebateState) -> bool:
    """C方案：长时间无进展强制结束。"""
    turn = state.get("turn_count", 0)
    fallback = state.get("fallback_counter", 0)
    stage = state.get("stage", "")
    alliance = state.get("alliance_score", 0.5)

    if stage == "rapport_build" and turn > 12 and fallback >= 5:
        return True
    if stage == "hypothesis_probe" and turn > 20 and fallback >= 4:
        return True
    return False


def _compute_next_stage(state: DebateState, brain_result: dict[str, Any]) -> str:
    """
    State machine transition logic.
    brain_result may suggest a stage via 'stage_transition'.
    Final decision is here, not in the LLM.
    """
    current = state.get("stage", "entry_detection")
    portrait = brain_result.get("updated_portrait") or state.get("portrait", {})
    alliance = brain_result.get("alliance_score", state.get("alliance_score", 0.5))
    suggested = brain_result.get("stage_transition", current)

    probable_count = _count_symptoms(portrait, "probable")
    confirmed_count = _count_symptoms(portrait, "confirmed")
    anchor = _anchor_established(portrait)
    leading = _leading_hypothesis(portrait)
    leading_conf = leading.get("confidence", 0) if leading else 0
    missing = leading.get("missing_criteria", []) if leading else []
    consistency = portrait.get("consistency_score", 1.0)
    turn = state.get("turn_count", 0)

    # ── Crisis takes over immediately ─────────────────────────────────────
    if state.get("phase") == "crisis":
        return "crisis"

    # ── Forced closure ────────────────────────────────────────────────────
    if _should_force_closure(state):
        return "forced_closure"

    # ── Awaiting student ID ───────────────────────────────────────────────
    if current == "awaiting_student_id":
        return "awaiting_student_id"

    # ── Entry detection (first 2 turns) ───────────────────────────────────
    if current == "entry_detection" and turn < 3:
        return "entry_detection"

    # ── From entry → patient-type state ──────────────────────────────────
    if current == "entry_detection":
        style = state.get("verbal_style", "unknown")
        mapping = {
            "expressive": "active_listen",
            "silent": "structured_probe",
            "resistant": "rapport_build",
            "somatic": "somatic_bridge",
            "hyperverbal": "anchoring",
        }
        return mapping.get(style, "structured_probe")

    # ── Alliance collapse → rapport_build ────────────────────────────────
    if alliance < 0.4 and current not in ("rapport_build", "crisis", "conclusion", "forced_closure"):
        return "rapport_build"

    # ── rapport_build exit ────────────────────────────────────────────────
    if current == "rapport_build":
        if alliance >= 0.6:
            style = state.get("verbal_style", "silent")
            return "active_listen" if style == "expressive" else "structured_probe"
        return "rapport_build"

    # ── somatic_bridge exit ───────────────────────────────────────────────
    if current == "somatic_bridge":
        bridge_done = portrait.get("somatic_bridge_complete", False)
        if bridge_done and probable_count >= 2:
            return "hypothesis_probe"
        return "somatic_bridge"

    # ── anchoring exit ────────────────────────────────────────────────────
    if current == "anchoring":
        focused_turns = portrait.get("consecutive_focused_turns", 0)
        if focused_turns >= 2 and probable_count >= 2:
            return "hypothesis_probe"
        return "anchoring"

    # ── active_listen / structured_probe → hypothesis_probe ───────────────
    if current in ("active_listen", "structured_probe"):
        # 放宽：probable≥3 + confirmed≥1 + 时间线锚定，即可进入假设探查
        if probable_count >= 3 and confirmed_count >= 1 and anchor:
            return "hypothesis_probe"
        return current

    # ── hypothesis_probe → conclusion ────────────────────────────────────
    if current == "hypothesis_probe":
        all_covered = len(missing) == 0
        if leading_conf >= 0.8 and all_covered and consistency >= 0.7:
            return "conclusion"
        return "hypothesis_probe"

    return current


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

class PsyNodes:
    def __init__(self, hub: DeepSeekHub) -> None:
        self.hub = hub

    # ── session_init ───────────────────────────────────────────────────────
    async def session_init(self, state: DebateState) -> dict[str, Any]:
        """
        First node. Handles student ID collection and DB loading.
        If student_id not yet collected, asks for it and halts pipeline.
        """
        student_id = state.get("student_id", "").strip()
        user_input = state.get("user_input", "").strip()

        # ── Step 1: collect student ID ────────────────────────────────────
        if not student_id:
            # Check if this turn's input looks like a student ID
            if re.match(r"^\d{6,15}$", user_input):
                student_id = user_input
            else:
                # Ask for student ID
                return {
                    "stage": "awaiting_student_id",
                    "assistant_output": "你好，欢迎使用心理健康初诊系统。在开始之前，请告诉我你的学号。",
                    "phase": "normal",
                }

        # ── Step 2: load from DB ──────────────────────────────────────────
        db_data = load_patient(student_id)
        is_new = db_data["is_new"]

        if is_new:
            upsert_patient(student_id)
            session_id = create_session_record(student_id)
            portrait: dict[str, Any] = {
                "symptoms": {},
                "timeline": {"anchor_established": False},
                "functional_impact": {},
                "hypotheses": [],
                "consistency_score": 1.0,
                "somatic_bridge_complete": False,
                "consecutive_focused_turns": 0,
            }
            verbal_style = "unknown"
            welcome = "好的，我已经记录你的学号。你可以从任何让你感到困扰的事情开始说起，不用担心，这里是安全的。"
        else:
            session_id = create_session_record(student_id)
            portrait = db_data["portrait"] or {
                "symptoms": {},
                "timeline": {"anchor_established": False},
                "functional_impact": {},
                "hypotheses": [],
                "consistency_score": 1.0,
                "somatic_bridge_complete": False,
                "consecutive_focused_turns": 0,
            }
            verbal_style = db_data["verbal_style"] or "unknown"

            # Summarize what we already know for the welcome message
            confirmed_list = [
                k for k, v in portrait.get("symptoms", {}).items()
                if v.get("status") in ("probable", "confirmed")
            ]
            if confirmed_list:
                welcome = (
                    f"欢迎回来，我记录了你上次分享的一些情况。"
                    f"你今天感觉怎么样，和上次相比有什么变化吗？"
                )
            else:
                welcome = "欢迎回来，今天有什么想聊的吗？"

        history = list(state.get("history", []))
        if user_input and not re.match(r"^\d{6,15}$", user_input):
            history.append({"role": "user", "content": user_input})

        return {
            "student_id": student_id,
            "session_id": session_id,
            "is_returning_patient": not is_new,
            "verbal_style": verbal_style,
            "portrait": portrait,
            "stage": "entry_detection",
            "turn_count": state.get("turn_count", 0),
            "fallback_counter": state.get("fallback_counter", 0),
            "alliance_score": portrait.get("alliance_score", 0.5),
            "phase": "normal",
            "history": history,
            "assistant_output": welcome if not state.get("student_id") else state.get("assistant_output", ""),
            "handoff_required": False,
            "handoff_reason": "",
            "risk_level": "low",
            "risk_factors": [],
        }

    # ── input_safety ───────────────────────────────────────────────────────
    async def input_safety(self, state: DebateState) -> dict[str, Any]:
        text = state.get("user_input", "").strip()
        # Skip if awaiting student ID or empty
        if state.get("stage") == "awaiting_student_id" or not text:
            return {"input_safety": {"blocked": True, "reason": "awaiting_id_or_empty"}}

        history = list(state.get("history", []))
        # Avoid double-appending if session_init already added it
        last = history[-1] if history else {}
        if not (last.get("role") == "user" and last.get("content") == text):
            history.append({"role": "user", "content": text})

        return {
            "input_safety": {"blocked": False},
            "history": history,
        }

    # ── analyze (risk_guard + clinical_brain in parallel) ─────────────────
    async def analyze(self, state: DebateState) -> dict[str, Any]:
        if state.get("input_safety", {}).get("blocked", False):
            return {}

        user_input = state.get("user_input", "")
        portrait = state.get("portrait", {})
        turn = state.get("turn_count", 0) + 1

        # Determine leading disorder for criteria injection
        leading = _leading_hypothesis(portrait)
        leading_disorder = leading.get("disorder") if leading else None

        brain_system = build_clinical_brain_prompt(leading_disorder)
        brain_payload = self._build_brain_payload(state, turn)

        # Run in parallel
        risk_result, brain_result = await asyncio.gather(
            self._safe_risk(user_input),
            self._safe_brain(brain_system, brain_payload),
        )

        # Merge risk into portrait/phase
        is_crisis = risk_result.get("is_crisis", False)
        final_risk = risk_result.get("risk_level", "low")
        phase = "crisis" if is_crisis else "normal"

        # Extract brain outputs
        raw_portrait = brain_result.get("updated_portrait") or portrait
        updated_portrait = _normalize_portrait(raw_portrait)
        updated_portrait["alliance_score"] = brain_result.get("alliance_score", state.get("alliance_score", 0.5))
        hypotheses = brain_result.get("hypotheses") or portrait.get("hypotheses", [])
        updated_portrait["hypotheses"] = hypotheses

        alliance_score = brain_result.get("alliance_score", state.get("alliance_score", 0.5))
        fallback_delta = int(brain_result.get("fallback_delta", 0))
        final_question = brain_result.get("final_question", "")

        # Verbal style: detect on turn 1-2; allow re-evaluation if silent but user
        # now provides long expressive content (style may evolve after initial reticence)
        verbal_style = state.get("verbal_style", "unknown")
        if verbal_style in ("unknown", "silent") and turn <= 4:
            detected = self._detect_verbal_style(user_input, turn)
            if detected != "silent" or verbal_style == "unknown":
                verbal_style = detected

        # Stage transition
        new_state_for_transition = {
            **state,
            "portrait": updated_portrait,
            "alliance_score": alliance_score,
            "phase": phase,
            "verbal_style": verbal_style,
            "turn_count": turn,
            "fallback_counter": state.get("fallback_counter", 0) + fallback_delta,
        }
        next_stage = _compute_next_stage(new_state_for_transition, brain_result)

        # Persist portrait to DB after every turn
        student_id = state.get("student_id", "")
        if student_id:
            save_portrait(student_id, updated_portrait)
            upsert_patient(student_id, verbal_style if verbal_style != "unknown" else None)

        return {
            "portrait": updated_portrait,
            "hypotheses": hypotheses,
            "alliance_score": alliance_score,
            "verbal_style": verbal_style,
            "stage": next_stage,
            "turn_count": turn,
            "fallback_counter": state.get("fallback_counter", 0) + fallback_delta,
            "risk_level": final_risk,
            "risk_factors": risk_result.get("risk_factors", []),
            "phase": phase,
            "assistant_output": final_question,
            "debug_notes": brain_result.get("debug", ""),
        }

    # ── crisis ─────────────────────────────────────────────────────────────
    async def crisis(self, state: DebateState) -> dict[str, Any]:
        risk_factors = state.get("risk_factors", [])
        factor_text = "、".join(risk_factors[:2]) if risk_factors else "高风险信号"
        msg = (
            "谢谢你告诉我这些，你现在的状态让我很担心。"
            "请立刻联系身边可信任的人陪同你，或直接前往最近的精神科急诊。"
            "如果情况紧急，请拨打 120 或心理援助热线 400-161-9995。"
            "你现在身边有人吗？"
        )
        return {
            "assistant_output": msg,
            "handoff_required": True,
            "handoff_reason": f"crisis_{factor_text}",
            "stage": "crisis",
        }

    # ── output_safety ──────────────────────────────────────────────────────
    async def output_safety(self, state: DebateState) -> dict[str, Any]:
        msg = state.get("assistant_output", "").strip()
        if not msg:
            msg = "我在这里，你可以慢慢说。"

        # Do not expose disease names to patient
        for criteria in ["抑郁症", "焦虑症", "精神分裂", "双相", "惊恐障碍", "适应障碍"]:
            msg = msg.replace(criteria, "你描述的情况")

        history = list(state.get("history", []))
        history.append({"role": "assistant", "content": msg})

        # Keep only recent window
        if len(history) > HISTORY_WINDOW * 2:
            history = history[-(HISTORY_WINDOW * 2):]

        return {
            "assistant_output": msg,
            "history": history,
            "output_safety": {"blocked": False},
        }

    # ── generate_report ────────────────────────────────────────────────────
    async def generate_report(self, state: DebateState) -> dict[str, Any]:
        portrait = state.get("portrait", {})
        hypotheses = portrait.get("hypotheses", [])
        stage = state.get("stage", "")
        risk_level = state.get("risk_level", "low")

        payload = json.dumps(
            {
                "stage_reached": stage,
                "turn_count": state.get("turn_count", 0),
                "risk_level": risk_level,
                "risk_factors": state.get("risk_factors", []),
                "symptoms": portrait.get("symptoms", {}),
                "timeline": portrait.get("timeline", {}),
                "functional_impact": portrait.get("functional_impact", {}),
                "treatment_history": portrait.get("treatment_history", {}),
                "hypotheses": hypotheses,
                "consistency_score": portrait.get("consistency_score", 1.0),
                "handoff_reason": state.get("handoff_reason", ""),
                "recent_history": state.get("history", [])[-6:],
            },
            ensure_ascii=False,
        )

        report_system = (
            "你是精神科临床报告生成模块。根据输入生成供接诊医生参考的结构化摘要。"
            "可以输出疾病名称和诊断假设（这是给医生看的）。"
            "输出严格 JSON，字段："
            "chief_complaint、"
            "confirmed_symptoms、"
            "probable_symptoms、"
            "suspected_symptoms、"
            "timeline_summary、"
            "functional_impact、"
            "treatment_history_summary（含用药史和既往治疗，若无则为null）、"
            "hypotheses（含 disorder_cn 和 confidence）、"
            "risk_summary、"
            "recommended_next_steps（列表）、"
            "information_completeness（0-1）。"
        )

        report = await self.hub.generate_report(report_system, payload)

        # Save to DB
        session_id = state.get("session_id", "")
        student_id = state.get("student_id", "")
        if session_id:
            close_session_record(
                session_id=session_id,
                stage_reached=stage,
                turn_count=state.get("turn_count", 0),
                risk_level=risk_level,
                report=report,
            )

        return {"session_report": report}

    # ── Routing ────────────────────────────────────────────────────────────
    def route_after_session_init(self, state: DebateState) -> str:
        if state.get("stage") == "awaiting_student_id":
            return "output_only"
        return "continue"

    def route_after_analyze(self, state: DebateState) -> str:
        if state.get("phase") == "crisis":
            return "crisis"
        if state.get("stage") in ("conclusion", "forced_closure"):
            return "generate_report"
        return "output_safety"

    def route_after_output(self, state: DebateState) -> str:
        if state.get("handoff_required") or state.get("stage") in ("conclusion", "forced_closure", "crisis"):
            return "generate_report"
        return "end"

    # ── Internal helpers ───────────────────────────────────────────────────
    async def _safe_risk(self, user_input: str) -> dict[str, Any]:
        try:
            return await self.hub.risk_guard(RISK_GUARD_SYSTEM, user_input)
        except Exception:
            return {"risk_level": "low", "is_crisis": False, "risk_factors": [], "rationale": "error"}

    async def _safe_brain(self, system: str, payload: str) -> dict[str, Any]:
        try:
            return await self.hub.clinical_brain(system, payload)
        except Exception:
            return {"final_question": "能跟我多说一些吗？", "fallback_delta": 1}

    def _detect_verbal_style(self, user_input: str, turn: int) -> str:
        text = user_input.strip()
        length = len(text)

        # 1. Hyperverbal: 跳题
        if self._is_disorganized(text) and length > 30:
            return "hyperverbal"

        # 2. Resistant: 明确否认或被动就诊（比 somatic 优先）
        _resistant_kw = [
            "没什么事", "没什么问题", "没什么大事", "别人让我来", "被逼来的",
            "不需要", "没必要", "只是来看看", "随便看看", "不想来",
        ]
        if any(w in text for w in _resistant_kw):
            return "resistant"

        # 3. Somatic: 只有躯体主诉，无情绪词
        _somatic_kw = [
            "头疼", "头痛", "胸闷", "心跳", "肚子疼", "睡不着", "吃不下",
            "恶心", "胸痛", "手抖", "出汗", "心慌", "身体不舒服",
        ]
        _emotional_kw = [
            "难过", "焦虑", "害怕", "担心", "压力", "抑郁", "崩溃",
            "低落", "沮丧", "绝望", "无助", "痛苦", "委屈", "伤心",
            "情绪", "心情", "开心不起来", "提不起", "没意思", "没动力",
            "烦躁", "恐惧", "紧张", "不想活",
        ]
        has_somatic = any(w in text for w in _somatic_kw)
        has_emotional = any(w in text for w in _emotional_kw)

        if has_somatic and not has_emotional:
            return "somatic"

        # 4. Expressive: 主动表达情绪，内容足够
        if has_emotional and length >= 5:
            return "expressive"

        # 5. Silent: 回复极短或内容单薄
        if length < 15:
            return "silent"

        return "silent"

    def _is_disorganized(self, text: str) -> bool:
        # Simple heuristic: multiple topic shifts indicated by transition words
        shift_markers = ["对了", "还有", "然后", "另外", "突然想到", "我还想说"]
        return sum(1 for m in shift_markers if m in text) >= 2

    def _build_brain_payload(self, state: DebateState, turn: int) -> str:
        history = state.get("history", [])
        recent = history[-(HISTORY_WINDOW):]
        return json.dumps(
            {
                "turn": turn,
                "stage": state.get("stage"),
                "verbal_style": state.get("verbal_style", "unknown"),
                "alliance_score": state.get("alliance_score", 0.5),
                "is_returning_patient": state.get("is_returning_patient", False),
                "portrait": state.get("portrait", {}),
                "latest_user_input": state.get("user_input", ""),
                "recent_history": recent,
                "risk_level": state.get("risk_level", "low"),
            },
            ensure_ascii=False,
        )
