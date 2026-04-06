"""
端到端测试：模拟精神分裂症患者对话
- Mock DB（无需MySQL）
- 真实 LLM API 调用
- 自动输入患者回复，观察系统表现
"""
from __future__ import annotations

import asyncio
import json
import sys
import uuid
from unittest.mock import MagicMock, patch

from dotenv import load_dotenv

load_dotenv()

# ── Mock DB层 ────────────────────────────────────────────────────────────────
_portrait_store: dict = {}
_session_store: dict = {}
_patient_store: dict = {}

def mock_load_patient(student_id):
    if student_id in _patient_store:
        return {"is_new": False, "verbal_style": _patient_store[student_id].get("verbal_style"), "portrait": _portrait_store.get(student_id)}
    return {"is_new": True, "verbal_style": None, "portrait": None}

def mock_upsert_patient(student_id, verbal_style=None):
    if student_id not in _patient_store:
        _patient_store[student_id] = {}
    if verbal_style:
        _patient_store[student_id]["verbal_style"] = verbal_style

def mock_save_portrait(student_id, portrait):
    _portrait_store[student_id] = portrait

def mock_create_session_record(student_id):
    sid = str(uuid.uuid4())
    _session_store[sid] = {"student_id": student_id}
    return sid

def mock_close_session_record(session_id, stage_reached, turn_count, risk_level, report):
    if session_id in _session_store:
        _session_store[session_id].update({
            "stage_reached": stage_reached,
            "turn_count": turn_count,
            "risk_level": risk_level,
            "report": report,
        })

# ── 精神分裂患者剧本 ─────────────────────────────────────────────────────────
# 模拟典型精神分裂症表现：
# - 被害妄想（邻居监视自己）
# - 幻听（听到声音命令自己）
# - 思维松弛（话题跳跃）
# - 情感淡漠
# - 功能损害（停学、不出门）
PATIENT_SCRIPT = [
    "202301001",                                          # 学号
    "我也不知道怎么说……就是感觉不对劲。",                      # T1 入场，略显迟疑
    "那个声音又来了，昨晚一直说让我别睡觉，说睡着了就危险了。",    # T2 幻听
    "对，就是声音，不是真人，但很清楚。已经好几个月了。对了，我邻居最近老盯着我，我觉得他们在录我。",  # T3 幻听细化+被害妄想（思维跳跃）
    "我不知道他们为什么要监视我，可能跟那件事有关。反正我现在不怎么出门了，怕被看见。",   # T4 妄想+回避行为
    "上学？我两个月没去了。课也没法上，脑子里乱，而且去了也觉得大家都在议论我。",      # T5 功能损害+关系妄想
    "以前挺好的，大一大二成绩还可以，就是去年下半年开始的，先是睡不好，然后声音就来了。",  # T6 时间线
    "怎么说呢，就是……麻木？不怎么高兴也不怎么难过。朋友说我变了很多，我自己感觉不到。",  # T7 情感平淡（阴性症状）
    "家里人不知道，我没说。他们以为我正常上学呢。",                 # T8 隐瞒，隔离
    "我没想过死，但有时候觉得这样下去没什么意思。",                  # T9 消极想法（risk探测）
    "我不确定，就是觉得自己和别人不一样了，好像隔了一层玻璃。",       # T10 人格解体感
]

# ── 主流程 ───────────────────────────────────────────────────────────────────
async def run_session():
    with patch("src.psy_debate.nodes.load_patient", side_effect=mock_load_patient), \
         patch("src.psy_debate.nodes.upsert_patient", side_effect=mock_upsert_patient), \
         patch("src.psy_debate.nodes.save_portrait", side_effect=mock_save_portrait), \
         patch("src.psy_debate.nodes.create_session_record", side_effect=mock_create_session_record), \
         patch("src.psy_debate.db.init_db", return_value=None):

        from src.psy_debate.models import DeepSeekHub
        from src.psy_debate.graph import build_graph

        hub = DeepSeekHub()
        graph = build_graph(hub)

        state: dict = {
            "student_id": "",
            "session_id": "",
            "user_input": "",
            "assistant_output": "",
            "history": [],
            "stage": "awaiting_student_id",
            "phase": "normal",
            "turn_count": 0,
            "fallback_counter": 0,
            "verbal_style": "unknown",
            "alliance_score": 0.5,
            "portrait": {},
            "hypotheses": [],
            "risk_level": "low",
            "risk_factors": [],
            "handoff_required": False,
            "handoff_reason": "",
            "session_report": {},
            "is_returning_patient": False,
            "input_safety": {},
            "output_safety": {},
            "debug_notes": "",
        }

        print("=" * 70)
        print("  精神分裂症患者模拟对话测试")
        print("=" * 70)

        for i, patient_input in enumerate(PATIENT_SCRIPT):
            state["user_input"] = patient_input

            if i == 0:
                print(f"\n[患者输入学号]: {patient_input}")
            else:
                print(f"\n[患者 T{i}]: {patient_input}")

            result = await graph.ainvoke(state)
            state.update(result)

            print(f"[系统回复]: {state.get('assistant_output', '')}")
            print(f"  stage={state.get('stage')} | phase={state.get('phase')} | "
                  f"turn={state.get('turn_count')} | risk={state.get('risk_level')} | "
                  f"alliance={state.get('alliance_score', 0):.2f} | "
                  f"verbal={state.get('verbal_style')}")

            # 症状摘要
            symptoms = state.get("portrait", {}).get("symptoms", {})
            if symptoms:
                sym_str = ", ".join(
                    f"{k}({v.get('status','?')})"
                    for k, v in symptoms.items()
                )
                print(f"  症状画像: {sym_str}")

            # 假设摘要
            hypotheses = state.get("portrait", {}).get("hypotheses", [])
            if hypotheses:
                hyp_str = " | ".join(
                    f"{h.get('disorder_cn','?')} conf={h.get('confidence',0):.2f}"
                    for h in hypotheses
                )
                print(f"  诊断假设: {hyp_str}")

            if state.get("debug_notes"):
                print(f"  [debug]: {state['debug_notes'][:120]}")

            # 如果进入危机或报告生成阶段，提前结束
            if state.get("phase") == "crisis" or state.get("stage") in ("conclusion", "forced_closure"):
                print("\n[系统判断对话结束，生成报告中...]")
                break

        # ── 若报告未自然触发，强制生成（模拟会话超时收尾）──────────────────
        report = state.get("session_report", {})
        if not report:
            print("\n[脚本结束，报告未自然触发，强制调用 generate_report...]")
            from src.psy_debate.nodes import PsyNodes
            nodes = PsyNodes(hub)
            # 标记为 forced_closure 以便 generate_report 正常处理
            state["stage"] = "forced_closure"
            report_result = await nodes.generate_report(state)
            state.update(report_result)
            report = state.get("session_report", {})

        # ── 打印最终报告 ─────────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("  最终医生报告")
        print("=" * 70)
        if report:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print("[报告生成失败]")
            print(json.dumps(state.get("portrait", {}), ensure_ascii=False, indent=2))

        return state

if __name__ == "__main__":
    asyncio.run(run_session())
