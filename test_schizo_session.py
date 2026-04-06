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
    "202301888",                                                                         # 学号
    "是大脑神经感觉高度的紧张紧绷，然后会带来强烈强度的焦虑感。",                              # T1 躯体化主诉（实为躯体妄想）
    "是感觉是大脑自己有问题，出现脑神经组织总是扭挤在一起，有点神经紊乱。",                    # T2 身体体验异常，思维松弛
    "主要是麻痹僵直的状态，神经麻痹僵直的状态。",                                          # T3 重复描述，语义贫乏
    "感觉就像脑袋被固定在一个类似有机玻璃一样的真空罩里，让人喘不上来气。",                     # T4 奇特躯体比喻（钟形玻璃罩感）
    "大概是从脑袋和眼睛两个地方可以感受到别人说话传出的信号能从我这边擦肩而过，然后我接受不到。",   # T5 被动体验/信号干扰感
    "有的时候感觉自己的脑袋是通透的，就是连一阵风好像都能浮过来，直接从旁边掠走。",              # T6 人格解体/现实解体
    "没有人的时候，一紧张或者做东西，摆放书籍，整理日常用品顺序、时间，位置不对就会感觉非常紊乱。",  # T7 强迫症状（摆放仪式）
    "感觉正常应该是大脑支配我做这件事情，但现在感觉是大脑在支配我，因为他生病了让我难受，他让我来完成这些事情，我是处于被动的状态。",  # T8 被控制感（被动体验）
    "对，他经常跟我这样说。",                                                              # T9 确认与"大脑"的对话（幻听/内部对话）
    "大约都是白天的时候对话。",                                                            # T10 幻听时间特征
    "三年了，我是阶段性住院的，最长一次是春夏秋三个季度。",                                    # T11 病程信息
    "用过氯氮平，在北大六院的时候开过，效果很不好，会有神经痉挛，高度兴奋紧张。",               # T12 用药史
    "做过三个疗程的ECT，第一次感觉挺好的，后两次效果不太好。",                               # T13 治疗史
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
