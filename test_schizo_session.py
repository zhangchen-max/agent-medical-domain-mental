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
    "202301999",                                                                         # 学号（案例二：15岁女性，精神分裂症谱系）
    "双相感情障碍。",                                                                     # T1 自我诊断，不恰当地笑
    "因为那个同学然后还有老师们。",                                                         # T2 答非所问，语义跳跃
    "组团打游戏不带我。",                                                                  # T3 具体事件，逻辑关联不清
    "不是是在家里玩游戏，然后之后然后为了玩游戏，然后登自己爸的号，然后和她爸一起得分。",         # T4 话题跳跃，联想散漫
    "憨憨笑，然后那次，任宣。",                                                             # T5 词语贫乏，不知所云
    "同学，好闺蜜，然后完事之后我俩本来要学习的，然后我用计算机卡卡嗯，然后我算出来得数算对了，然后那次我俩去吃饭，吃饭的时候服务员账算错了，然后我俩又算了一遍，然后完了之后服务员气急败坏的走了。",  # T6 大量细节但与主题无关
    "她抢我活干。",                                                                       # T7 莫名其妙的结论
    "因为她说她胖。",                                                                     # T8 荒诞的逻辑跳跃
    "食欲挺好，睡觉也挺好，吃得挺多。",                                                     # T9 躯体问询（食欲睡眠）
    "变胖一点。",                                                                        # T10 体重问题
    "比如高考才考97.5分，初升高没考好。",                                                   # T11 学业退步（原因模糊）
    "对，就是好笑。不知道为什么笑，控制不住，有时候半夜笑醒。",                                # T12 不恰当情感（不自主大笑）
    "有。是别人故意捣乱坏我的。",                                                           # T13 被害想法
    "54个。有人告诉我的，有人偷偷打字。",                                                   # T14 幻听/思维插入（精确数字来自"有人"）
    "有，太多声音跟我讲话了，我听不懂。",                                                   # T15 明确幻听
    "从看到她那天起，两年半，可能就一个月，我一个月请了一次病假，然后我又生病了。",              # T16 病程（时间混乱）
    "发病以后不好的，之前学习挺好的。",                                                     # T17 功能退步时间线
    "护士。公务员。",                                                                     # T18 父母职业（被问及）
    "让我出门，我要去锦江山玩。",                                                           # T19 就诊动机（不自知有病）
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
