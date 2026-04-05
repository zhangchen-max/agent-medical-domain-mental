"""非交互式测试脚本：连接真实模型跑几轮对话，验证新增功能"""
import asyncio
import os

# 清除本地代理，直连服务器
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

from dotenv import load_dotenv
load_dotenv()

from src.psy_debate.graph import build_graph
from src.psy_debate.models import DomesticModelHub
from src.psy_debate.schema import DebateState


def initial_state() -> DebateState:
    return DebateState(
        user_input="",
        history=[],
        stage="rapport",
        stage_confidence=0.5,
        readiness_to_advance=0.0,
        timeline_confidence=0.0,
        alliance_score=0.5,
        missing_slots=[],
        slot_coverage=0.0,
        risk_level="low",
        next_turn_goal="build_rapport",
        hypotheses=[],
        empathy_opinion={},
        completeness_opinion={},
        risk_opinion={},
        scale_opinion={},
        consistency_opinion={},
        social_support_opinion={},
        input_safety={},
        output_safety={},
        assistant_output="",
        handoff_required=False,
        handoff_reason="",
        phase="normal",
    )


TEST_INPUTS = [
    "我最近睡眠很差，每天只能睡三四个小时，已经持续一个多月了",
    "心情一直很低落，对什么事情都提不起兴趣，以前喜欢打球现在完全不想动",
    "室友关系也不好，感觉自己在宿舍里格格不入，家里人也不太理解我",
    "有时候会觉得活着没什么意思，但没有想过伤害自己",
]


async def run_test():
    hub = DomesticModelHub()
    app = build_graph(hub)
    state = initial_state()

    print("=" * 60)
    print("测试开始：连接服务器", os.getenv("LOCAL_MODEL_BASE_URL"))
    print("=" * 60)

    for i, user_text in enumerate(TEST_INPUTS, 1):
        print(f"\n[第{i}轮] 患者> {user_text}")
        state["user_input"] = user_text

        try:
            result = await app.ainvoke(state)
            state = {**state, **result}
        except Exception as exc:
            print(f"  [ERROR] {type(exc).__name__}: {exc}")
            continue

        print(f"  助手> {state.get('assistant_output', '')}")
        print(f"  stage={state.get('stage')} | risk={state.get('risk_level')} | "
              f"slot_coverage={state.get('slot_coverage'):.0%} | phase={state.get('phase')}")

        # 新功能输出
        scale = state.get("scale_opinion", {})
        print(f"  [量表] PHQ9估分={scale.get('phq9_estimated_score','?')} "
              f"(覆盖{len(scale.get('phq9_items_covered',[]))}/9) | "
              f"GAD7估分={scale.get('gad7_estimated_score','?')} "
              f"(覆盖{len(scale.get('gad7_items_covered',[]))}/7)")
        if scale.get("next_scale_question"):
            print(f"  [量表建议] {scale.get('next_scale_question')}")

        consistency = state.get("consistency_opinion", {})
        print(f"  [一致性] 发现矛盾={consistency.get('has_significant_contradiction',False)} | "
              f"矛盾数={len(consistency.get('contradictions',[]))}")
        if consistency.get("probe_suggestion"):
            print(f"  [矛盾建议] {consistency.get('probe_suggestion')}")

        social = state.get("social_support_opinion", {})
        print(f"  [社会支持] 水平={social.get('support_level','?')} | "
              f"来源={social.get('support_sources',[])} | "
              f"关注点={social.get('concern_flags',[])}")

        if state.get("session_report"):
            report = state["session_report"]
            print(f"\n{'='*60}")
            print(f"[会话报告已生成]")
            print(f"  主诉: {report.get('chief_complaint_summary')}")
            print(f"  症状: {report.get('symptom_summary')}")
            print(f"  风险: {report.get('risk_summary')}")
            print(f"  PHQ9={report.get('phq9_estimated_score')} GAD7={report.get('gad7_estimated_score')}")
            print(f"  社会支持: {report.get('social_support_summary')}")
            print(f"  后续建议: {report.get('recommended_next_steps')}")

    print("\n" + "=" * 60)
    print("测试完成")


if __name__ == "__main__":
    asyncio.run(run_test())
