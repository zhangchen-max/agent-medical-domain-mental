"""完整会话模拟脚本：模拟患者从建立关系到移交/结束的全流程，含计时和记录文档输出"""
import asyncio
import json
import os
import time
from datetime import datetime

# 清除本地代理，直连服务器
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

from dotenv import load_dotenv
load_dotenv()

from src.psy_debate.graph import build_graph
from src.psy_debate.models import DomesticModelHub
from src.psy_debate.nodes import DebateNodes, _build_payload, _compact_history, _compact_value, _as_dict, MAX_PAYLOAD_TEXT_CHARS, _truncate_text
from src.psy_debate.schema import DebateState

# ── 模拟患者输入（大三男生，持续2个月抑郁/焦虑，学业压力，社会支持不足）──
PATIENT_INPUTS = [
    "我也不知道从哪里说起，就是最近状态很差",
    "大概从两个月前开始的，那时候期中考试没考好，然后就一直没恢复过来",
    "睡眠很差，每天凌晨两三点才睡着，早上七点就醒了，白天很累但就是睡不着",
    "对什么事情都提不起兴趣，以前喜欢打游戏、跑步，现在完全不想动，就躺着刷手机",
    "吃饭也不规律，有时候一天就吃一顿，不太饿，体重好像轻了五六斤",
    "上课注意力很难集中，老师讲什么都进不去，作业一直拖着，感觉自己越来越废了",
    "室友关系还好，但他们都很忙，感觉不好意思打扰他们。家里父母知道我成绩下滑了，一直催我努力，但我没法跟他们说这些",
    "有时候会觉得活着挺没意思的，觉得自己是个负担，但没有想过结束生命或者伤害自己",
    "以前高中也有过一段时间情绪很低落，但那次过了一两个月就好了，这次感觉更严重",
    "没吃过精神科的药，也没看过心理医生，就是一直扛着",
    "我不太清楚自己算不算抑郁，就是觉得很累，希望有人能帮我，我愿意去看医生",
]


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


async def diagnose_report(hub: DomesticModelHub, state: DebateState):
    """直接调用模型，打印原始 JSON 输出，不做 schema 验证"""
    from src.psy_debate.prompts import session_report_prompt
    recent_history = _compact_history(state.get("history", [])[-8:])
    scale = _as_dict(state.get("scale_opinion", {}))
    social = _as_dict(state.get("social_support_opinion", {}))
    consistency = _as_dict(state.get("consistency_opinion", {}))
    payload = json.dumps({
        "recent_history": recent_history,
        "stage": state.get("stage"),
        "risk_level": state.get("risk_level"),
        "slot_coverage": state.get("slot_coverage", 0.0),
        "hypotheses": state.get("hypotheses", []),
        "handoff_reason": state.get("handoff_reason", ""),
        "phq9_score": scale.get("phq9_estimated_score", 0),
        "phq9_items": scale.get("phq9_items_covered", []),
        "gad7_score": scale.get("gad7_estimated_score", 0),
        "gad7_items": scale.get("gad7_items_covered", []),
        "support_level": social.get("support_level", "unknown"),
        "support_sources": social.get("support_sources", []),
        "contradictions": consistency.get("contradictions", []),
    }, ensure_ascii=False)
    completion = await hub.client.chat.completions.create(
        model=hub.model_names.local_model,
        temperature=0.2,
        max_tokens=2000,
        timeout=hub.request_timeout_seconds,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": session_report_prompt()},
            {"role": "user", "content": payload},
        ],
    )
    raw = completion.choices[0].message.content or ""
    finish = completion.choices[0].finish_reason
    print(f"finish_reason={finish}")
    print(f"原始输出 (前1000字):\n{raw[:1000]}")


async def force_generate_report(hub: DomesticModelHub, state: DebateState) -> dict:
    """会话结束时若未自动触发报告，手动调用报告生成节点"""
    nodes = DebateNodes(hub)
    return await nodes.generate_report(state)


async def run_full_session():
    hub = DomesticModelHub()
    app = build_graph(hub)
    state = initial_state()

    session_start = time.time()
    turn_times = []
    records = []

    model_url = os.getenv("LOCAL_MODEL_BASE_URL", "")
    model_name = os.getenv("LOCAL_MODEL", "")
    print(f"模型: {model_name}  地址: {model_url}")
    print("=" * 70)

    for i, user_text in enumerate(PATIENT_INPUTS, 1):
        state["user_input"] = user_text

        t0 = time.time()
        try:
            result = await app.ainvoke(state)
            state = {**state, **result}
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"[第{i:02d}轮 {elapsed:.1f}s] ERROR: {type(exc).__name__}: {exc}")
            records.append({
                "turn": i, "patient": user_text,
                "assistant": f"[ERROR] {exc}",
                "elapsed": elapsed,
            })
            continue

        elapsed = time.time() - t0
        turn_times.append(elapsed)

        scale = state.get("scale_opinion", {})
        consistency = state.get("consistency_opinion", {})
        social = state.get("social_support_opinion", {})

        print(f"[第{i:02d}轮 {elapsed:.1f}s]")
        print(f"  患者> {user_text}")
        print(f"  助手> {state.get('assistant_output', '')}")
        print(f"  stage={state.get('stage')} | risk={state.get('risk_level')} | "
              f"slot={state.get('slot_coverage'):.0%} | phase={state.get('phase')}")
        print(f"  PHQ9={scale.get('phq9_estimated_score','?')}({len(scale.get('phq9_items_covered',[]))}/9) "
              f"GAD7={scale.get('gad7_estimated_score','?')}({len(scale.get('gad7_items_covered',[]))}/7) | "
              f"support={social.get('support_level','?')} | "
              f"contradiction={consistency.get('has_significant_contradiction',False)}")
        print()

        records.append({
            "turn": i,
            "patient": user_text,
            "assistant": state.get("assistant_output", ""),
            "elapsed": elapsed,
            "stage": state.get("stage"),
            "risk": state.get("risk_level"),
            "slot_coverage": state.get("slot_coverage", 0),
            "phase": state.get("phase"),
            "phq9": scale.get("phq9_estimated_score", 0),
            "phq9_covered": len(scale.get("phq9_items_covered", [])),
            "phq9_items": scale.get("phq9_items_covered", []),
            "gad7": scale.get("gad7_estimated_score", 0),
            "gad7_covered": len(scale.get("gad7_items_covered", [])),
            "gad7_items": scale.get("gad7_items_covered", []),
            "support_level": social.get("support_level", "unknown"),
            "support_sources": social.get("support_sources", []),
            "concern_flags": social.get("concern_flags", []),
            "contradictions": consistency.get("contradictions", []),
            "has_contradiction": consistency.get("has_significant_contradiction", False),
            "probe_suggestion": consistency.get("probe_suggestion", ""),
            "scale_next_q": scale.get("next_scale_question", ""),
            "hypotheses": state.get("hypotheses", []),
        })

        if state.get("handoff_required"):
            print(f"  *** 触发移交: {state.get('handoff_reason')} ***")
            break

    total_time = time.time() - session_start
    avg_time = sum(turn_times) / len(turn_times) if turn_times else 0

    print("=" * 70)
    print(f"对话结束 | 总轮数={len(records)} | 总用时={total_time:.1f}s | 平均每轮={avg_time:.1f}s")

    # ── 确保报告生成 ──
    report = state.get("session_report")
    if not report:
        print("\n[会话报告未自动触发，手动生成中...]")
        t0 = time.time()
        try:
            report_result = await force_generate_report(hub, state)
            state = {**state, **report_result}
            report = state.get("session_report")
            print(f"[报告生成完成，用时 {time.time()-t0:.1f}s]")
        except Exception as exc:
            # 打印完整异常链，找到根本原因
            import traceback
            print(f"[报告生成失败]")
            traceback.print_exc()
            # 尝试直接调用模型，打印原始输出
            print("\n[诊断：直接调用模型查看原始输出...]")
            try:
                await diagnose_report(hub, state)
            except Exception as diag_exc:
                print(f"[诊断也失败: {diag_exc}]")

    generate_markdown(records, state, report, total_time, avg_time, model_name)


def generate_markdown(records, final_state, report, total_time, avg_time, model_name):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []

    lines.append(f"# 精神科初筛访谈会话记录")
    lines.append(f"\n**生成时间：** {now}  ")
    lines.append(f"**模型：** {model_name}  ")
    lines.append(f"**总轮数：** {len(records)}  ")
    lines.append(f"**总用时：** {total_time:.1f}s  ")
    lines.append(f"**平均每轮：** {avg_time:.1f}s  ")
    lines.append(f"**最终阶段：** {final_state.get('stage')}  ")
    lines.append(f"**最终风险等级：** {final_state.get('risk_level')}  ")
    lines.append(f"**信息槽位覆盖率：** {final_state.get('slot_coverage', 0):.0%}  ")

    # 每轮计时概览
    lines.append(f"\n## 每轮用时概览")
    lines.append(f"\n| 轮次 | 用时(s) | 阶段 | 风险 | 槽位 | PHQ-9 | GAD-7 | 社会支持 | 矛盾 |")
    lines.append(f"|------|---------|------|------|------|-------|-------|----------|------|")
    for r in records:
        lines.append(
            f"| 第{r['turn']:02d}轮 | {r.get('elapsed', 0):.1f} | "
            f"{r.get('stage','-')} | {r.get('risk','-')} | "
            f"{r.get('slot_coverage', 0):.0%} | "
            f"{r.get('phq9','?')}({r.get('phq9_covered','?')}/9) | "
            f"{r.get('gad7','?')}({r.get('gad7_covered','?')}/7) | "
            f"{r.get('support_level','-')} | "
            f"{'⚠️' if r.get('has_contradiction') else '—'} |"
        )

    # 完整对话
    lines.append(f"\n## 完整对话记录")
    for r in records:
        lines.append(f"\n### 第{r['turn']:02d}轮（用时 {r.get('elapsed',0):.1f}s）")
        lines.append(f"\n**患者：** {r['patient']}")
        lines.append(f"\n**助手：** {r['assistant']}")
        lines.append(f"\n> `stage={r.get('stage','-')}` · `risk={r.get('risk','-')}` · "
                     f"`phase={r.get('phase','-')}` · `slot={r.get('slot_coverage',0):.0%}`")

        lines.append(f"\n**量表追踪**")
        lines.append(f"- PHQ-9 估分：**{r.get('phq9','?')}**/27，覆盖条目：`{r.get('phq9_items', [])}`")
        lines.append(f"- GAD-7 估分：**{r.get('gad7','?')}**/21，覆盖条目：`{r.get('gad7_items', [])}`")
        if r.get("scale_next_q"):
            lines.append(f"- 建议量表问题：{r['scale_next_q']}")

        lines.append(f"\n**社会支持评估**")
        lines.append(f"- 支持水平：`{r.get('support_level','unknown')}`")
        lines.append(f"- 支持来源：{r.get('support_sources', [])}")
        lines.append(f"- 关注点：{r.get('concern_flags', [])}")

        lines.append(f"\n**一致性检查**")
        if r.get("has_contradiction"):
            for c in r.get("contradictions", []):
                lines.append(f"- ⚠️ [{c.get('severity','?')}] {c.get('item','')}")
            if r.get("probe_suggestion"):
                lines.append(f"- 建议澄清方式：{r['probe_suggestion']}")
        else:
            lines.append(f"- 未发现显著矛盾")

        if r.get("hypotheses"):
            lines.append(f"\n**临床假设：** {', '.join(r['hypotheses'])}")

    # 新功能汇总
    lines.append(f"\n## 新增功能运行汇总")

    last = records[-1] if records else {}

    lines.append(f"\n### 量表追踪最终状态")
    lines.append(f"- PHQ-9 最终估分：**{last.get('phq9','?')}** / 27")
    lines.append(f"  - 覆盖条目（{last.get('phq9_covered','?')}/9）：{last.get('phq9_items', [])}")
    lines.append(f"- GAD-7 最终估分：**{last.get('gad7','?')}** / 21")
    lines.append(f"  - 覆盖条目（{last.get('gad7_covered','?')}/7）：{last.get('gad7_items', [])}")

    lines.append(f"\n### 社会支持最终评估")
    lines.append(f"- 支持水平：**{last.get('support_level','?')}**")
    lines.append(f"- 来源：{last.get('support_sources', [])}")
    lines.append(f"- 关注点：{last.get('concern_flags', [])}")

    has_any_contradiction = any(r.get("has_contradiction") for r in records)
    contradiction_turns = [r["turn"] for r in records if r.get("has_contradiction")]
    lines.append(f"\n### 一致性检查汇总")
    lines.append(f"- 全程发现重要矛盾：{'是' if has_any_contradiction else '否'}")
    if has_any_contradiction:
        lines.append(f"- 发现矛盾的轮次：第 {', '.join(map(str, contradiction_turns))} 轮")

    # 会话报告
    lines.append(f"\n---\n")
    lines.append(f"## 会话临床摘要报告（自动生成）")

    if report:
        lines.append(f"\n| 项目 | 内容 |")
        lines.append(f"|------|------|")
        lines.append(f"| PHQ-9 估分 | **{report.get('phq9_estimated_score')}** / 27 |")
        lines.append(f"| GAD-7 估分 | **{report.get('gad7_estimated_score')}** / 21 |")
        lines.append(f"| 槽位覆盖率 | {report.get('slot_coverage', 0):.0%} |")
        lines.append(f"| 移交原因 | {report.get('handoff_reason') or '—'} |")

        lines.append(f"\n**主诉摘要**\n\n{report.get('chief_complaint_summary', '')}")

        lines.append(f"\n**主要症状**")
        for s in report.get("symptom_summary", []):
            lines.append(f"- {s}")

        lines.append(f"\n**症状时间线**\n\n{report.get('timeline_summary', '')}")

        lines.append(f"\n**风险评估**\n\n{report.get('risk_summary', '')}")

        lines.append(f"\n**社会支持**\n\n{report.get('social_support_summary', '')}")

        if report.get("consistency_flags"):
            lines.append(f"\n**陈述矛盾**")
            for f in report["consistency_flags"]:
                lines.append(f"- {f}")

        if report.get("hypotheses"):
            lines.append(f"\n**临床假设**")
            for h in report["hypotheses"]:
                lines.append(f"- {h}")

        lines.append(f"\n**建议后续步骤**")
        for s in report.get("recommended_next_steps", []):
            lines.append(f"- {s}")
    else:
        lines.append(f"\n*报告生成失败。*")

    out_path = "C:/Users/31906/Desktop/1/session_record.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n记录文档已保存: {out_path}")


if __name__ == "__main__":
    asyncio.run(run_full_session())
