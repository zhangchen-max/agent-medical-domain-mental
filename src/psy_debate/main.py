from __future__ import annotations

import asyncio
import json
from typing import Any

from dotenv import load_dotenv

from .db import init_db
from .graph import build_graph
from .models import DeepSeekHub
from .schema import DebateState


def _initial_state() -> DebateState:
    return DebateState(
        student_id="",
        session_id="",
        is_returning_patient=False,
        turn_count=0,
        user_input="",
        assistant_output="",
        history=[],
        verbal_style="unknown",
        alliance_score=0.5,
        stage="awaiting_student_id",
        fallback_counter=0,
        portrait={
            "symptoms": {},
            "timeline": {"anchor_established": False},
            "functional_impact": {},
            "hypotheses": [],
            "consistency_score": 1.0,
            "somatic_bridge_complete": False,
            "consecutive_focused_turns": 0,
        },
        risk_level="low",
        risk_factors=[],
        phase="normal",
        input_safety={},
        output_safety={},
        handoff_required=False,
        handoff_reason="",
    )


async def _chat_loop() -> None:
    load_dotenv()
    init_db()

    hub = DeepSeekHub()
    app = build_graph(hub)
    state = _initial_state()

    print("心理健康初诊系统已启动，输入 exit 退出。")
    print("=" * 50)

    # Trigger the first turn to get the welcome/student-ID prompt
    state["user_input"] = ""
    try:
        result = await app.ainvoke(state)
        state = {**state, **result}
        print(f"\n系统> {state.get('assistant_output', '')}")
    except Exception as exc:
        print(f"[启动失败] {exc}")
        return

    while True:
        user_text = input("\n患者> ").strip()
        if user_text.lower() in {"exit", "quit", "退出"}:
            print("会话结束。")
            break
        if not user_text:
            continue

        state["user_input"] = user_text
        try:
            result = await app.ainvoke(state)
            state = {**state, **result}
        except Exception as exc:
            print(f"[error] {type(exc).__name__}: {exc}")
            continue

        print(f"\n系统> {state.get('assistant_output', '')}")
        print(
            f"[debug] stage={state.get('stage')} | "
            f"turn={state.get('turn_count')} | "
            f"risk={state.get('risk_level')} | "
            f"alliance={state.get('alliance_score', 0):.2f} | "
            f"style={state.get('verbal_style')}"
        )

        portrait = state.get("portrait", {})
        confirmed = [k for k, v in portrait.get("symptoms", {}).items() if v.get("status") == "confirmed"]
        probable = [k for k, v in portrait.get("symptoms", {}).items() if v.get("status") == "probable"]
        hypotheses = portrait.get("hypotheses", [])
        if hypotheses:
            top = max(hypotheses, key=lambda h: h.get("confidence", 0))
            print(f"[hypothesis] {top.get('disorder_cn', top.get('disorder'))} conf={top.get('confidence', 0):.2f} missing={top.get('missing_criteria', [])}")
        if confirmed or probable:
            print(f"[symptoms] confirmed={confirmed} probable={probable}")

        if state.get("session_report"):
            report = state["session_report"]
            print("\n" + "=" * 50)
            print("[临床报告（仅供医生参考）]")
            print(json.dumps(report, ensure_ascii=False, indent=2))
            print("=" * 50)


def run() -> None:
    asyncio.run(_chat_loop())


if __name__ == "__main__":
    run()
