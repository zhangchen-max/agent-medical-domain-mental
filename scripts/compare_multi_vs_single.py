from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

from psy_debate.graph import build_graph
from psy_debate.main import _initial_state
from psy_debate.models import DomesticModelHub


BASELINE_SYSTEM_PROMPT = """
你是一名精神科门诊初诊医生，请用温和、专业、非评判的语气与患者对话。
目标是帮助患者更清楚地描述症状、时间线和风险信息。
要求:
1. 每轮只问一个主问题，必要时最多加一个很短的补充问题。
2. 不下最终诊断，不给确定性病名结论。
3. 若有明显自伤/他伤风险，请先做安全确认。
""".strip()

DEFAULT_UTTERANCES = [
    "我最近总觉得有人在监视我，还经常听到有人在背后骂我。",
    "这些声音大概持续半年了，晚上更明显，我几乎睡不着。",
    "我不太敢出门，感觉路人都在针对我。",
    "我上班效率也下降了，和同事基本不说话。",
    "以前没有这么严重，最近两个月明显加重。",
    "有时候我会怀疑家里装了摄像头。",
    "我会反复检查门窗，基本每天都这样。",
    "我现在没有想伤害自己或别人的打算，但非常害怕。",
    "家里人说我变得很敏感，脾气也更急了。",
    "我想知道我现在该怎么处理，先从哪里开始。",
]


@dataclass
class TurnResult:
    turn: int
    user_text: str
    multi_output: str
    multi_latency_s: float
    multi_stage: str
    multi_risk: str
    single_output: str
    single_latency_s: float


def _timeout_seconds() -> float:
    raw = os.getenv("MODEL_TIMEOUT_SECONDS", "45").strip()
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 45.0


def _single_max_tokens() -> int:
    raw = os.getenv("SINGLE_MAX_TOKENS", "480").strip()
    try:
        return max(64, int(raw))
    except ValueError:
        return 480


async def _run_multi_turns(hub: DomesticModelHub, utterances: list[str]) -> list[dict[str, Any]]:
    app = build_graph(hub)
    state = _initial_state()
    rows: list[dict[str, Any]] = []

    for idx, text in enumerate(utterances, start=1):
        state["user_input"] = text
        start = time.perf_counter()
        try:
            result = await app.ainvoke(state)
            state = {**state, **result}
            output = state.get("assistant_output", "")
            stage = state.get("stage", "")
            risk = state.get("risk_level", "")
        except Exception as exc:  # noqa: BLE001
            output = f"[ERROR] {type(exc).__name__}: {exc}"
            stage = state.get("stage", "")
            risk = state.get("risk_level", "")
        cost = time.perf_counter() - start
        rows.append(
            {
                "turn": idx,
                "user_text": text,
                "multi_output": output,
                "multi_latency_s": cost,
                "multi_stage": stage,
                "multi_risk": risk,
            }
        )
    return rows


async def _run_single_turns(hub: DomesticModelHub, utterances: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    messages: list[dict[str, str]] = [{"role": "system", "content": BASELINE_SYSTEM_PROMPT}]

    for idx, text in enumerate(utterances, start=1):
        messages.append({"role": "user", "content": text})
        start = time.perf_counter()
        try:
            completion = await hub.qwen.chat.completions.create(
                model=os.getenv("SINGLE_MODEL_NAME", hub.model_names.qwen_max),
                temperature=0.4,
                max_tokens=_single_max_tokens(),
                timeout=_timeout_seconds(),
                messages=messages,
            )
            output = completion.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            output = f"[ERROR] {type(exc).__name__}: {exc}"
        cost = time.perf_counter() - start
        messages.append({"role": "assistant", "content": output})
        rows.append({"turn": idx, "single_output": output, "single_latency_s": cost})
    return rows


def _print_report(merged: list[TurnResult]) -> None:
    print("=== A/B 对比（多节点系统 vs 单模型Prompt）===")
    for row in merged:
        print(f"\n--- Turn {row.turn} ---")
        print(f"用户: {row.user_text}")
        print(f"[多节点] 耗时={row.multi_latency_s:.2f}s stage={row.multi_stage} risk={row.multi_risk}")
        print(f"[多节点] 输出: {row.multi_output}")
        print(f"[单模型] 耗时={row.single_latency_s:.2f}s")
        print(f"[单模型] 输出: {row.single_output}")

    multi_avg = sum(x.multi_latency_s for x in merged) / len(merged)
    single_avg = sum(x.single_latency_s for x in merged) / len(merged)
    print("\n=== 汇总 ===")
    print(f"轮数: {len(merged)}")
    print(f"多节点平均耗时: {multi_avg:.2f}s/轮")
    print(f"单模型平均耗时: {single_avg:.2f}s/轮")


async def main() -> None:
    load_dotenv()
    hub = DomesticModelHub()

    utterances = DEFAULT_UTTERANCES
    multi_rows, single_rows = await asyncio.gather(
        _run_multi_turns(hub, utterances),
        _run_single_turns(hub, utterances),
    )

    merged: list[TurnResult] = []
    for m, s in zip(multi_rows, single_rows):
        merged.append(
            TurnResult(
                turn=m["turn"],
                user_text=m["user_text"],
                multi_output=m["multi_output"],
                multi_latency_s=m["multi_latency_s"],
                multi_stage=m["multi_stage"],
                multi_risk=m["multi_risk"],
                single_output=s["single_output"],
                single_latency_s=s["single_latency_s"],
            )
        )
    _print_report(merged)


if __name__ == "__main__":
    asyncio.run(main())
