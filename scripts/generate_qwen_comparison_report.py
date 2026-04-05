from __future__ import annotations

import argparse
import asyncio
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

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
    "我最近总觉得有人在监视我。",
    "我晚上经常听到有人在背后议论我。",
    "这些声音大概从半年前开始，最近两个月更明显。",
    "我现在出门会很紧张，总觉得路人盯着我。",
    "我上班注意力下降，和同事也不太说话了。",
    "我有时怀疑家里被装了摄像头。",
    "这段时间睡得很差，经常到凌晨三四点还睡不着。",
    "我没有喝酒，也没有用任何毒品。",
    "我现在没有想伤害自己或别人的想法。",
    "我想知道下一步该怎么做，先从哪里开始。",
]


@dataclass
class MultiTurnResult:
    turn: int
    user_text: str
    output: str
    latency_s: float
    stage: str
    risk: str
    phase: str


@dataclass
class SingleTurnResult:
    turn: int
    output: str
    latency_s: float


def _float_env(name: str, default: float, *, min_value: float = 1.0) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value >= min_value else default


def _int_env(name: str, default: int, *, min_value: int = 64) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= min_value else default


def _md_escape_inline(text: str) -> str:
    return text.replace("|", "\\|")


def _md_block(text: str) -> str:
    if not text.strip():
        return "_(空)_"
    return "\n".join(f"> {line}" for line in text.strip().splitlines())


async def _run_multi_system(hub: DomesticModelHub, utterances: list[str]) -> list[MultiTurnResult]:
    app = build_graph(hub)
    state = _initial_state()
    results: list[MultiTurnResult] = []

    for i, text in enumerate(utterances, start=1):
        state["user_input"] = text
        t0 = time.perf_counter()
        try:
            result = await app.ainvoke(state)
            state = {**state, **result}
            output = state.get("assistant_output", "")
            stage = str(state.get("stage", ""))
            risk = str(state.get("risk_level", ""))
            phase = str(state.get("phase", ""))
        except Exception as exc:  # noqa: BLE001
            output = f"[ERROR] {type(exc).__name__}: {exc}"
            stage = str(state.get("stage", ""))
            risk = str(state.get("risk_level", ""))
            phase = str(state.get("phase", ""))
        latency_s = time.perf_counter() - t0
        results.append(
            MultiTurnResult(
                turn=i,
                user_text=text,
                output=output,
                latency_s=latency_s,
                stage=stage,
                risk=risk,
                phase=phase,
            )
        )
    return results


async def _run_qwen_single(hub: DomesticModelHub, utterances: list[str], model_name: str) -> list[SingleTurnResult]:
    timeout_s = _float_env("MODEL_TIMEOUT_SECONDS", 45.0)
    max_tokens = _int_env("SINGLE_MAX_TOKENS", 500)

    results: list[SingleTurnResult] = []
    messages: list[dict[str, str]] = [{"role": "system", "content": BASELINE_SYSTEM_PROMPT}]
    for i, text in enumerate(utterances, start=1):
        messages.append({"role": "user", "content": text})
        t0 = time.perf_counter()
        try:
            completion = await hub.qwen.chat.completions.create(
                model=model_name,
                temperature=0.4,
                max_tokens=max_tokens,
                timeout=timeout_s,
                messages=messages,
            )
            output = completion.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            output = f"[ERROR] {type(exc).__name__}: {exc}"
        latency_s = time.perf_counter() - t0
        messages.append({"role": "assistant", "content": output})
        results.append(SingleTurnResult(turn=i, output=output, latency_s=latency_s))
    return results


def _build_markdown_report(
    *,
    multi: list[MultiTurnResult],
    single: list[SingleTurnResult],
    single_model_name: str,
    generated_at: datetime,
) -> str:
    total_turns = len(multi)
    multi_avg = sum(x.latency_s for x in multi) / total_turns if total_turns else 0.0
    single_avg = sum(x.latency_s for x in single) / total_turns if total_turns else 0.0

    lines: list[str] = []
    lines.append("# 10轮对话对比记录：多节点系统 vs 云端Qwen")
    lines.append("")
    lines.append(f"- 生成时间: `{generated_at.isoformat(timespec='seconds')}`")
    lines.append(f"- 轮数: `{total_turns}`")
    lines.append(f"- 对比对象A: `当前多节点系统`")
    lines.append(f"- 对比对象B: `云端Qwen单模型（{single_model_name}）`")
    lines.append("")
    lines.append("## 汇总")
    lines.append("")
    lines.append(f"- 多节点系统平均耗时: `{multi_avg:.2f}s/轮`")
    lines.append(f"- 云端Qwen平均耗时: `{single_avg:.2f}s/轮`")
    lines.append("")
    lines.append("## 逐轮对话")
    lines.append("")

    for m, s in zip(multi, single):
        lines.append(f"### 第{m.turn}轮")
        lines.append("")
        lines.append(f"- 患者输入: {_md_escape_inline(m.user_text)}")
        lines.append(
            f"- 多节点状态: `stage={m.stage or '-'} risk={m.risk or '-'} phase={m.phase or '-'} latency={m.latency_s:.2f}s`"
        )
        lines.append(f"- Qwen耗时: `{s.latency_s:.2f}s`")
        lines.append("")
        lines.append("多节点系统回复:")
        lines.append(_md_block(m.output))
        lines.append("")
        lines.append("云端Qwen回复:")
        lines.append(_md_block(s.output))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 10-turn comparison report: multi-system vs Qwen single model")
    parser.add_argument(
        "--out",
        default="reports/qwen_comparison_report.md",
        help="Output markdown path",
    )
    parser.add_argument(
        "--single-model",
        default=os.getenv("SINGLE_MODEL_NAME", "qwen-max"),
        help="Qwen model name for single-model baseline",
    )
    parser.add_argument(
        "--qwen-api-key",
        default="",
        help="Optional override for QWEN_API_KEY (if omitted, read from .env)",
    )
    return parser.parse_args()


async def _async_main(args: argparse.Namespace) -> Path:
    load_dotenv()
    if args.qwen_api_key.strip():
        os.environ["QWEN_API_KEY"] = args.qwen_api_key.strip()

    hub = DomesticModelHub()
    utterances = DEFAULT_UTTERANCES

    multi_results, single_results = await asyncio.gather(
        _run_multi_system(hub, utterances),
        _run_qwen_single(hub, utterances, args.single_model),
    )

    report = _build_markdown_report(
        multi=multi_results,
        single=single_results,
        single_model_name=args.single_model,
        generated_at=datetime.now(),
    )

    out_path = Path(args.out).expanduser()
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    return out_path


def main() -> None:
    args = _parse_args()
    out_path = asyncio.run(_async_main(args))
    print(f"report_saved={out_path}")


if __name__ == "__main__":
    main()
