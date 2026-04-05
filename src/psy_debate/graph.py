from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .models import DeepSeekHub
from .nodes import PsyNodes
from .schema import DebateState


def build_graph(hub: DeepSeekHub):
    nodes = PsyNodes(hub)
    graph = StateGraph(DebateState)

    # ── Register nodes ────────────────────────────────────────────────────
    graph.add_node("session_init", nodes.session_init)
    graph.add_node("input_safety", nodes.input_safety)
    graph.add_node("analyze", nodes.analyze)
    graph.add_node("crisis", nodes.crisis)
    graph.add_node("output_safety", nodes.output_safety)
    graph.add_node("generate_report", nodes.generate_report)

    # output_only: for the "awaiting_student_id" path — skip analysis
    # We reuse output_safety directly since it just formats and returns
    graph.add_node("output_only", nodes.output_safety)

    # ── Edges ─────────────────────────────────────────────────────────────
    graph.add_edge(START, "session_init")

    graph.add_conditional_edges(
        "session_init",
        nodes.route_after_session_init,
        {"output_only": "output_only", "continue": "input_safety"},
    )

    graph.add_edge("output_only", END)
    graph.add_edge("input_safety", "analyze")

    graph.add_conditional_edges(
        "analyze",
        nodes.route_after_analyze,
        {
            "crisis": "crisis",
            "output_safety": "output_safety",
            "generate_report": "output_safety",  # still format output first
        },
    )

    graph.add_edge("crisis", "output_safety")

    graph.add_conditional_edges(
        "output_safety",
        nodes.route_after_output,
        {"generate_report": "generate_report", "end": END},
    )

    graph.add_edge("generate_report", END)

    return graph.compile()
