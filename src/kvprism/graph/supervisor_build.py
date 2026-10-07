"""Supervisor 중심 그래프와 하위 노드의 입출력·재작업 계약을 조립한다."""

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

from langgraph.graph import END, START, StateGraph

from .state import ResearchResult, PerspectiveResult, SynthesisResult, merge_sources, validate_graph_state
from .supervisor_state import ReportArtifact, ReportQualityResult, SupervisorState, WORKER_KEYS
from ..rag.config import ROOT

NodeFunction = Callable[[SupervisorState], dict]


@dataclass(frozen=True)
class SupervisorNodes:
    supervisor: NodeFunction
    research: NodeFunction
    market: NodeFunction
    stakeholder: NodeFunction
    domain: NodeFunction
    synthesize: NodeFunction
    report: NodeFunction
    report_quality: NodeFunction


RESULT_MODELS = {
    "research": ResearchResult, "market": PerspectiveResult,
    "stakeholder": PerspectiveResult, "domain": PerspectiveResult,
    "synthesize": SynthesisResult, "report": ReportArtifact, "report_quality": ReportQualityResult,
}
DOWNSTREAM = {
    "research": ["market_eval", "stakeholder_eval", "domain_eval", "synthesis", "report", "quality"],
    "market": ["synthesis", "report", "quality"],
    "stakeholder": ["synthesis", "report", "quality"],
    "domain": ["synthesis", "report", "quality"],
    "synthesize": ["report", "quality"], "report": ["quality"], "report_quality": [],
}
INPUT_KEYS = {
    "research": ("research",),
    "market": ("research", "market_eval"),
    "stakeholder": ("research", "stakeholder_eval"),
    "domain": ("research", "domain_eval"),
    "synthesize": ("research", "market_eval", "stakeholder_eval", "domain_eval", "synthesis"),
    "report": ("research", "market_eval", "stakeholder_eval", "domain_eval", "synthesis"),
    "report_quality": ("report", "synthesis"),
}
COLLECTORS = {"research", "market", "stakeholder", "domain"}


def actual_supervisor_nodes() -> SupervisorNodes:
    """기본 실행에는 실제 에이전트만 연결한다."""
    from ..agents.supervisor import supervisor_node
    from ..agents.research import research_node
    from ..agents.market import market_node
    from ..agents.stakeholder import stakeholder_node
    from ..agents.domain import domain_node
    from ..agents.synthesize import synthesize_node
    from ..agents.report import report_node
    from ..agents.report_quality import report_quality_node

    return SupervisorNodes(
        supervisor=supervisor_node,
        research=research_node,
        market=market_node,
        stakeholder=stakeholder_node,
        domain=domain_node,
        synthesize=synthesize_node,
        report=report_node,
        report_quality=report_quality_node,
    )


def record_event(output_root: Path, state, node: str, update: dict, elapsed: float) -> None:
    """제어 결정만 실행별 로그에 저장한다. 근거·LLM 대화 이력은 State에 쌓지 않는다."""
    directory = output_root / state["trace_id"]
    directory.mkdir(parents=True, exist_ok=True)
    decision = update["decision"] if "decision" in update else state.get("decision")
    event = {
        "trace_id": state["trace_id"], "node": node, "duration_seconds": round(elapsed, 3),
        "decision": decision.model_dump(mode="json") if decision else None,
        "route": update.get("route"), "status": update.get("status", state["status"]),
        "attempts": update.get("attempts", state["attempts"]),
        "error": update.get("last_error"),
    }
    with (directory / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")


def worker_wrapper(name: str, node: NodeFunction, output_root: Path, *, actual_report=False):
    """노드는 자기 결과만 반환하고 제어·무효화·계약 검증은 중앙에서 처리한다."""
    def run(state: SupervisorState) -> dict:
        started = perf_counter()
        attempts = dict(state["attempts"])
        statuses = dict(state["node_status"])
        attempts[name] += 1
        try:
            if attempts[name] > state["limits"].max_worker_attempts[name]:
                raise ValueError(f"{name} 실행 상한 초과")
            packet = {
                "request": state["request"].model_copy(deep=True),
                "decision": state["decision"].model_copy(deep=True),
            }
            for key in INPUT_KEYS[name]:
                value = state.get(key)
                packet[key] = value.model_copy(deep=True) if value is not None else None
            # 이전 결과에 인용된 원문을 함께 전달한다. 비관련 신규 출처도 해당 기술의 검색 재사용에 쓴다.
            packet["sources"] = [s.model_copy(deep=True) for s in state["sources"]]
            kwargs = {}
            if actual_report:
                kwargs["output_dir"] = output_root / state["trace_id"] / f"report-{attempts[name]}"
            output = node(packet, **kwargs)
            key = WORKER_KEYS[name]
            allowed = {key, "sources"} if name in COLLECTORS else {key}
            if not isinstance(output, dict) or key not in output or set(output) - allowed:
                raise ValueError(f"{name}의 반환 키는 {sorted(allowed)}입니다")
            result = RESULT_MODELS[name].model_validate(output[key])
            incoming = output.get("sources", [])
            merged = merge_sources(state["sources"], incoming)
            invalidated = dict.fromkeys(DOWNSTREAM[name])
            candidate = {**state, **invalidated, key: result, "sources": merged}
            validate_graph_state(candidate)
            # 성공한 결과를 반영할 때만 종속 산출물을 폐기한다.
            for other, result_key in WORKER_KEYS.items():
                if result_key in invalidated and state.get(result_key) is not None:
                    statuses[other] = "stale"
            statuses[name] = "succeeded"
            update = {
                **invalidated,
                key: result,
                "attempts": attempts,
                "node_status": statuses,
                "last_error": None,
            }
            if name in COLLECTORS:
                old_ids = {s.source_id for s in state["sources"]}
                update["sources"] = [s for s in merged if s.source_id not in old_ids]
        except Exception as error:
            statuses[name] = "failed"
            # Pydantic 오류 전체를 노출하면 원문·입력 값이 포함될 수 있다.
            message = type(error).__name__
            if type(error) is ValueError:
                message += ": " + str(error)[:800]
            update = {
                "attempts": attempts,
                "node_status": statuses,
                "last_error": f"{name}: {message}",
            }
        record_event(output_root, state, name, update, perf_counter() - started)
        return update
    return run


def route_supervisor(state: SupervisorState) -> str:
    """확정된 목적지만 전달하며 조사 순서를 결정하지 않는다."""
    return state["route"]


def build_supervisor_graph(nodes: SupervisorNodes | None = None, *, checkpointer=None,
                           output_root: str | Path | None = None):
    """모든 하위 노드가 Supervisor로 복귀하는 그래프를 컴파일한다."""
    actual = nodes is None
    nodes = nodes or actual_supervisor_nodes()
    output_root = Path(output_root) if output_root else ROOT / "outputs/runs"
    graph = StateGraph(SupervisorState)

    def supervise(state):
        start = perf_counter()
        update = nodes.supervisor(state)
        record_event(output_root, state, "supervisor", update, perf_counter() - start)
        return update

    graph.add_node("supervisor", supervise)
    graph.add_edge(START, "supervisor")
    for name in WORKER_KEYS:
        worker = worker_wrapper(
            name,
            getattr(nodes, name),
            output_root,
            actual_report=actual and name == "report",
        )
        graph.add_node(name, worker)
        graph.add_edge(name, "supervisor")
    graph.add_conditional_edges("supervisor", route_supervisor, {
        **{name: name for name in WORKER_KEYS}, "supervisor": "supervisor", "end": END,
    })
    return graph.compile(checkpointer=checkpointer)
