"""LangGraph 실행 흐름을 조립한다.

이 모듈은 각 에이전트의 업무 로직을 구현하지 않는다. 외부에서 전달받은 노드
함수를 설계서의 실행 순서에 맞게 연결하고, 미달 관점만 한 번 재실행한다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .state import GraphState, Perspective


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 노드 이름
# ---------------------------------------------------------------------------


RESEARCH = "research"
MARKET = "market"
STAKEHOLDER = "stakeholder"
DOMAIN = "domain"
JUDGE = "judge"
PREPARE_RETRY = "prepare_retry"
SYNTHESIZE = "synthesize"
REPORT = "report"

PERSPECTIVE_NODES: dict[Perspective, str] = {
    "market": MARKET,
    "stakeholder": STAKEHOLDER,
    "domain": DOMAIN,
}


# ---------------------------------------------------------------------------
# 외부에서 주입받는 에이전트 노드
# ---------------------------------------------------------------------------


NodeFunction = Callable[[GraphState], dict]


@dataclass(frozen=True)
class GraphNodes:
    """그래프 조립에 필요한 일곱 개 에이전트 함수."""

    research: NodeFunction
    market: NodeFunction
    stakeholder: NodeFunction
    domain: NodeFunction
    judge: NodeFunction
    synthesize: NodeFunction
    report: NodeFunction


def actual_graph_nodes() -> GraphNodes:
    """현재 패키지에 구현된 실제 에이전트 함수 일곱 개를 반환한다."""

    # 모듈을 불러올 때 외부 도구 의존성까지 로딩하지 않도록 조립 시점에 가져온다.
    from ..agents.domain import domain_node
    from ..agents.judge import judge_node
    from ..agents.market import market_node
    from ..agents.report import report_node
    from ..agents.research import research_node
    from ..agents.stakeholder import stakeholder_node
    from ..agents.synthesize import synthesize_node

    return GraphNodes(
        research=research_node,
        market=market_node,
        stakeholder=stakeholder_node,
        domain=domain_node,
        judge=judge_node,
        synthesize=synthesize_node,
        report=report_node,
    )


# ---------------------------------------------------------------------------
# 재실행 제어
# ---------------------------------------------------------------------------


def route_after_judge(state: GraphState) -> str:
    """Judge 결과와 재실행 횟수에 따라 다음 단계를 선택한다."""

    judge = state.get("judge")
    if judge is None:
        raise ValueError("judge 노드가 JudgeResult를 반환하지 않았습니다")
    if judge.passed:
        return SYNTHESIZE
    if state.get("retry_count", 0) == 0:
        return PREPARE_RETRY
    log.warning(
        "관점별 평가를 한 번 재실행했지만 검증 미달이 남아 종합 단계로 진행합니다: %s",
        "; ".join(judge.warnings),
    )
    return SYNTHESIZE


def prepare_retry(state: GraphState) -> dict:
    """첫 검증 실패 후 재실행 횟수를 한 번만 증가시킨다."""

    if state.get("retry_count", 0) != 0:
        raise ValueError("관점별 평가는 한 번만 재실행할 수 있습니다")
    targets = state.get("retry_targets", [])
    if not targets:
        raise ValueError("검증에 실패했지만 retry_targets가 비어 있습니다")
    return {"retry_count": 1}


def route_retry_targets(state: GraphState) -> list[str]:
    """미달한 관점에 해당하는 노드 이름만 반환한다."""

    targets = state.get("retry_targets", [])
    if not targets:
        raise ValueError("재실행할 관점이 없습니다")
    return [PERSPECTIVE_NODES[target] for target in targets]


# ---------------------------------------------------------------------------
# 그래프 조립
# ---------------------------------------------------------------------------


def build_graph(nodes: GraphNodes | None = None) -> CompiledStateGraph:
    """실제 노드 또는 주입된 노드로 설계서의 실행 구조를 컴파일한다."""

    nodes = nodes or actual_graph_nodes()

    graph = StateGraph(GraphState)

    graph.add_node(RESEARCH, nodes.research)
    graph.add_node(MARKET, nodes.market)
    graph.add_node(STAKEHOLDER, nodes.stakeholder)
    graph.add_node(DOMAIN, nodes.domain)
    graph.add_node(JUDGE, nodes.judge)
    graph.add_node(PREPARE_RETRY, prepare_retry)
    graph.add_node(SYNTHESIZE, nodes.synthesize)
    graph.add_node(REPORT, nodes.report)

    graph.add_edge(START, RESEARCH)

    # 기술 조사가 끝나면 세 관점 평가를 같은 단계에서 병렬로 실행한다.
    graph.add_edge(RESEARCH, MARKET)
    graph.add_edge(RESEARCH, STAKEHOLDER)
    graph.add_edge(RESEARCH, DOMAIN)

    # 같은 단계에서 끝난 관점 결과는 다음 단계의 Judge 실행 한 번으로 합쳐진다.
    graph.add_edge(MARKET, JUDGE)
    graph.add_edge(STAKEHOLDER, JUDGE)
    graph.add_edge(DOMAIN, JUDGE)

    graph.add_conditional_edges(
        JUDGE,
        route_after_judge,
        {
            PREPARE_RETRY: PREPARE_RETRY,
            SYNTHESIZE: SYNTHESIZE,
        },
    )
    graph.add_conditional_edges(
        PREPARE_RETRY,
        route_retry_targets,
        {
            MARKET: MARKET,
            STAKEHOLDER: STAKEHOLDER,
            DOMAIN: DOMAIN,
        },
    )

    graph.add_edge(SYNTHESIZE, REPORT)
    graph.add_edge(REPORT, END)

    return graph.compile()
