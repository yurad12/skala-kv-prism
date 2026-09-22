"""LangGraph 병렬 실행과 재실행 라우팅 테스트."""

from __future__ import annotations

import unittest
from collections import Counter
from collections.abc import Iterable
from unittest.mock import patch

from kvprism.agents.domain import domain_node
from kvprism.agents.judge import judge_node
from kvprism.agents.market import market_node
from kvprism.agents.report import report_node
from kvprism.agents.research import research_node
from kvprism.agents.stakeholder import stakeholder_node
from kvprism.agents.synthesize import synthesize_node
from kvprism.graph.build import GraphNodes, actual_graph_nodes, build_graph
from kvprism.graph.state import (
    GraphState,
    JudgeChecks,
    JudgeResult,
    Perspective,
    PerspectiveJudgment,
)


PERSPECTIVES: tuple[Perspective, ...] = ("market", "stakeholder", "domain")


def make_judge_result(failed: set[Perspective]) -> JudgeResult:
    """실패 관점 집합으로 테스트용 Judge 결과를 만든다."""

    judgments: list[PerspectiveJudgment] = []
    for perspective in PERSPECTIVES:
        passed = perspective not in failed
        checks = JudgeChecks(
            evidence_balance_ok=passed,
            citations_ok=True,
            trl_basis_ok=True,
        )
        judgments.append(
            PerspectiveJudgment(
                perspective=perspective,
                passed=passed,
                checks=checks,
                issues=[] if passed else ["긍정·부정 근거가 부족합니다"],
                retry_instruction=None if passed else "반대 입장의 근거를 보완하세요",
            )
        )
    return JudgeResult(judgments=judgments)


def make_test_graph(
    judge_failures: Iterable[set[Perspective]],
) -> tuple[object, Counter]:
    """실행 횟수를 기록하는 가짜 노드로 그래프를 만든다."""

    calls: Counter = Counter()
    remaining_failures = iter(judge_failures)

    def simple_node(name: str):
        def node(_: GraphState) -> dict:
            calls[name] += 1
            return {"sources": []}

        return node

    def judge_node(_: GraphState) -> dict:
        calls["judge"] += 1
        failed = next(remaining_failures, set())
        result = make_judge_result(failed)
        return {
            "judge": result,
            "retry_targets": result.failed_perspectives,
        }

    graph = build_graph(
        GraphNodes(
            research=simple_node("research"),
            market=simple_node("market"),
            stakeholder=simple_node("stakeholder"),
            domain=simple_node("domain"),
            judge=judge_node,
            synthesize=simple_node("synthesize"),
            report=simple_node("report"),
        )
    )
    return graph, calls


def initial_test_state() -> GraphState:
    """라우팅 테스트에 필요한 최소 초기 상태를 만든다."""

    return {
        "retry_count": 0,
        "retry_targets": [],
        "sources": [],
    }


class GraphBuildTests(unittest.TestCase):
    def test_actual_graph_nodes_connects_all_implemented_functions(self) -> None:
        nodes = actual_graph_nodes()

        self.assertIs(nodes.research, research_node)
        self.assertIs(nodes.market, market_node)
        self.assertIs(nodes.stakeholder, stakeholder_node)
        self.assertIs(nodes.domain, domain_node)
        self.assertIs(nodes.judge, judge_node)
        self.assertIs(nodes.synthesize, synthesize_node)
        self.assertIs(nodes.report, report_node)

    def test_default_graph_has_required_actual_node_topology(self) -> None:
        graph = build_graph().get_graph()
        edges = {
            (edge.source, edge.target, edge.conditional) for edge in graph.edges
        }

        self.assertIn(("__start__", "research", False), edges)
        for perspective in PERSPECTIVES:
            self.assertIn(("research", perspective, False), edges)
            self.assertIn((perspective, "judge", False), edges)
            self.assertIn(("prepare_retry", perspective, True), edges)
        self.assertIn(("judge", "prepare_retry", True), edges)
        self.assertIn(("judge", "synthesize", True), edges)
        self.assertIn(("synthesize", "report", False), edges)
        self.assertIn(("report", "__end__", False), edges)

    def test_all_perspectives_run_in_parallel_then_judge_runs_once(self) -> None:
        graph, calls = make_test_graph([set()])

        result = graph.invoke(initial_test_state())

        self.assertEqual(calls["research"], 1)
        self.assertEqual(calls["market"], 1)
        self.assertEqual(calls["stakeholder"], 1)
        self.assertEqual(calls["domain"], 1)
        self.assertEqual(calls["judge"], 1)
        self.assertEqual(calls["synthesize"], 1)
        self.assertEqual(calls["report"], 1)
        self.assertEqual(result["retry_count"], 0)

    def test_only_failed_perspective_runs_again(self) -> None:
        graph, calls = make_test_graph([{"market"}, set()])

        with patch("kvprism.graph.build.log.warning") as warning:
            result = graph.invoke(initial_test_state())

        self.assertEqual(calls["market"], 2)
        self.assertEqual(calls["stakeholder"], 1)
        self.assertEqual(calls["domain"], 1)
        self.assertEqual(calls["judge"], 2)
        self.assertEqual(calls["synthesize"], 1)
        self.assertEqual(calls["report"], 1)
        self.assertEqual(result["retry_count"], 1)
        self.assertEqual(result["retry_targets"], [])
        warning.assert_not_called()

    def test_graph_stops_retrying_after_one_retry(self) -> None:
        graph, calls = make_test_graph(
            [
                {"market", "stakeholder", "domain"},
                {"market"},
            ]
        )

        with patch("kvprism.graph.build.log.warning") as warning:
            result = graph.invoke(initial_test_state())

        self.assertEqual(calls["market"], 2)
        self.assertEqual(calls["stakeholder"], 2)
        self.assertEqual(calls["domain"], 2)
        self.assertEqual(calls["judge"], 2)
        self.assertEqual(calls["synthesize"], 1)
        self.assertEqual(calls["report"], 1)
        self.assertEqual(result["retry_count"], 1)
        self.assertEqual(result["retry_targets"], ["market"])
        warning.assert_called_once()
        self.assertIn("market: 긍정·부정 근거가 부족합니다", warning.call_args.args[1])


if __name__ == "__main__":
    unittest.main()
