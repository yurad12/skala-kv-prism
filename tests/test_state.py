"""팀 전체가 공유하는 그래프 상태 계약 테스트."""

import operator
import unittest
from typing import Annotated, get_args, get_origin, get_type_hints

from pydantic import ValidationError

from kvprism.graph.state import (
    GraphState,
    JudgeResult,
    PerspectiveJudgment,
    PipelineInput,
    Source,
    Technology,
    initial_state,
)


def make_request() -> PipelineInput:
    return PipelineInput(
        domain="데이터센터 · 클라우드 LLM 서빙",
        scenario="장문맥 요청이 섞인 멀티테넌트 환경",
        technologies=[
            Technology(
                technology_id="turboquant",
                name="TurboQuant",
                kind="software",
                paper_doc_id="TQ",
                paper_title="TurboQuant",
            ),
            Technology(
                technology_id="itme",
                name="ITME",
                kind="hardware",
                paper_doc_id="ITME",
                paper_title="Inference Tiered Memory Expansion",
            ),
        ],
    )


class StateContractTests(unittest.TestCase):
    def test_initial_state_contains_only_graph_entry_fields(self) -> None:
        state = initial_state(make_request())
        self.assertEqual(
            set(state), {"request", "retry_count", "retry_targets", "sources"}
        )
        self.assertEqual(state["retry_count"], 0)

    def test_input_requires_one_software_and_one_hardware_technology(self) -> None:
        request = make_request().model_dump()
        request["technologies"][1]["kind"] = "software"
        with self.assertRaisesRegex(ValidationError, "SW와 HW 기술"):
            PipelineInput.model_validate(request)

    def test_source_rejects_empty_excerpt(self) -> None:
        with self.assertRaises(ValidationError):
            Source(
                source_id="TQ-p7",
                source_kind="paper",
                title="TurboQuant",
                author_or_org="Zandieh et al.",
                url_or_page="[TQ p.7]",
                excerpt="",
            )

    def test_failed_judgment_requires_actionable_retry_instruction(self) -> None:
        with self.assertRaisesRegex(ValidationError, "retry_instruction"):
            PerspectiveJudgment(
                perspective="market",
                passed=False,
                issues=["negative evidence is missing"],
            )

    def test_judge_exposes_retry_targets(self) -> None:
        result = JudgeResult(
            judgments=[
                PerspectiveJudgment(perspective="market", passed=True),
                PerspectiveJudgment(
                    perspective="stakeholder",
                    passed=False,
                    issues=["citation missing"],
                    retry_instruction="Add a cited counterargument.",
                ),
                PerspectiveJudgment(perspective="domain", passed=True),
            ]
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.failed_perspectives, ["stakeholder"])

    def test_graph_state_has_eleven_keys_and_top_level_source_reducer(self) -> None:
        hints = get_type_hints(GraphState, include_extras=True)
        self.assertEqual(len(hints), 11)
        self.assertEqual(get_origin(hints["sources"]), Annotated)
        self.assertIn(operator.add, get_args(hints["sources"]))


if __name__ == "__main__":
    unittest.main()
