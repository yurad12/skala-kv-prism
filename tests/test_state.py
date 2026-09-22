"""팀 전체가 공유하는 그래프 상태 계약 테스트."""

import unittest
from typing import Annotated, get_args, get_origin, get_type_hints

from pydantic import ValidationError

try:
    from langgraph.graph import END, START, StateGraph
except ModuleNotFoundError:
    END = START = StateGraph = None

from kvprism.graph.state import (
    ConflictPoint,
    EvidenceClaim,
    GraphState,
    JudgeChecks,
    JudgeResult,
    MatrixCell,
    PerspectiveJudgment,
    PerspectiveResult,
    PipelineInput,
    ReportResult,
    ResearchResult,
    Source,
    SynthesisResult,
    Technology,
    TechnologyEvaluation,
    TechnologyResearch,
    TRLEstimate,
    initial_state,
    merge_sources,
    validate_graph_state,
    validate_pipeline_output,
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


def passing_checks() -> JudgeChecks:
    return JudgeChecks(
        evidence_balance_ok=True,
        citations_ok=True,
        trl_basis_ok=True,
    )


def paper_source() -> Source:
    return Source(
        source_id="TQ-p7",
        source_kind="paper",
        title="TurboQuant",
        author_or_org="Zandieh et al.",
        url_or_page="[TQ p.7]",
        excerpt="TurboQuant reduces the KV cache footprint.",
        doc_id="TQ",
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
                doc_id="TQ",
            )

    def test_failed_judgment_requires_actionable_retry_instruction(self) -> None:
        with self.assertRaisesRegex(ValidationError, "retry_instruction"):
            PerspectiveJudgment(
                perspective="market",
                passed=False,
                checks=JudgeChecks(
                    evidence_balance_ok=False,
                    citations_ok=True,
                    trl_basis_ok=True,
                ),
                issues=["negative evidence is missing"],
            )

    def test_judge_exposes_retry_targets(self) -> None:
        result = JudgeResult(
            judgments=[
                PerspectiveJudgment(
                    perspective="market", passed=True, checks=passing_checks()
                ),
                PerspectiveJudgment(
                    perspective="stakeholder",
                    passed=False,
                    checks=JudgeChecks(
                        evidence_balance_ok=True,
                        citations_ok=False,
                        trl_basis_ok=True,
                    ),
                    issues=["citation missing"],
                    retry_instruction="Add a cited counterargument.",
                ),
                PerspectiveJudgment(
                    perspective="domain", passed=True, checks=passing_checks()
                ),
            ]
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.failed_perspectives, ["stakeholder"])

    def test_judge_exposes_failed_issues_as_warnings(self) -> None:
        result = JudgeResult(
            judgments=[
                PerspectiveJudgment(
                    perspective="market",
                    passed=True,
                    checks=passing_checks(),
                ),
                PerspectiveJudgment(
                    perspective="stakeholder",
                    passed=False,
                    checks=JudgeChecks(
                        evidence_balance_ok=False,
                        citations_ok=True,
                        trl_basis_ok=True,
                    ),
                    issues=["긍정 근거가 부족합니다", "부정 근거가 부족합니다"],
                    retry_instruction="양쪽 근거를 보완하세요.",
                ),
                PerspectiveJudgment(
                    perspective="domain",
                    passed=True,
                    checks=passing_checks(),
                ),
            ]
        )

        self.assertEqual(
            result.warnings,
            [
                "stakeholder: 긍정 근거가 부족합니다",
                "stakeholder: 부정 근거가 부족합니다",
            ],
        )
        self.assertNotIn("warnings", result.model_dump())

    def test_graph_state_has_eleven_keys_and_top_level_source_reducer(self) -> None:
        hints = get_type_hints(GraphState, include_extras=True)
        self.assertEqual(len(hints), 11)
        self.assertEqual(get_origin(hints["sources"]), Annotated)
        self.assertIn(merge_sources, get_args(hints["sources"]))

    def test_source_reducer_removes_exact_duplicates(self) -> None:
        source = paper_source()
        self.assertEqual(merge_sources([source], [source]), [source])

    def test_source_reducer_rejects_conflicting_duplicate_ids(self) -> None:
        first = Source(
            source_id="TQ-p7",
            source_kind="paper",
            title="TurboQuant",
            author_or_org="Zandieh et al.",
            url_or_page="[TQ p.7]",
            excerpt="첫 번째 원문",
            doc_id="TQ",
        )
        second = first.model_copy(update={"excerpt": "서로 다른 원문"})
        with self.assertRaisesRegex(ValueError, "서로 다른 출처 내용"):
            merge_sources([first], [second])

    def test_judge_passed_must_match_individual_checks(self) -> None:
        with self.assertRaisesRegex(ValidationError, "checks의 전체 통과"):
            PerspectiveJudgment(
                perspective="market",
                passed=False,
                checks=passing_checks(),
                issues=["임의 실패"],
                retry_instruction="다시 조사합니다.",
            )

    @unittest.skipIf(StateGraph is None, "LangGraph가 설치된 환경에서 실행합니다")
    def test_langgraph_parallel_reducer_removes_duplicate_sources(self) -> None:
        source = paper_source()

        def first_node(_: GraphState) -> dict:
            return {"sources": [source]}

        def second_node(_: GraphState) -> dict:
            return {"sources": [source]}

        builder = StateGraph(GraphState)
        builder.add_node("first", first_node)
        builder.add_node("second", second_node)
        builder.add_edge(START, "first")
        builder.add_edge(START, "second")
        builder.add_edge("first", END)
        builder.add_edge("second", END)

        result = builder.compile().invoke(initial_state(make_request()))
        self.assertEqual(result["sources"], [source])

    def test_graph_validation_rejects_unregistered_source_id(self) -> None:
        source = paper_source()
        claim = EvidenceClaim(
            statement="KV 캐시 크기를 줄인다.",
            source_id="등록되지-않은-출처",
        )
        research = ResearchResult(
            technologies=[
                TechnologyResearch(
                    technology_id=technology_id,
                    overview="기술 개요",
                    mechanism=[claim],
                    performance_metrics=[claim],
                    experimental_conditions=[claim],
                    limitations=[claim],
                    trl_signals=[claim],
                )
                for technology_id in ("turboquant", "itme")
            ]
        )
        state = initial_state(make_request())
        state.update({"research": research, "sources": [source]})
        with self.assertRaisesRegex(ValueError, "등록되지 않은 source_id"):
            validate_graph_state(state)

    def test_synthesis_requires_all_six_matrix_cells(self) -> None:
        estimates = [
            TRLEstimate(
                technology_id=technology_id,
                paper_trl=4,
                enabling_technology_trl=5,
                rationale="공개된 프로토타입을 기준으로 추정",
                source_ids=["source-1"],
            )
            for technology_id in ("turboquant", "itme")
        ]
        duplicate_cells = [
            MatrixCell(
                technology_id="turboquant",
                perspective="market",
                assessment="평가",
                source_ids=["source-1"],
            )
            for _ in range(6)
        ]
        conflicts = [
            ConflictPoint(
                title=f"상충 {index}",
                description="관점별 해석이 다름",
                source_ids=["source-1", "source-2"],
            )
            for index in (1, 2)
        ]
        with self.assertRaisesRegex(ValidationError, "두 기술 × 세 관점"):
            SynthesisResult(
                trl_estimates=estimates,
                matrix=duplicate_cells,
                conflicts=conflicts,
                neutral_summary="두 기술의 맞바꿈을 중립적으로 정리함",
            )

    def test_complete_pipeline_state_passes_final_validation(self) -> None:
        sources = [
            paper_source(),
            Source(
                source_id="ITME-p3",
                source_kind="paper",
                title="ITME",
                author_or_org="Jang et al.",
                url_or_page="[ITME p.3]",
                excerpt="ITME expands memory with a CXL-hybrid memory tier.",
                doc_id="ITME",
            ),
        ]
        claims = {
            "turboquant": EvidenceClaim(
                statement="KV 캐시 크기를 줄인다.",
                source_id="TQ-p7",
            ),
            "itme": EvidenceClaim(
                statement="CXL 계층으로 메모리를 확장한다.",
                source_id="ITME-p3",
            ),
        }
        research = ResearchResult(
            technologies=[
                TechnologyResearch(
                    technology_id=technology_id,
                    overview="기술 개요",
                    mechanism=[claim],
                    performance_metrics=[claim],
                    experimental_conditions=[claim],
                    limitations=[claim],
                    trl_signals=[claim],
                )
                for technology_id, claim in claims.items()
            ]
        )

        perspective_results = {}
        for perspective in ("market", "stakeholder", "domain"):
            perspective_results[perspective] = PerspectiveResult(
                perspective=perspective,
                evaluations=[
                    TechnologyEvaluation(
                        technology_id=technology_id,
                        summary="근거를 바탕으로 작성한 평가",
                        positive_evidence=[
                            claim.model_copy(update={"stance": "positive"})
                        ],
                        negative_evidence=[
                            claim.model_copy(update={"stance": "negative"})
                        ],
                        trl_signals=[claim],
                    )
                    for technology_id, claim in claims.items()
                ],
            )

        judge = JudgeResult(
            judgments=[
                PerspectiveJudgment(
                    perspective=perspective,
                    passed=True,
                    checks=passing_checks(),
                )
                for perspective in ("market", "stakeholder", "domain")
            ]
        )
        synthesis = SynthesisResult(
            trl_estimates=[
                TRLEstimate(
                    technology_id=technology_id,
                    paper_trl=4,
                    enabling_technology_trl=5,
                    rationale="공개 자료 기반 추정",
                    source_ids=[claim.source_id],
                )
                for technology_id, claim in claims.items()
            ],
            matrix=[
                MatrixCell(
                    technology_id=technology_id,
                    perspective=perspective,
                    assessment="관점별 평가",
                    source_ids=[claim.source_id],
                )
                for technology_id, claim in claims.items()
                for perspective in ("market", "stakeholder", "domain")
            ],
            conflicts=[
                ConflictPoint(
                    title=f"상충 {index}",
                    description="압축과 확장의 맞바꿈",
                    source_ids=[source.source_id for source in sources],
                )
                for index in (1, 2)
            ],
            neutral_summary="두 접근의 맞바꿈을 중립적으로 정리함",
        )
        state = initial_state(make_request())
        state.update(
            {
                "research": research,
                "market_eval": perspective_results["market"],
                "stakeholder_eval": perspective_results["stakeholder"],
                "domain_eval": perspective_results["domain"],
                "judge": judge,
                "synthesis": synthesis,
                "report": ReportResult(
                    markdown_path="outputs/report.md",
                    pdf_path="outputs/report.pdf",
                    reference_source_ids=[source.source_id for source in sources],
                ),
                "sources": sources,
            }
        )
        output = validate_pipeline_output(state)
        self.assertEqual(len(output.sources), 2)


if __name__ == "__main__":
    unittest.main()
