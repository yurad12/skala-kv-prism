"""종합·보고서 노드의 반환 계약과 최종 출력 검증 테스트."""

from datetime import date
from pathlib import Path
from types import SimpleNamespace

from kvprism.agents.report import report_node
from kvprism.agents.synthesize import synthesize_node
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
    TRLEstimate,
    Technology,
    TechnologyEvaluation,
    TechnologyResearch,
    validate_pipeline_output,
)


TECHNOLOGIES = ("turboquant", "itme")
PERSPECTIVES = ("market", "stakeholder", "domain")


class StructuredLLM:
    """synthesize_node가 요구하는 구조화 출력만 반환한다."""

    def __init__(self, result: SynthesisResult):
        self.result = result

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        return self.result


class ReportLLM:
    """각 보고서 장에 등록된 인용이 포함된 본문을 반환한다."""

    def invoke(self, messages):
        return SimpleNamespace(content="공개 자료 기반의 중립적 분석입니다 [paper-turboquant]")


def make_source(technology_id: str, kind: str) -> Source:
    source_id = f"{kind}-{technology_id}"
    if kind == "paper":
        return Source(
            source_id=source_id,
            source_kind="paper",
            title=f"{technology_id} 논문",
            author_or_org="연구팀",
            published_at=date(2026, 1, 1),
            url_or_page=f"[{technology_id} p.1]",
            excerpt="프로토타입 실험 결과",
            doc_id=technology_id,
        )
    return Source(
        source_id=source_id,
        source_kind="web",
        title=f"{technology_id} 릴리스",
        author_or_org="프로젝트 팀",
        published_at=date(2026, 1, 2),
        url_or_page=f"https://example.com/{technology_id}",
        excerpt="공개 구현이 릴리스되었습니다.",
    )


def make_claim(statement: str, source_id: str, stance: str = "neutral") -> EvidenceClaim:
    return EvidenceClaim(statement=statement, source_id=source_id, stance=stance)


def make_state() -> GraphState:
    technologies = [
        Technology(
            technology_id="turboquant",
            name="TurboQuant",
            kind="software",
            paper_doc_id="turboquant",
            paper_title="TurboQuant",
        ),
        Technology(
            technology_id="itme",
            name="ITME",
            kind="hardware",
            paper_doc_id="itme",
            paper_title="ITME",
        ),
    ]
    research = ResearchResult(
        technologies=[
            TechnologyResearch(
                technology_id=technology_id,
                overview=f"{technology_id} 개요",
                performance_metrics=[make_claim("성능 측정", f"paper-{technology_id}")],
                experimental_conditions=[make_claim("실험 조건", f"paper-{technology_id}")],
                limitations=[make_claim("기술 한계", f"paper-{technology_id}")],
                trl_signals=[make_claim("프로토타입 검증", f"paper-{technology_id}")],
            )
            for technology_id in TECHNOLOGIES
        ]
    )
    evaluations = {
        perspective: PerspectiveResult(
            perspective=perspective,
            evaluations=[
                TechnologyEvaluation(
                    technology_id=technology_id,
                    summary=f"{technology_id} {perspective} 평가",
                    positive_evidence=[
                        make_claim("도입 효과", f"web-{technology_id}", "positive")
                    ],
                    negative_evidence=[
                        make_claim("도입 제약", f"web-{technology_id}", "negative")
                    ],
                    trl_signals=[
                        make_claim("공개 구현 릴리스", f"web-{technology_id}")
                    ],
                )
                for technology_id in TECHNOLOGIES
            ],
        )
        for perspective in PERSPECTIVES
    }
    judge = JudgeResult(
        judgments=[
            PerspectiveJudgment(
                perspective=perspective,
                passed=True,
                checks=JudgeChecks(
                    evidence_balance_ok=True,
                    citations_ok=True,
                    trl_basis_ok=True,
                ),
            )
            for perspective in PERSPECTIVES
        ]
    )
    return {
        "request": PipelineInput(
            domain="데이터센터",
            scenario="장문맥 추론",
            technologies=technologies,
        ),
        "research": research,
        "market_eval": evaluations["market"],
        "stakeholder_eval": evaluations["stakeholder"],
        "domain_eval": evaluations["domain"],
        "judge": judge,
        "retry_count": 0,
        "retry_targets": [],
        "sources": [
            make_source(technology_id, kind)
            for technology_id in TECHNOLOGIES
            for kind in ("paper", "web")
        ],
    }


def make_synthesis() -> SynthesisResult:
    return SynthesisResult(
        trl_estimates=[
            TRLEstimate(
                technology_id=technology_id,
                paper_trl=3,
                enabling_technology_trl=7,
                rationale=(
                    f"논문 기술: 프로토타입 근거 [paper-{technology_id}] "
                    f"기반 기술: 공개 구현 릴리스 [web-{technology_id}]"
                ),
                source_ids=[f"paper-{technology_id}", f"web-{technology_id}"],
            )
            for technology_id in TECHNOLOGIES
        ],
        matrix=[
            MatrixCell(
                technology_id=technology_id,
                perspective=perspective,
                assessment=f"조건별 평가 [web-{technology_id}]",
                source_ids=[f"web-{technology_id}"],
            )
            for technology_id in TECHNOLOGIES
            for perspective in PERSPECTIVES
        ],
        conflicts=[
            ConflictPoint(
                title=f"상충 조건 {number}",
                description="논문 조건과 공개 구현 환경의 차이",
                source_ids=["paper-turboquant", "web-itme"],
            )
            for number in (1, 2)
        ],
        neutral_summary="두 기술은 적용 조건이 다릅니다 [paper-turboquant]",
    )


def test_synthesize_returns_only_validated_synthesis() -> None:
    state = make_state()

    output = synthesize_node(state, llm=StructuredLLM(make_synthesis()))

    assert set(output) == {"synthesis"}
    assert isinstance(output["synthesis"], SynthesisResult)
    assert len(output["synthesis"].matrix) == 6


def test_report_returns_report_result_and_pipeline_output_is_valid(
    tmp_path: Path, monkeypatch
) -> None:
    state = make_state()
    state["synthesis"] = make_synthesis()

    def fake_render_pdf(markdown: str, output_path: Path) -> Path:
        output_path.write_bytes(b"%PDF-1.4 test")
        return output_path

    monkeypatch.setattr("kvprism.agents.report.render_pdf", fake_render_pdf)
    output = report_node(state, llm=ReportLLM(), output_dir=tmp_path)
    state.update(output)

    assert set(output) == {"report"}
    assert isinstance(output["report"], ReportResult)
    assert output["report"].markdown_path.exists()
    assert output["report"].pdf_path.exists()
    final = validate_pipeline_output(state)
    assert final.report == output["report"]
    assert final.synthesis == state["synthesis"]
