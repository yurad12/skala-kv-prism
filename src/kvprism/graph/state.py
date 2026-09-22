"""모든 그래프 노드가 공유하는 데이터 계약.

각 노드는 자신이 담당하는 키만 반환해야 한다. 특히 ``sources``에는 노드가
새로 발견한 출처만 반환한다. LangGraph는 :class:`GraphState`에 선언된
리듀서를 사용해 새 출처 목록을 기존 목록에 이어 붙인다.
"""

from __future__ import annotations

import operator
from datetime import date
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator


TechnologyKind = Literal["software", "hardware"]
Perspective = Literal["market", "stakeholder", "domain"]
Stance = Literal["positive", "negative", "neutral"]
SourceKind = Literal["paper", "web"]


# ---------------------------------------------------------------------------
# 공통 기반 타입
# ---------------------------------------------------------------------------


class StrictModel(BaseModel):
    """오탈자나 정의되지 않은 필드 입력을 거부하는 공통 모델."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# ---------------------------------------------------------------------------
# 파이프라인 입력
# ---------------------------------------------------------------------------


class Technology(StrictModel):
    """선정 기술 한 건과 주 근거로 사용할 논문 정보."""

    technology_id: str = Field(
        min_length=1,
        description="변하지 않는 기술 식별자. 예: turboquant",
    )
    name: str = Field(min_length=1)
    kind: TechnologyKind
    paper_doc_id: str = Field(
        min_length=1,
        description="rag_retrieve가 검색 대상으로 받는 문서 식별자",
    )
    paper_title: str = Field(min_length=1)


class PipelineInput(StrictModel):
    """``app.py``가 받아 검증하는 외부 입력."""

    domain: str = Field(min_length=1)
    scenario: str = Field(min_length=1)
    technologies: list[Technology] = Field(min_length=2, max_length=2)
    replay: bool = False

    @model_validator(mode="after")
    def require_one_technology_per_kind(self) -> "PipelineInput":
        ids = [item.technology_id for item in self.technologies]
        if len(set(ids)) != len(ids):
            raise ValueError("technology_id는 서로 달라야 합니다")
        kinds = {item.kind for item in self.technologies}
        if kinds != {"software", "hardware"}:
            raise ValueError("technologies에는 SW와 HW 기술이 각각 하나씩 있어야 합니다")
        return self


# ---------------------------------------------------------------------------
# 출처 및 근거
# ---------------------------------------------------------------------------


class Source(StrictModel):
    """RAG 도구와 웹 도구가 함께 사용하는 표준 인용 정보."""

    source_id: str = Field(min_length=1, description="변하지 않는 인용 식별자")
    source_kind: SourceKind
    title: str = Field(min_length=1)
    author_or_org: str = Field(min_length=1)
    published_at: date | None = None
    url_or_page: str = Field(
        min_length=1,
        description="웹 URL 또는 [TQ p.7] 형식의 논문 페이지 태그",
    )
    excerpt: str = Field(
        min_length=1,
        description="LLM 요약문이 아닌 출처 원문의 발췌문",
    )
    doc_id: str | None = Field(
        default=None,
        description="논문 문서 식별자. 웹 출처일 때는 생략",
    )


class EvidenceClaim(StrictModel):
    """등록된 출처 한 건으로 역추적할 수 있는 주장."""

    statement: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    stance: Stance = "neutral"


# ---------------------------------------------------------------------------
# 기술 조사 결과
# ---------------------------------------------------------------------------


class TechnologyResearch(StrictModel):
    """기술 하나에 대한 기술 조사 노드의 결과."""

    technology_id: str = Field(min_length=1)
    overview: str = Field(min_length=1)
    mechanism: list[EvidenceClaim] = Field(default_factory=list)
    performance: list[EvidenceClaim] = Field(default_factory=list)
    experimental_conditions: list[EvidenceClaim] = Field(default_factory=list)
    limitations: list[EvidenceClaim] = Field(default_factory=list)
    trl_signals: list[EvidenceClaim] = Field(default_factory=list)


class ResearchResult(StrictModel):
    """두 논문을 대상으로 수행한 기술 조사 노드의 전체 결과."""

    technologies: list[TechnologyResearch] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def require_unique_technology_ids(self) -> "ResearchResult":
        ids = [item.technology_id for item in self.technologies]
        if len(set(ids)) != len(ids):
            raise ValueError("기술 조사 결과의 technology_id는 서로 달라야 합니다")
        return self


# ---------------------------------------------------------------------------
# 관점별 평가 결과
# ---------------------------------------------------------------------------


class TechnologyEvaluation(StrictModel):
    """하나의 관점에서 기술 한 건을 양면으로 평가한 결과."""

    technology_id: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    positive_evidence: list[EvidenceClaim] = Field(default_factory=list)
    negative_evidence: list[EvidenceClaim] = Field(default_factory=list)
    neutral_evidence: list[EvidenceClaim] = Field(default_factory=list)
    trl_signals: list[EvidenceClaim] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_evidence_stance_to_match_bucket(self) -> "TechnologyEvaluation":
        for item in self.positive_evidence:
            if item.stance != "positive":
                raise ValueError("positive_evidence 항목의 stance는 positive여야 합니다")
        for item in self.negative_evidence:
            if item.stance != "negative":
                raise ValueError("negative_evidence 항목의 stance는 negative여야 합니다")
        return self


class PerspectiveResult(StrictModel):
    """시장·이해관계자·도메인 노드가 공통으로 따르는 출력 계약."""

    perspective: Perspective
    evaluations: list[TechnologyEvaluation] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def require_unique_technology_ids(self) -> "PerspectiveResult":
        ids = [item.technology_id for item in self.evaluations]
        if len(set(ids)) != len(ids):
            raise ValueError("관점별 평가의 technology_id는 서로 달라야 합니다")
        return self


# ---------------------------------------------------------------------------
# 검증 결과 및 재실행 판단
# ---------------------------------------------------------------------------


class PerspectiveJudgment(StrictModel):
    """관점 하나에 대한 검증 결과와 재실행용 보완 지시."""

    perspective: Perspective
    passed: bool
    issues: list[str] = Field(default_factory=list)
    retry_instruction: str | None = None

    @model_validator(mode="after")
    def require_retry_instruction_on_failure(self) -> "PerspectiveJudgment":
        if not self.passed and not self.retry_instruction:
            raise ValueError("검증 실패 시 retry_instruction이 필요합니다")
        return self


class JudgeResult(StrictModel):
    """병렬 실행된 세 관점을 한 번에 검사한 검증 결과."""

    judgments: list[PerspectiveJudgment] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def require_each_perspective_once(self) -> "JudgeResult":
        perspectives = [item.perspective for item in self.judgments]
        if set(perspectives) != {"market", "stakeholder", "domain"}:
            raise ValueError("검증 결과에는 market, stakeholder, domain이 모두 필요합니다")
        if len(set(perspectives)) != 3:
            raise ValueError("검증 대상 관점은 중복될 수 없습니다")
        return self

    @property
    def passed(self) -> bool:
        return all(item.passed for item in self.judgments)

    @property
    def failed_perspectives(self) -> list[Perspective]:
        return [item.perspective for item in self.judgments if not item.passed]


# ---------------------------------------------------------------------------
# 종합 및 보고서 결과
# ---------------------------------------------------------------------------


class TRLEstimate(StrictModel):
    """논문 기술과 기반 기술을 분리한 TRL 추정 결과."""

    technology_id: str = Field(min_length=1)
    paper_trl: int = Field(ge=1, le=9)
    enabling_technology_trl: int = Field(ge=1, le=9)
    rationale: str = Field(min_length=1)
    source_ids: list[str] = Field(min_length=1)
    disclosure: Literal["공개 정보 기반 추정"] = "공개 정보 기반 추정"


class MatrixCell(StrictModel):
    technology_id: str = Field(min_length=1)
    perspective: Perspective
    assessment: str = Field(min_length=1)
    source_ids: list[str] = Field(min_length=1)


class ConflictPoint(StrictModel):
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    source_ids: list[str] = Field(min_length=2)


class SynthesisResult(StrictModel):
    """보고서 노드가 입력으로 사용하는 종합 노드 결과."""

    trl_estimates: list[TRLEstimate] = Field(min_length=2, max_length=2)
    matrix: list[MatrixCell] = Field(min_length=6, max_length=6)
    conflicts: list[ConflictPoint] = Field(min_length=2)
    neutral_summary: str = Field(min_length=1)


class ReportResult(StrictModel):
    """최종 보고서 경로와 보고서에서 사용한 인용 목록."""

    markdown_path: Path
    pdf_path: Path
    reference_source_ids: list[str]


class PipelineOutput(StrictModel):
    """``app.py``가 최종 반환하는 검증된 외부 출력."""

    report: ReportResult
    synthesis: SynthesisResult
    sources: list[Source]

    @model_validator(mode="after")
    def validate_source_registry(self) -> "PipelineOutput":
        source_ids = [source.source_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("sources의 source_id는 서로 달라야 합니다")
        missing = set(self.report.reference_source_ids) - set(source_ids)
        if missing:
            raise ValueError(
                f"보고서가 등록되지 않은 source_id를 참조합니다: {sorted(missing)}"
            )
        return self


# ---------------------------------------------------------------------------
# LangGraph 공유 상태
# ---------------------------------------------------------------------------


class GraphState(TypedDict, total=False):
    """LangGraph 파이프라인에서 사용하는 11개 공유 상태 키.

    그래프 실행 중에는 모든 키가 선택 사항이다. 각 노드는 자신이 담당하는
    키만 기록한다. ``sources``만 병렬 노드가 함께 쓰며 리듀서로 누적한다.
    """

    request: PipelineInput
    research: ResearchResult
    market_eval: PerspectiveResult
    stakeholder_eval: PerspectiveResult
    domain_eval: PerspectiveResult
    judge: JudgeResult
    retry_count: Literal[0, 1]
    retry_targets: list[Perspective]
    synthesis: SynthesisResult
    report: ReportResult
    sources: Annotated[list[Source], operator.add]


def initial_state(request: PipelineInput) -> GraphState:
    """신규 실행과 재현 실행에 공통으로 사용할 초기 상태를 생성한다."""

    return {
        "request": request,
        "retry_count": 0,
        "retry_targets": [],
        "sources": [],
    }


def validate_pipeline_output(state: GraphState) -> PipelineOutput:
    """그래프 종료 상태에 필수 출력이 모두 있는지 검사한다."""

    missing = [key for key in ("report", "synthesis", "sources") if key not in state]
    if missing:
        raise ValueError(f"그래프 출력에 필수 필드가 없습니다: {', '.join(missing)}")
    return PipelineOutput(
        report=state["report"],
        synthesis=state["synthesis"],
        sources=state["sources"],
    )
