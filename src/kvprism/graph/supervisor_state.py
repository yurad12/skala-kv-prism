"""Supervisor의 공용 데이터 계약과 초기화·최종 검증."""

from pathlib import Path
from typing import Annotated, Literal, TypedDict
from pydantic import Field, model_validator
from kvprism.graph.state import (
    StrictModel, PipelineInput, ResearchResult, PerspectiveResult,
    SynthesisResult, ReportResult, PipelineOutput, Source, merge_sources,
)


# 공통 타입
WorkerName = Literal[
    "research", "market", "stakeholder", "domain",
    "synthesize", "report", "report_quality",
]
Action = Literal[
    "research", "market", "stakeholder", "domain",
    "synthesize", "report", "report_quality", "finish", "abort",
]
Route = Literal[
    "research", "market", "stakeholder", "domain",
    "synthesize", "report", "report_quality", "supervisor", "end",
]
RunStatus = Literal["running", "succeeded", "incomplete", "failed"]
NodeStatus = Literal["not_run", "succeeded", "failed", "stale"]
Criterion = Literal[
    "groundedness", "neutrality", "bias_control", "perspective_coverage", "format",
]


# Supervisor의 최신 판단

class SupervisorDecision(StrictModel):
    action: Action
    reason: str = Field(min_length=1, max_length=2000)
    instruction: str = Field(min_length=1, max_length=3000)
    evidence_status: Literal["insufficient", "sufficient", "unknown"]
    gaps: list[str] = Field(max_length=12)


# 보고서 파일: 기존 ReportResult를 확장하며 기존 세 필드는 유지한다.

class ReportArtifact(ReportResult):
    source_markdown_path: Path


# 품질 평가: 항목별 판정과 재작업 사유

class QualityIssue(StrictModel):
    criterion: Criterion
    location: str = Field(min_length=1, max_length=300)
    reason: str = Field(min_length=1, max_length=1500)
    source_ids: list[str] = Field(max_length=20)
    suggested_action: WorkerName
    instruction: str = Field(min_length=1, max_length=2000)


class QualityChecks(StrictModel):
    groundedness: bool
    neutrality: bool
    bias_control: bool
    perspective_coverage: bool
    format: bool


class ReportQualityResult(StrictModel):
    checks: QualityChecks
    issues: list[QualityIssue] = Field(max_length=20)
    report_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    page_count: int = Field(ge=1)

    @property
    def passed(self) -> bool:
        return all(self.checks.model_dump().values())

    @model_validator(mode="after")
    def require_issues_for_failed_checks(self):
        failed = {k for k, v in self.checks.model_dump().items() if not v}
        if failed != {issue.criterion for issue in self.issues}:
            raise ValueError("각 미달 항목에는 사유가 있어야 하며 통과 항목에 실패 사유를 쓰지 않습니다")
        if self.page_count > 10 and self.checks.format:
            raise ValueError("10페이지를 초과한 보고서는 형식 통과로 판정할 수 없습니다")
        return self


# 실행 설정: 모든 횟수는 최초 실행을 포함한다.

class RunLimits(StrictModel):
    max_decisions: int = Field(default=24, ge=1, le=100)
    max_invalid_decisions: int = Field(default=2, ge=1, le=5)
    max_worker_attempts: dict[WorkerName, int] = Field(default_factory=lambda: {
        "research": 2, "market": 2, "stakeholder": 2, "domain": 2,
        "synthesize": 3, "report": 3, "report_quality": 3,
    })

    @model_validator(mode="after")
    def require_all_worker_limits(self):
        expected = {"research", "market", "stakeholder", "domain", "synthesize", "report", "report_quality"}
        if set(self.max_worker_attempts) != expected:
            raise ValueError("모든 하위 노드의 시도 상한을 지정해야 합니다")
        if any(value < 1 or value > 5 for value in self.max_worker_attempts.values()):
            raise ValueError("하위 노드 시도 상한은 1~5입니다")
        return self


# 제어 메타데이터와 작업 결과를 한 State 안에서 구분한다.

class SupervisorState(TypedDict):
    # 작업 페이로드
    request: PipelineInput
    research: ResearchResult | None
    market_eval: PerspectiveResult | None
    stakeholder_eval: PerspectiveResult | None
    domain_eval: PerspectiveResult | None
    synthesis: SynthesisResult | None
    report: ReportArtifact | None
    sources: Annotated[list[Source], merge_sources]
    quality: ReportQualityResult | None
    # 제어 메타데이터
    decision: SupervisorDecision | None
    route: Route
    trace_id: str
    limits: RunLimits
    step_count: int
    invalid_decisions: int
    attempts: dict[WorkerName, int]
    node_status: dict[WorkerName, NodeStatus]
    last_error: str | None
    status: RunStatus
    termination_reason: str | None


# CLI와 호출자에게 반환할 결과

class SupervisorOutput(StrictModel):
    status: Literal["succeeded", "incomplete", "failed"]
    trace_id: str
    termination_reason: str = Field(min_length=1)
    output: PipelineOutput | None = None
    quality: ReportQualityResult | None = None

    @model_validator(mode="after")
    def require_valid_success(self):
        if self.status == "succeeded":
            if self.output is None or self.quality is None or not self.quality.passed:
                raise ValueError("성공에는 검증된 출력과 통과한 품질 결과가 필요합니다")
        elif self.output is not None:
            raise ValueError("미완료·실패 결과를 최종 제출 출력으로 반환하지 않습니다")
        return self



# 공통 노드 이름과 초기 상태
WORKER_KEYS = {
    "research": "research", "market": "market_eval",
    "stakeholder": "stakeholder_eval", "domain": "domain_eval",
    "synthesize": "synthesis", "report": "report", "report_quality": "quality",
}


def initial_supervisor_state(request: PipelineInput, *, trace_id: str,
                             limits: RunLimits | None = None) -> SupervisorState:
    """제어 상태를 초기화한다. 실행 ID는 파일 경로로도 쓰므로 UUID로 제한한다."""
    from uuid import UUID

    trace_id = str(UUID(trace_id))
    return {
        "request": request, **dict.fromkeys(WORKER_KEYS.values()),
        "sources": [], "decision": None, "route": "supervisor",
        "trace_id": trace_id, "limits": limits or RunLimits(),
        "step_count": 0, "invalid_decisions": 0,
        "attempts": dict.fromkeys(WORKER_KEYS, 0),
        "node_status": dict.fromkeys(WORKER_KEYS, "not_run"),
        "last_error": None, "status": "running", "termination_reason": None,
    }


def report_digest(report: ReportArtifact) -> str:
    """세 산출물의 실제 바이트로 품질 판정의 최신성을 확인한다."""
    import hashlib

    return hashlib.sha256(b"\x00".join(path.read_bytes() for path in (
        report.source_markdown_path, report.markdown_path, report.pdf_path,
    ))).hexdigest()


def validate_success_payload(state: SupervisorState) -> PipelineOutput:
    """Judge 필드를 요구하지 않고 현재 결과·인용·실제 보고서를 검증한다."""
    from .state import validate_graph_state
    from ..agents.report_rules import validate_report_artifact

    required = ("research", "market_eval", "stakeholder_eval", "domain_eval",
                "synthesis", "report", "quality")
    missing = [key for key in required if state.get(key) is None]
    if missing:
        raise ValueError(f"최종 결과가 없습니다: {', '.join(missing)}")
    validate_graph_state(state)
    quality = state["quality"]
    report = state["report"]
    if not quality.passed or quality.report_digest != report_digest(report):
        raise ValueError("최신 보고서의 품질 통과가 필요합니다")
    page_count = validate_report_artifact(report, state["sources"])
    if page_count > 10 or page_count != quality.page_count:
        raise ValueError("보고서 페이지 수가 품질 평가와 다르거나 10페이지를 넘습니다")
    return PipelineOutput(report=report, synthesis=state["synthesis"], sources=state["sources"])


def validate_supervisor_output(state: SupervisorState) -> SupervisorOutput:
    """미완료 초안을 정상 제출 출력과 구분한다."""
    status = state.get("status")
    if status not in ("succeeded", "incomplete", "failed"):
        raise ValueError("그래프가 종료 상태를 반환하지 않았습니다")
    return SupervisorOutput(
        status=status, trace_id=state["trace_id"],
        termination_reason=state.get("termination_reason") or "종료 사유 없음",
        output=validate_success_payload(state) if status == "succeeded" else None,
        quality=state.get("quality"),
    )
