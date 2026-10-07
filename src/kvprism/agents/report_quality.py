"""최신 보고서를 코드 검사와 독립적인 LLM 판단으로 평가한다."""

import json
import os
from pathlib import Path

import pymupdf
from langchain_openai import ChatOpenAI
from pydantic import Field

from ..graph.state import StrictModel
from ..graph.supervisor_state import (
    QualityChecks, QualityIssue, ReportQualityResult, report_digest,
)
from .report_rules import CITATION
from .report_rules import (
    REPORT_METHODOLOGY, TRL_GUIDE, check_forbidden, validate_report_artifact,
)

PROMPT = Path(__file__).resolve().parents[1] / "prompts/report_quality.md"


class SemanticQuality(StrictModel):
    """해시와 페이지 수는 LLM에 맡기지 않는다."""
    checks: QualityChecks
    issues: list[QualityIssue] = Field(max_length=20)


def report_quality_node(state, *, llm=None) -> dict:
    """품질 판정만 반환한다. 재작업 노드 선택은 Supervisor가 수행한다."""
    report, sources = state["report"], state["sources"]
    original = report.source_markdown_path.read_text(encoding="utf-8")
    body = original.split("# REFERENCE", 1)[0]
    with pymupdf.open(report.pdf_path) as pdf:
        page_count = len(pdf)
        pdf_text = "\n".join(page.get_text() for page in pdf)
    digest = report_digest(report)
    code_issues = []

    def issue(criterion, reason):
        code_issues.append(QualityIssue(
            criterion=criterion, location="보고서 전체", reason=reason,
            source_ids=[], suggested_action="report", instruction=reason,
        ))

    try:
        validate_report_artifact(report, sources)
    except ValueError as error:
        issue("groundedness", str(error))
    try:
        check_forbidden(body, sources)
    except ValueError as error:
        issue("neutrality", str(error))
    if page_count > 10 or not page_count or "# SUMMARY" not in body or "# REFERENCE" not in original:
        issue("format", "SUMMARY·REFERENCE를 포함한 PDF를 1~10페이지로 작성하세요")
    labels = ("기술 성숙도", "시장", "이해관계자", "도메인")
    if any(label not in body for label in labels):
        issue("perspective_coverage", "성숙도·시장·이해관계자·도메인을 모두 설명하세요")
    display = report.markdown_path.read_text(encoding="utf-8")
    reference_tags = CITATION.findall(display.split("# REFERENCE", 1)[-1])
    web_tags = list(dict.fromkeys(tag for tag in reference_tags if tag.startswith("web_")))
    # PDF의 번호 인용과 원본 ID를 함께 제공해 같은 주장을 대조한다.
    citation_labels = {str(index): source_id for index, source_id in enumerate(web_tags, start=1)}
    synthesis_context = state["synthesis"].model_dump(mode="json")
    for estimate in synthesis_context["trl_estimates"]:
        paper_basis, separator, enabling_basis = estimate["rationale"].partition("기반 기술:")
        for field, basis in (("paper_trl", paper_basis),
                             ("enabling_technology_trl", enabling_basis if separator else "판정 불가")):
            if any(term in basis for term in ("판정 불가", "스키마", "호환용 최소", "schema")):
                # 독자에게 표시하지 않은 내부 값은 평가할 TRL 점수가 아니다.
                estimate[field] = None
    context = json.dumps({
        "request": state["request"].model_dump(mode="json"),
        "report": body, "pdf_text": pdf_text,
        "methodology": REPORT_METHODOLOGY, "trl_criteria": TRL_GUIDE,
        "pdf_citation_labels": citation_labels,
        "synthesis": synthesis_context,
        "sources": [s.model_dump(mode="json") for s in sources if s.source_id in body],
    }, ensure_ascii=False)
    if len(context) > 180_000:
        raise ValueError("품질 평가 입력이 180,000자를 초과했습니다")
    if llm is None:
        llm = ChatOpenAI(model=os.getenv("JUDGE_MODEL") or "gpt-5.6-luna", timeout=120, max_retries=1)
    messages = [("system", PROMPT.read_text(encoding="utf-8")), ("human", context)]
    model = llm.with_structured_output(SemanticQuality)
    for attempt in range(2):
        try:
            semantic = SemanticQuality.model_validate(model.invoke(messages))
            # 구조화 출력의 실패 사유 일관성도 검증한다.
            ReportQualityResult(checks=semantic.checks, issues=semantic.issues,
                                report_digest=digest, page_count=1)
            break
        except ValueError:
            if attempt:
                raise
            messages.append(("human", "미달 항목마다 사유를 제공하고 통과 항목의 사유는 제거하세요."))
    checks = semantic.checks.model_copy()
    checks.format = not any(i.criterion == "format" for i in code_issues)
    semantic_issues = [i for i in semantic.issues if i.criterion != "format"]
    for found in code_issues:
        setattr(checks, found.criterion, False)
    issues = semantic_issues + code_issues
    # 목록 상한 안에서 각 실패 항목의 사유를 반드시 보존한다.
    mandatory = {i.criterion: i for i in issues}
    selected = list(mandatory.values())
    selected.extend(i for i in issues if i not in selected)
    return {"quality": ReportQualityResult(
        checks=checks, issues=selected[:20], report_digest=digest, page_count=page_count,
    )}
