"""관점 평가 노드 공통 실행 엔진."""

from __future__ import annotations

import os
from typing import Callable
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from ..graph.state import (
    EvidenceClaim,
    GraphState,
    Perspective,
    PerspectiveResult,
    Source,
    Stance,
    TechnologyEvaluation,
    merge_sources,
)
from ..tools.web_search import web_search

load_dotenv()


class _RawClaim(BaseModel):
    statement: str
    source_id: str
    stance: Stance = "neutral"


class _RawEval(BaseModel):
    technology_id: str
    summary: str
    positive_evidence: list[_RawClaim] = Field(default_factory=list)
    negative_evidence: list[_RawClaim] = Field(default_factory=list)
    neutral_evidence: list[_RawClaim] = Field(default_factory=list)
    trl_signals: list[_RawClaim] = Field(default_factory=list)


class _RawPerspectiveResult(BaseModel):
    evaluations: list[_RawEval]


def _make_claims(claims: list[_RawClaim], stance: Stance | None = None) -> list[EvidenceClaim]:
    """StrictModel stance 검증자를 통과하도록 정규화합니다."""
    return [
        EvidenceClaim(statement=c.statement, source_id=c.source_id, stance=stance or c.stance)
        for c in claims
    ]


def evaluate_perspective_node(
    state: GraphState,
    perspective: Perspective,
    system_prompt: str,
    query_fn: Callable[[str, str, str], list[str]],
    extra_sources: list[Source] | None = None,
    use_domain_context: bool = False,
    research_fields: tuple[str, ...] = (),
) -> dict:
    req = state["request"]
    domain = req.domain if use_domain_context else ""
    scenario = req.scenario if use_domain_context else ""

    # Supervisor의 지시와 자기 이전 결과만 재작업에 사용한다.
    decision = state.get("decision")
    instruction = decision.instruction if decision else ""
    previous = state.get(f"{perspective}_eval")
    retry_note = f"\n[Supervisor 작업 지시]: {instruction}" if instruction else ""
    if previous is not None:
        retry_note += "\n[이전 결과 - 타당한 근거는 유지]: " + previous.model_dump_json()

    # 기존 근거와 새 논문 청크를 함께 읽되 동일 ID의 내용 충돌은 허용하지 않는다.
    sources_by_id = {
        s.source_id: s for s in merge_sources(state.get("sources", []), extra_sources)
    }

    tech_contexts: list[str] = []

    for tech in req.technologies:
        research_text = _format_research(state, tech.technology_id, research_fields)
        # 논문 원문은 이 노드가 직접 찾은 청크(도메인 RAG)만 넣는다. 기술 조사 내용은 research_text로 이미 전달되며,
        # 기술 조사의 청크까지 모두 넣으면 웹 출처 대신 논문만 인용하게 된다. 재작업이면 이전 관점 근거를 함께 전달한다.
        tech_source_ids = [s.source_id for s in (extra_sources or []) if s.doc_id == tech.paper_doc_id]
        if previous is not None:
            old = next(e for e in previous.evaluations if e.technology_id == tech.technology_id)
            tech_source_ids.extend(c.source_id for c in (
                old.positive_evidence + old.negative_evidence + old.neutral_evidence + old.trl_signals
            ) if c.source_id in sources_by_id and c.source_id not in tech_source_ids)
        queries = query_fn(tech.name, domain, scenario)
        if previous is not None and instruction:
            queries = [f"{tech.name} {instruction}"]
        for q in queries:
            for s in web_search(query=q, max_results=2):
                sources_by_id = {
                    item.source_id: item
                    for item in merge_sources(list(sources_by_id.values()), [s])
                }
                if s.source_id not in tech_source_ids:
                    tech_source_ids.append(s.source_id)

        sources_text = "\n".join(
            f"- [{s.source_id}] ({s.author_or_org}) {s.title}: {s.excerpt}"
            for source_id in tech_source_ids
            for s in [sources_by_id[source_id]]
        )
        tech_contexts.append(
            f"### 기술: {tech.name} (ID: {tech.technology_id}, 구분: {tech.kind})\n"
            f"논문 기술 조사:\n{research_text}\n"
            f"참조 출처:\n{sources_text if sources_text else '(수집된 웹 출처 없음)'}"
        )

    domain_header = f"도메인: {domain}\n시나리오: {scenario}\n\n" if use_domain_context else ""
    user_content = (
        f"{domain_header}{chr(10).join(tech_contexts)}{retry_note}\n\n"
        f"위 2개 기술({[t.technology_id for t in req.technologies]}) 각각의 평가를 작성하세요."
    )

    messages = [
        SystemMessage(content=system_prompt + "\nSupervisor의 작업 지시를 반영하고 이전 결과의 타당한 근거는 유지하세요. "
                      "기존 주장도 원문 발췌로 다시 확인하며, 지지가 없으면 수정하세요. "
                      "source_id에는 제공된 참조 출처만 사용하고 근거가 없는 사실은 만들지 마세요. "
                      "관점 판단(채택·입장·적용 조건)의 근거는 참조 출처를 우선 인용하고, "
                      "기술 조사 항목의 인용은 기술 자체를 설명할 때만 쓰세요."),
        HumanMessage(content=user_content),
    ]

    model_name = os.getenv("GENERATOR_MODEL") or "gpt-5.6-luna"
    llm = ChatOpenAI(model=model_name, timeout=120, max_retries=1).with_structured_output(_RawPerspectiveResult)
    raw: _RawPerspectiveResult = llm.invoke(messages)

    target_ids = {t.technology_id for t in req.technologies}
    evaluations: list[TechnologyEvaluation] = [
        TechnologyEvaluation(
            technology_id=ev.technology_id,
            summary=ev.summary,
            positive_evidence=_make_claims(ev.positive_evidence, stance="positive"),
            negative_evidence=_make_claims(ev.negative_evidence, stance="negative"),
            neutral_evidence=_make_claims(ev.neutral_evidence, stance="neutral"),
            trl_signals=_make_claims(ev.trl_signals),
        )
        for ev in raw.evaluations
    ]

    if len(evaluations) != 2 or {e.technology_id for e in evaluations} != target_ids:
        raise ValueError("관점 결과에는 요청한 두 기술이 정확히 한 번씩 필요합니다")
    if any(not (e.positive_evidence or e.negative_evidence or e.neutral_evidence) for e in evaluations):
        raise ValueError("근거 없는 빈 관점 결과를 성공으로 반환하지 않습니다")
    for evaluation in evaluations:
        claims = (evaluation.positive_evidence + evaluation.negative_evidence
                  + evaluation.neutral_evidence + evaluation.trl_signals)
        if any(claim.source_id not in sources_by_id for claim in claims):
            raise ValueError("관점 결과가 참조 출처에 없는 source_id를 사용했습니다")
    merged = merge_sources(state.get("sources", []), list(sources_by_id.values()))
    old_ids = {s.source_id for s in state.get("sources", [])}
    return {
        f"{perspective}_eval": PerspectiveResult(perspective=perspective, evaluations=evaluations),
        "sources": [s for s in merged if s.source_id not in old_ids],
    }


def _format_research(
    state: GraphState,
    technology_id: str,
    fields: tuple[str, ...],
) -> str:
    """관점 평가에 필요한 기술 조사 항목만 인용 식별자와 함께 정리한다."""

    research = state.get("research")
    if research is None:
        raise ValueError("관점별 평가에 research 결과가 필요합니다")
    profile = next(
        (item for item in research.technologies if item.technology_id == technology_id),
        None,
    )
    if profile is None:
        raise ValueError(f"research에 기술 조사 결과가 없습니다: {technology_id}")

    lines: list[str] = []
    for field in fields:
        value = getattr(profile, field)
        if isinstance(value, str):
            lines.append(f"- {field}: {value}")
            continue
        lines.extend(
            f"- {field}: {claim.statement} [{claim.source_id}]" for claim in value
        )
    return "\n".join(lines) or "(사용할 기술 조사 항목 없음)"
