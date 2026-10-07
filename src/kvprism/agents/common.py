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

    sources_by_id: dict[str, Source] = {s.source_id: s for s in state.get("sources", [])}

    # 전달받은 외부 출처(domain 노드의 RAG 청크 등)가 있으면 먼저 등록
    if extra_sources:
        for s in extra_sources:
            sources_by_id[s.source_id] = s

    tech_contexts: list[str] = []

    for tech in req.technologies:
        research_text = _format_research(state, tech.technology_id, research_fields)
        # 이 기술의 논문 RAG 청크(extra_sources)를 먼저 넣어 LLM 이 인용할 수 있게 한다
        tech_source_ids = [s.source_id for s in sources_by_id.values() if s.doc_id == tech.paper_doc_id]
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
                sources_by_id[s.source_id] = s
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

    # 3. LLM 메세지 조립
    domain_header = f"도메인: {domain}\n시나리오: {scenario}\n\n" if use_domain_context else ""
    user_content = (
        f"{domain_header}{chr(10).join(tech_contexts)}{retry_note}\n\n"
        f"위 2개 기술({[t.technology_id for t in req.technologies]}) 각각의 평가를 작성하세요."
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]

    model_name = os.getenv("GENERATOR_MODEL") or "gpt-5.6-luna"
    llm = ChatOpenAI(model=model_name, timeout=120, max_retries=1).with_structured_output(_RawPerspectiveResult)
    raw: _RawPerspectiveResult = llm.invoke(messages)

    # 4. StrictModel 규격 변환 및 stance 검증 통과 보장
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
