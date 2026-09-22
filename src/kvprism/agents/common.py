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
    use_domain_context: bool = False,
    research_fields: tuple[str, ...] = (),
) -> dict:
    req = state["request"]
    domain = req.domain if use_domain_context else ""
    scenario = req.scenario if use_domain_context else ""

    # 1. 재실행 지침 추출
    retry_note = ""
    if perspective in state.get("retry_targets", []) and state.get("judge"):
        retry_note = next(
            (f"\n[보완 지침]: {j.retry_instruction}" for j in state["judge"].judgments if j.perspective == perspective and not j.passed),
            "",
        )

    # 2. 웹 검색 및 출처 수집 (딕셔너리로 source_id 중복 즉시 차단)
    sources_by_id: dict[str, Source] = {}
    tech_contexts: list[str] = []

    for tech in req.technologies:
        research_text = _format_research(state, tech.technology_id, research_fields)
        tech_source_ids: list[str] = []
        for q in query_fn(tech.name, domain, scenario):
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

    model_name = os.getenv("OPENAI_MODEL_NAME") or os.getenv("GENERATOR_MODEL") or "gpt-4o-mini"
    llm = ChatOpenAI(model=model_name, temperature=0).with_structured_output(_RawPerspectiveResult)
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
        for ev in raw.evaluations if ev.technology_id in target_ids
    ]

    # 2개 기술 누락 방지 fallback
    done_ids = {e.technology_id for e in evaluations}
    for t in req.technologies:
        if t.technology_id not in done_ids:
            evaluations.append(
                TechnologyEvaluation(
                    technology_id=t.technology_id,
                    summary=f"{t.name} {perspective} 관점 요약",
                )
            )

    return {
        f"{perspective}_eval": PerspectiveResult(perspective=perspective, evaluations=evaluations[:2]),
        "sources": list(sources_by_id.values()),
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
