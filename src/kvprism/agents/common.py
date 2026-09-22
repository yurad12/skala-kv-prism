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

load_dotenv(override=True)


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
    query_fn: Any,
    extra_sources: list[Source] | None = None,
    use_domain_context: bool = False,
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

    # 전달받은 외부 출처(domain 노드의 RAG 청크 등)가 있으면 먼저 등록
    if extra_sources:
        for s in extra_sources:
            sources_by_id[s.source_id] = s
            
    tech_contexts: list[str] = []

    for tech in req.technologies:
        for q in query_fn(tech.name, domain, scenario):
            for s in web_search(query=q, max_results=2):
                sources_by_id[s.source_id] = s

        sources_text = "\n".join(
            f"- [{s.source_id}] ({s.author_or_org}) {s.title}: {s.excerpt}"
            for s in sources_by_id.values()
        )
        tech_contexts.append(
            f"### 기술: {tech.name} (ID: {tech.technology_id}, 구분: {tech.kind})\n"
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