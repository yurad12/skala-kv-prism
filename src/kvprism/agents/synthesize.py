"""
@desc   : 평가 종합 노드. TRL 추정과 관점 x 기술 매트릭스, 상충 지점 도출
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from ..graph.state import EvidenceClaim, GraphState, Source, SynthesisResult, TRLEstimate
from .report_rules import CITATION, check_forbidden, cited_sources

PROMPT = Path(__file__).resolve().parents[1] / "prompts" / "synthesize.md"
PERSPECTIVES = ("market", "stakeholder", "domain")
# 두 노드가 프롬프트에 넣는 State 키
INPUT_KEYS = ("request", "research", "market_eval", "stakeholder_eval", "domain_eval", "judge")
# 3.3의 상태 어휘
ADOPTED_WORDS = ("병합", "릴리스", "적용")
PENDING_WORDS = ("제안", "않", "못", "예정", "계획", "미정")


def synthesize_node(state: GraphState, *, llm=None) -> dict:
    """담당 키 synthesis만 반환."""
    context = {key: state[key].model_dump(mode="json") for key in INPUT_KEYS}
    context["sources"] = [source.model_dump(mode="json") for source in state["sources"]]
    if llm is None:
        load_dotenv()
        # 설계서 2.4의 Generator 설정. 추론 모델이라 temperature 미지정
        llm = ChatOpenAI(model=os.getenv("GENERATOR_MODEL") or "gpt-5.6-luna", reasoning_effort="medium")
    response = llm.with_structured_output(SynthesisResult).invoke([
        ("system", PROMPT.read_text(encoding="utf-8")),
        ("human", json.dumps(context, ensure_ascii=False)),
    ])
    result = SynthesisResult.model_validate(response)
    return {"synthesis": validate_synthesis(result, state)}


def validate_synthesis(result: SynthesisResult, state: GraphState) -> SynthesisResult:
    """3.3 TRL 근거성과 3.4 중립성 검사. 입력 불변."""
    result = result.model_copy(deep=True)
    sources = {source.source_id: source for source in state["sources"]}
    technology_ids = {tech.technology_id for tech in state["request"].technologies}

    if {item.technology_id for item in result.trl_estimates} != technology_ids:
        raise ValueError("TRL 추정 대상이 선정 기술과 다릅니다")
    expected = {(tech_id, perspective) for tech_id in technology_ids for perspective in PERSPECTIVES}
    if {(cell.technology_id, cell.perspective) for cell in result.matrix} != expected:
        raise ValueError("매트릭스는 기술 2건 × 관점 3개를 중복 없이 채워야 합니다")
    for item in [*result.trl_estimates, *result.matrix, *result.conflicts]:
        unknown = sorted(set(item.source_ids) - sources.keys())
        if unknown:
            raise ValueError(f"등록되지 않은 출처를 인용했습니다: {unknown}")
    for conflict in result.conflicts:
        if len(set(conflict.source_ids)) < 2:
            raise ValueError("상충 지점에는 서로 다른 출처가 2건 이상 필요합니다")

    for estimate in result.trl_estimates:
        _limit_trl(estimate, state, sources)

    texts = [
        result.neutral_summary,
        *(item.rationale for item in result.trl_estimates),
        *(item.assessment for item in result.matrix),
        *(f"{item.title}\n{item.description}" for item in result.conflicts),
    ]
    for text in texts:
        check_forbidden(text, state["sources"])
        cited_sources(text, state["sources"])
    return result


def _limit_trl(estimate: TRLEstimate, state: GraphState, sources: dict[str, Source]) -> None:
    """rationale의 근거 범위로 논문 기술·기반 기술 단계 제한."""
    signals = _trl_signals(state, estimate.technology_id)
    unlinked = sorted(set(estimate.source_ids) - {signal.source_id for signal in signals})
    if unlinked:
        raise ValueError(f"TRL 근거는 해당 기술의 trl_signals에 있어야 합니다: {unlinked}")

    parts = _split_rationale(estimate.rationale)
    for part, field in zip(parts, ("paper_trl", "enabling_technology_trl")):
        cited = set(CITATION.findall(part)) & set(estimate.source_ids)
        ceiling = _ceiling([signal for signal in signals if signal.source_id in cited], sources)
        if getattr(estimate, field) > ceiling:
            setattr(estimate, field, ceiling)


def _split_rationale(rationale: str) -> tuple[str, str]:
    """'논문 기술:'과 '기반 기술:' 구간 분리. 한 줄로 이어 써도 동작."""
    paper = rationale.find("논문 기술:")
    enabling = rationale.find("기반 기술:")
    if paper < 0 or enabling < 0:
        raise ValueError("rationale에는 '논문 기술:'과 '기반 기술:' 근거가 각각 필요합니다")
    if paper < enabling:
        return rationale[paper:enabling], rationale[enabling:]
    return rationale[paper:], rationale[enabling:paper]


def _trl_signals(state: GraphState, technology_id: str) -> list[EvidenceClaim]:
    """기술 조사·시장 평가가 남긴 해당 기술의 TRL 신호."""
    items = [*state["research"].technologies, *state["market_eval"].evaluations]
    return [
        signal
        for item in items
        if item.technology_id == technology_id
        for signal in item.trl_signals
    ]


def _ceiling(signals: list[EvidenceClaim], sources: dict[str, Source]) -> int:
    """근거별 TRL 상한. 논문만 3, 논문 밖 4, 채택 상태 확인 시 제한 없음."""
    outside = [
        signal.statement for signal in signals if sources[signal.source_id].source_kind == "web"
    ]
    if not outside:
        return 3
    return 9 if any(_adopted(statement) for statement in outside) else 4


def _adopted(statement: str) -> bool:
    return any(word in statement for word in ADOPTED_WORDS) and not any(
        word in statement for word in PENDING_WORDS
    )
