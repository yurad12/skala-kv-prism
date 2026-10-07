"""TRL 추정, 관점별 매트릭스와 상충 지점을 작성하는 종합 노드."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from ..graph.state import EvidenceClaim, GraphState, Source, SynthesisResult, TRLEstimate
from .report_rules import CITATION, check_forbidden, cited_sources

PROMPT = Path(__file__).resolve().parents[1] / "prompts" / "synthesize.md"
PERSPECTIVES = ("market", "stakeholder", "domain")
INPUT_KEYS = ("request", "research", "market_eval", "stakeholder_eval", "domain_eval")


def synthesize_node(state: GraphState, *, llm=None) -> dict:
    """담당 키 synthesis만 반환. 검사에 걸리면 한 번 다시 요청."""
    context = {key: state[key].model_dump(mode="json") for key in INPUT_KEYS}
    context["previous_synthesis"] = state["synthesis"].model_dump(mode="json") if state.get("synthesis") else None
    context["instruction"] = state["decision"].instruction if state.get("decision") else ""
    context["sources"] = [source.model_dump(mode="json") for source in state["sources"]]
    context["source_ids"] = [source.source_id for source in state["sources"]]
    # TRL 근거로 쓸 수 있는 ID. 기술별 trl_signals에 기록된 것만 인정한다
    context["trl_source_ids"] = {
        tech.technology_id: sorted({signal.source_id for signal in _trl_signals(state, tech.technology_id)})
        for tech in state["request"].technologies
    }
    if llm is None:
        load_dotenv()
        # 설계서 2.4의 Generator 설정. 추론 모델이라 temperature 미지정
        llm = ChatOpenAI(model=os.getenv("GENERATOR_MODEL") or "gpt-5.6-luna", reasoning_effort="medium", timeout=120, max_retries=1)

    model = llm.with_structured_output(SynthesisResult)
    messages = [
        ("system", PROMPT.read_text(encoding="utf-8")),
        ("human", json.dumps(context, ensure_ascii=False)),
    ]
    for retried in (False, True):
        result = SynthesisResult.model_validate(model.invoke(messages))
        try:
            return {"synthesis": validate_synthesis(result, state)}
        except ValueError as error:
            if retried:
                raise
            messages.append(
                ("human", f"직전 출력의 문제: {error}. source_ids 목록의 값을 글자 그대로 옮겨 다시 작성한다.")
            )


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
            raise ValueError(
                f"{estimate.technology_id}의 {field}={getattr(estimate, field)}는 "
                f"인용한 근거 범위의 상한 {ceiling}을 넘습니다. 점수와 rationale을 함께 수정하세요"
            )


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
    """해당 기술의 검증 환경과 채택 상태를 판단할 수 있는 근거."""
    signals = []
    for item in state["research"].technologies:
        if item.technology_id == technology_id:
            # 실험 환경도 성숙도 판단의 직접 근거이므로 별도 태그에만 의존하지 않는다.
            signals.extend(item.trl_signals)
            signals.extend(item.experimental_conditions)
    for item in state["market_eval"].evaluations:
        if item.technology_id == technology_id:
            signals.extend(item.trl_signals)
    return signals


def _ceiling(signals: list[EvidenceClaim], sources: dict[str, Source]) -> int:
    """논문만으로는 3까지 허용한다. 외부 근거의 실제 단계는 품질 평가에서 확인한다."""
    outside = [
        signal.statement for signal in signals if sources[signal.source_id].source_kind == "web"
    ]
    if not outside:
        return 3
    return 9
