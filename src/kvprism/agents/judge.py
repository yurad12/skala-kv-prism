"""세 관점 평가의 근거 품질을 검사하는 Judge 노드."""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from ..graph.state import (
    EvidenceClaim,
    GraphState,
    JudgeChecks,
    JudgeResult,
    Perspective,
    PerspectiveJudgment,
    PerspectiveResult,
    Source,
    TechnologyEvaluation,
)


PERSPECTIVE_KEYS: dict[Perspective, str] = {
    "market": "market_eval",
    "stakeholder": "stakeholder_eval",
    "domain": "domain_eval",
}


class ClaimSupportVerdict(BaseModel):
    """주장이 출처 원문에 의해 직접 뒷받침되는지 판정한 결과."""

    supported: bool
    reason: str = Field(min_length=1)


SupportChecker = Callable[[EvidenceClaim, Source], bool]


def _make_support_checker() -> SupportChecker:
    """한국어 주장과 원문 발췌문의 의미 일치를 판정하는 함수를 만든다."""

    load_dotenv()
    model = os.getenv("JUDGE_MODEL") or os.getenv("GENERATOR_MODEL") or "gpt-5.6-luna"
    llm = ChatOpenAI(model=model, reasoning_effort="low").with_structured_output(
        ClaimSupportVerdict
    )

    def supports(claim: EvidenceClaim, source: Source) -> bool:
        verdict = llm.invoke(
            [
                (
                    "system",
                    "주장이 출처 발췌문에 의해 직접 뒷받침되는지 판정하세요. "
                    "같은 주제라는 이유만으로 통과시키지 말고, 수치·조건·인과관계가 "
                    "발췌문과 일치할 때만 supported=true로 답하세요.",
                ),
                (
                    "human",
                    f"주장: {claim.statement}\n\n출처 발췌문: {source.excerpt}",
                ),
            ]
        )
        return ClaimSupportVerdict.model_validate(verdict).supported

    return supports


def _evidence_claims(evaluation: TechnologyEvaluation) -> list[EvidenceClaim]:
    """출처 원문과 의미 적합성을 검사할 일반 평가 근거를 모은다."""

    return [
        *evaluation.positive_evidence,
        *evaluation.negative_evidence,
        *evaluation.neutral_evidence,
    ]


def _cited_claims(evaluation: TechnologyEvaluation) -> list[EvidenceClaim]:
    """source_id 등록 여부를 검사할 모든 근거를 모은다."""

    return [*_evidence_claims(evaluation), *evaluation.trl_signals]


def _judge_perspective(
    perspective: Perspective,
    result: PerspectiveResult,
    sources: dict[str, Source],
    supports: SupportChecker,
) -> PerspectiveJudgment:
    """관점 하나의 근거 균형, 인용 적합성, TRL 신호 누락을 검사한다."""

    issues: list[str] = []

    unbalanced = [
        evaluation.technology_id
        for evaluation in result.evaluations
        if not evaluation.positive_evidence or not evaluation.negative_evidence
    ]
    evidence_balance_ok = not unbalanced
    if unbalanced:
        issues.append(
            "긍정·부정 근거가 모두 필요합니다: " + ", ".join(unbalanced)
        )

    invalid_citations: list[str] = []
    # 주장별 판정은 서로 독립이라 병렬 호출. 판정 단위와 결과는 순차 호출과 같다
    pending = {
        (claim.statement, claim.source_id): (claim, sources[claim.source_id])
        for evaluation in result.evaluations
        for claim in _evidence_claims(evaluation)
        if claim.source_id in sources
    }
    with ThreadPoolExecutor(max_workers=8) as pool:
        verdicts = pool.map(lambda pair: supports(*pair), pending.values())
        support_cache = dict(zip(pending, verdicts))
    for evaluation in result.evaluations:
        for claim in _cited_claims(evaluation):
            source = sources.get(claim.source_id)
            label = f"{evaluation.technology_id}/{claim.source_id}"
            if source is None:
                invalid_citations.append(f"{label}(미등록 출처)")
        for claim in _evidence_claims(evaluation):
            if claim.source_id not in sources:
                continue
            label = f"{evaluation.technology_id}/{claim.source_id}"
            if not support_cache[(claim.statement, claim.source_id)]:
                invalid_citations.append(f"{label}(발췌문 근거 불충분)")

    citations_ok = not invalid_citations
    if invalid_citations:
        issues.append("인용을 확인해야 합니다: " + ", ".join(invalid_citations))

    # TRL 단계는 synthesize가 추정한다. Judge는 입력 신호의 누락만 확인한다.
    missing_trl = [
        evaluation.technology_id
        for evaluation in result.evaluations
        if not evaluation.trl_signals
    ]
    trl_basis_ok = not missing_trl
    if missing_trl:
        issues.append("TRL 추정 입력 신호가 필요합니다: " + ", ".join(missing_trl))

    checks = JudgeChecks(
        evidence_balance_ok=evidence_balance_ok,
        citations_ok=citations_ok,
        trl_basis_ok=trl_basis_ok,
    )
    passed = checks.passed
    retry_instruction = None
    if not passed:
        retry_instruction = " ".join(issues) + " 등록된 출처의 원문 근거로 보완하세요."

    return PerspectiveJudgment(
        perspective=perspective,
        passed=passed,
        checks=checks,
        issues=issues,
        retry_instruction=retry_instruction,
    )


def judge_node(state: GraphState, *, support_checker: SupportChecker | None = None) -> dict:
    """세 관점 결과를 검증하고 실패 관점의 재실행 대상을 반환한다."""

    missing = [key for key in PERSPECTIVE_KEYS.values() if state.get(key) is None]
    if missing:
        raise ValueError(f"Judge 입력에 관점별 평가가 없습니다: {', '.join(missing)}")

    sources = {source.source_id: source for source in state.get("sources", [])}
    supports = support_checker or _make_support_checker()
    judgments = []
    for perspective, key in PERSPECTIVE_KEYS.items():
        result = state[key]
        if result.perspective != perspective:
            raise ValueError(
                f"{key}의 perspective는 {perspective}여야 합니다: {result.perspective}"
            )
        judgments.append(_judge_perspective(perspective, result, sources, supports))
    result = JudgeResult(judgments=judgments)
    return {
        "judge": result,
        "retry_targets": result.failed_perspectives,
    }
