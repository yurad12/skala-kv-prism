"""Judge 노드의 근거 품질 검사 테스트."""

from datetime import date

import pytest

from kvprism.agents.judge import judge_node
from kvprism.graph.state import (
    EvidenceClaim,
    Perspective,
    PerspectiveResult,
    Source,
    TechnologyEvaluation,
)


PERSPECTIVES: tuple[Perspective, ...] = ("market", "stakeholder", "domain")


def make_source(source_id: str, excerpt: str = "TurboQuant reduces KV cache memory.") -> Source:
    """테스트용 웹 출처를 만든다."""

    return Source(
        source_id=source_id,
        source_kind="web",
        title="검증용 출처",
        author_or_org="테스트 기관",
        published_at=date(2026, 1, 1),
        url_or_page="https://example.com/source",
        excerpt=excerpt,
    )


def make_evaluation(technology_id: str, source_id: str) -> TechnologyEvaluation:
    """모든 Judge 검사 항목을 통과하는 기술 평가를 만든다."""

    return TechnologyEvaluation(
        technology_id=technology_id,
        summary=f"{technology_id} 평가",
        positive_evidence=[
            EvidenceClaim(
                statement="KV cache 메모리를 줄입니다.",
                source_id=source_id,
                stance="positive",
            )
        ],
        negative_evidence=[
            EvidenceClaim(
                statement="적용 조건에 제한이 있습니다.",
                source_id=source_id,
                stance="negative",
            )
        ],
        trl_signals=[
            EvidenceClaim(
                statement="공개 구현이 릴리스되었습니다.",
                source_id=source_id,
            )
        ],
    )


def make_state() -> dict:
    """세 관점과 등록 출처가 모두 있는 Judge 입력을 만든다."""

    sources = [make_source("source-tq"), make_source("source-itme")]
    state = {"sources": sources}
    for perspective in PERSPECTIVES:
        state[f"{perspective}_eval"] = PerspectiveResult(
            perspective=perspective,
            evaluations=[
                make_evaluation("turboquant", "source-tq"),
                make_evaluation("itme", "source-itme"),
            ],
        )
    return state


def test_judge_passes_three_perspectives_with_balanced_supported_evidence() -> None:
    state = make_state()

    output = judge_node(state, support_checker=lambda claim, source: True)

    assert set(output) == {"judge", "retry_targets"}
    assert output["judge"].passed is True
    assert output["retry_targets"] == []
    assert all(judgment.issues == [] for judgment in output["judge"].judgments)


def test_judge_returns_only_failed_perspective_as_retry_target() -> None:
    state = make_state()
    market = state["market_eval"]
    market.evaluations[0].negative_evidence = []
    market.evaluations[1].trl_signals = []

    output = judge_node(state, support_checker=lambda claim, source: True)

    assert output["retry_targets"] == ["market"]
    judgment = output["judge"].judgments[0]
    assert judgment.checks.evidence_balance_ok is False
    assert judgment.checks.citations_ok is True
    assert judgment.checks.trl_basis_ok is False
    assert "turboquant" in judgment.retry_instruction
    assert "itme" in judgment.retry_instruction


def test_judge_rejects_unknown_and_unsupported_citations() -> None:
    state = make_state()
    market = state["market_eval"]
    market.evaluations[0].positive_evidence[0].source_id = "unknown"

    def supports(claim: EvidenceClaim, source: Source) -> bool:
        return claim.statement != "적용 조건에 제한이 있습니다."

    output = judge_node(state, support_checker=supports)

    assert output["retry_targets"] == ["market", "stakeholder", "domain"]
    market_judgment = output["judge"].judgments[0]
    assert market_judgment.checks.citations_ok is False
    assert "미등록 출처" in market_judgment.issues[0]
    assert "발췌문 근거 불충분" in market_judgment.issues[0]


def test_judge_checks_only_presence_of_trl_signals() -> None:
    state = make_state()

    def supports(claim: EvidenceClaim, source: Source) -> bool:
        return "릴리스" not in claim.statement

    output = judge_node(state, support_checker=supports)

    assert output["judge"].passed is True
    assert output["retry_targets"] == []


def test_judge_requires_all_three_perspective_inputs() -> None:
    state = make_state()
    del state["domain_eval"]

    with pytest.raises(ValueError, match="domain_eval"):
        judge_node(state, support_checker=lambda claim, source: True)


def test_judge_rejects_perspective_result_stored_under_wrong_key() -> None:
    state = make_state()
    state["market_eval"] = state["domain_eval"]

    with pytest.raises(ValueError, match="market_eval의 perspective는 market"):
        judge_node(state, support_checker=lambda claim, source: True)
