"""조사 결과를 읽고 다음 작업을 선택하는 Supervisor."""

import json
import os
from pathlib import Path

from langchain_openai import ChatOpenAI

from ..graph.supervisor_state import (
    SupervisorDecision, SupervisorState,
    report_digest, validate_success_payload,
)

PROMPT = Path(__file__).resolve().parents[1] / "prompts/supervisor.md"


# 실행 가능한 행동: 관점 간 순서는 지정하지 않는다.
def available_actions(state: SupervisorState) -> list[str]:
    def remaining(node: str) -> bool:
        return state["attempts"][node] < state["limits"].max_worker_attempts[node]

    actions = ["abort"]
    report, quality = state.get("report"), state.get("quality")
    if report is not None and quality is None:
        return actions + (["report_quality"] if remaining("report_quality") else [])
    if quality is not None and quality.passed and report is not None:
        return actions + ["finish"]
    if remaining("research"):
        actions.append("research")
    if state.get("research") is not None:
        actions.extend(node for node in ("market", "stakeholder", "domain") if remaining(node))
    if all(state.get(key) is not None for key in ("research", "market_eval", "stakeholder_eval", "domain_eval")):
        if remaining("synthesize"):
            actions.append("synthesize")
    if state.get("synthesis") is not None and remaining("report"):
        actions.append("report")
    return actions


def supervisor_context(state: SupervisorState, actions: list[str]) -> str:
    """판단에 필요한 결과와 실제 인용 원문을 전달한다. 기록 이력은 넣지 않는다."""
    results = {key: state[key].model_dump(mode="json") for key in (
        "research", "market_eval", "stakeholder_eval", "domain_eval", "synthesis", "quality",
    ) if state.get(key) is not None}
    serialized = json.dumps(results, ensure_ascii=False)
    sources = [s.model_dump(mode="json") for s in state["sources"] if s.source_id in serialized]
    context = json.dumps({
        "request": state["request"].model_dump(mode="json"),
        "available_actions": actions, "results": results, "sources": sources,
        "attempts": state["attempts"], "limits": state["limits"].model_dump(),
        "node_status": state["node_status"], "last_error": state["last_error"],
        "last_decision": state["decision"].model_dump() if state.get("decision") else None,
        "has_report": state.get("report") is not None,
        "invalidation": {"research": "세 관점과 모든 산출물", "perspective": "종합·보고서·품질"},
    }, ensure_ascii=False)
    if len(context) > 180_000:
        raise ValueError("Supervisor 입력이 180,000자를 초과했습니다")
    return context


def supervisor_node(state: SupervisorState, *, llm=None) -> dict:
    """구조화된 LLM 결정을 검증하고, 허용하지 않는 선택은 재판단한다."""
    if state["step_count"] >= state["limits"].max_decisions:
        return {"route": "end", "status": "incomplete", "termination_reason": "Supervisor 판단 상한 도달"}
    # 평가 이후 파일 변경은 기존 통과 판정을 무효화한다.
    freshness = {}
    if state.get("report") is not None and state.get("quality") is not None:
        try:
            current = report_digest(state["report"])
        except OSError:
            current = None
        if current != state["quality"].report_digest:
            freshness = {"quality": None, "last_error": "평가 이후 보고서 파일 변경 또는 누락",
                         "node_status": {**state["node_status"], "report_quality": "stale"}}
            state = {**state, **freshness}
    update = {**freshness, "step_count": state["step_count"] + 1}
    try:
        actions = available_actions(state)
        if actions == ["abort"]:
            return {**update, "route": "end", "status": "failed" if state.get("last_error") else "incomplete",
                    "termination_reason": "실행 가능한 작업의 예산이 없습니다"}
        if llm is None:
            llm = ChatOpenAI(
                model=os.getenv("SUPERVISOR_MODEL") or os.getenv("GENERATOR_MODEL") or "gpt-5.6-luna",
                timeout=120, max_retries=1,
            )
        decision = SupervisorDecision.model_validate(llm.with_structured_output(SupervisorDecision).invoke([
            ("system", PROMPT.read_text(encoding="utf-8")),
            ("human", supervisor_context(state, actions)),
        ]))
        if decision.action not in actions:
            raise ValueError(f"허용되지 않은 행동: {decision.action}; 허용 목록: {actions}")
        if decision.evidence_status == "sufficient" and decision.gaps:
            raise ValueError("근거 충분 판정과 미해결 gaps가 모순됩니다")
        if decision.action in ("synthesize", "report") and decision.evidence_status != "sufficient":
            raise ValueError("종합·보고서에는 Supervisor의 근거 충분 판정이 필요합니다")
        update.update(decision=decision, invalid_decisions=0, last_error=None)
        if decision.action == "finish":
            validate_success_payload(state)
            return {**update, "route": "end", "status": "succeeded", "termination_reason": decision.reason}
        if decision.action == "abort":
            return {**update, "route": "end", "status": "failed" if state.get("last_error") else "incomplete",
                    "termination_reason": decision.reason}
        return {**update, "route": decision.action}
    except Exception as error:
        # 원문 API 예외에는 인증 정보가 섞일 수 있어 유형과 계약 오류만 남긴다.
        message = str(error)[:1000] if type(error) is ValueError else type(error).__name__
        invalid = state["invalid_decisions"] + 1
        failed = invalid >= state["limits"].max_invalid_decisions
        return {**update, "decision": None, "invalid_decisions": invalid,
                "last_error": message, "route": "end" if failed else "supervisor",
                "status": "failed" if failed else "running",
                "termination_reason": "Supervisor 판단 오류 상한: " + message if failed else None}
