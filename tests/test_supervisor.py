"""실제 Supervisor·wrapper·품질 노드·SQLite 복구를 외부 API 없이 검증한다."""

import sqlite3
from collections import Counter
from dataclasses import replace
from uuid import uuid4

import pymupdf
import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

import app
from kvprism.agents.supervisor import available_actions, supervisor_node
from kvprism.agents.report_quality import SemanticQuality, report_quality_node
from kvprism.agents.report_rules import build_source_references, display_report
from kvprism.graph.supervisor_build import SupervisorNodes, build_supervisor_graph, worker_wrapper
from kvprism.graph.supervisor_state import (
    QualityChecks, QualityIssue, ReportArtifact, ReportQualityResult, RunLimits,
    SupervisorDecision, initial_supervisor_state, report_digest, validate_supervisor_output,
)
from test_output_nodes import make_state, make_synthesis


class ScriptedSupervisor:
    """모델 응답만 바꾸며 실제 Supervisor의 결정 검증은 유지한다."""
    def __init__(self, actions):
        self.actions = iter(actions)
        self.contexts = []

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        self.contexts.append(messages)
        action = next(self.actions)
        return SupervisorDecision(
            action=action, reason=f"{action}에 필요한 근거 확인",
            instruction="기존 근거를 유지하고 부족한 실험 조건을 보완하세요",
            evidence_status="sufficient" if action in ("synthesize", "report", "finish") else "insufficient",
            gaps=[],
        )


def artifact(directory, sources, pages=1):
    """실제 파일 해시와 인용 검증에 사용할 작은 PDF를 만든다."""
    directory.mkdir(parents=True, exist_ok=True)
    ids = [s.source_id for s in sources]
    body = "# SUMMARY\n\n공개 자료 분석\n\n# 4. 관점별 평가\n\n"
    body += "## 기술 성숙도\n## 시장성\n## 이해관계자\n## 도메인 적용\n"
    body += " ".join(f"[{sid}]" for sid in ids) + "\n\n"
    original = body + build_source_references(sources)
    raw, md, pdf = directory / "report.source.md", directory / "report.md", directory / "report.pdf"
    raw.write_text(original, encoding="utf-8")
    md.write_text(display_report(original, sources), encoding="utf-8")
    with pymupdf.open() as document:
        for _ in range(pages):
            page = document.new_page()
            page.insert_text((50, 50), "Evidence report for supervisor contract test")
        document.save(pdf)
    return ReportArtifact(source_markdown_path=raw, markdown_path=md, pdf_path=pdf, reference_source_ids=ids)


def quality(report, passed=True):
    return ReportQualityResult(
        checks=QualityChecks(groundedness=passed, neutrality=True, bias_control=True,
                             perspective_coverage=True, format=True),
        issues=[] if passed else [QualityIssue(
            criterion="groundedness", location="도메인", reason="실험 조건 근거 부족",
            source_ids=[], suggested_action="domain", instruction="실험 조건을 보완하세요",
        )], report_digest=report_digest(report), page_count=1,
    )


def setup_nodes(tmp_path, actions, verdicts=(True,), calls=None):
    fixtures = make_state()
    calls = calls if calls is not None else Counter()
    llm = ScriptedSupervisor(actions)
    outcomes = iter(verdicts)

    def collect(name, key):
        def node(packet):
            calls[name] += 1
            assert "attempts" not in packet and "judge" not in packet
            assert packet["decision"].action == name
            return {key: fixtures[key], "sources": fixtures["sources"] if name == "research" else []}
        return node

    def synthesize(packet):
        calls["synthesize"] += 1
        return {"synthesis": make_synthesis()}

    def report(packet):
        calls["report"] += 1
        return {"report": artifact(tmp_path / f"report-{calls['report']}", packet["sources"])}

    def evaluate(packet):
        calls["report_quality"] += 1
        return {"quality": quality(packet["report"], next(outcomes))}

    nodes = SupervisorNodes(
        lambda state: supervisor_node(state, llm=llm),
        collect("research", "research"), collect("market", "market_eval"),
        collect("stakeholder", "stakeholder_eval"), collect("domain", "domain_eval"),
        synthesize, report, evaluate,
    )
    state = initial_supervisor_state(fixtures["request"], trace_id=str(uuid4()))
    return nodes, state, calls, llm


NORMAL = ["research", "market", "stakeholder", "domain", "synthesize", "report", "report_quality", "finish"]


@pytest.mark.parametrize("order", [NORMAL, ["research", "domain", "stakeholder", "market", *NORMAL[4:]]])
def test_actual_supervisor_accepts_different_orders(tmp_path, order):
    nodes, state, calls, llm = setup_nodes(tmp_path, order)
    result = build_supervisor_graph(nodes, output_root=tmp_path).invoke(state)
    assert validate_supervisor_output(result).status == "succeeded"
    assert all(count == 1 for count in calls.values())
    assert "프로토타입 실험 결과" in str(llm.contexts[1])


def test_repair_only_failed_view_and_regenerate_outputs(tmp_path):
    actions = NORMAL[:-1] + ["domain", "synthesize", "report", "report_quality", "finish"]
    nodes, state, calls, _ = setup_nodes(tmp_path, actions, (False, True))
    final = build_supervisor_graph(nodes, output_root=tmp_path).invoke(state, {"recursion_limit": 53})
    assert final["status"] == "succeeded"
    assert calls["domain"] == calls["synthesize"] == calls["report"] == calls["report_quality"] == 2
    assert calls["research"] == calls["market"] == calls["stakeholder"] == 1
    assert len(final["sources"]) == 4


def test_subagents_only_return_to_supervisor(tmp_path):
    nodes, _, _, _ = setup_nodes(tmp_path, ["abort"])
    edges = build_supervisor_graph(nodes, output_root=tmp_path).get_graph().edges
    for edge in edges:
        if edge.source not in ("supervisor", "__start__"):
            assert edge.target == "supervisor"


def test_default_graph_uses_actual_nodes():
    from kvprism.graph.supervisor_build import actual_supervisor_nodes
    from kvprism.agents.research import research_node
    nodes = actual_supervisor_nodes()
    assert nodes.research is research_node
    assert nodes.supervisor is supervisor_node
    assert nodes.report_quality is report_quality_node


def test_early_finish_is_rejected_and_bounded(tmp_path):
    nodes, state, calls, _ = setup_nodes(tmp_path, ["finish", "finish"])
    final = build_supervisor_graph(nodes, output_root=tmp_path).invoke(state)
    assert final["status"] == "failed" and not calls
    assert final["invalid_decisions"] == 2


def test_step_limit_is_incomplete(tmp_path):
    nodes, state, _, _ = setup_nodes(tmp_path, NORMAL)
    state["limits"] = RunLimits(max_decisions=1)
    final = build_supervisor_graph(nodes, output_root=tmp_path).invoke(state)
    assert final["status"] == "incomplete"
    assert validate_supervisor_output(final).output is None


def test_worker_cannot_overwrite_control_state(tmp_path):
    nodes, state, _, _ = setup_nodes(tmp_path, ["research", "abort"])
    nodes = replace(nodes, research=lambda packet: {"research": make_state()["research"], "status": "succeeded"})
    final = build_supervisor_graph(nodes, output_root=tmp_path).invoke(state)
    assert final["research"] is None and final["attempts"]["research"] == 1
    assert final["node_status"]["research"] == "failed"
    assert final["status"] == "failed"


def test_invalid_source_does_not_enter_state(tmp_path):
    nodes, state, _, _ = setup_nodes(tmp_path, ["research", "abort"])
    nodes = replace(nodes, research=lambda packet: {"research": make_state()["research"], "sources": []})
    final = build_supervisor_graph(nodes, output_root=tmp_path).invoke(state)
    assert final["research"] is None and not final["sources"]


def test_failed_research_preserves_valid_dependents(tmp_path):
    _, state, _, llm = setup_nodes(tmp_path, ["research"])
    fixture = make_state()
    state.update({key: fixture[key] for key in ("research", "market_eval", "domain_eval", "stakeholder_eval", "sources")})
    state.update(supervisor_node(state, llm=llm))
    def failing(packet):
        raise RuntimeError("외부 호출 실패")
    result = worker_wrapper("research", failing, tmp_path)(state)
    assert "market_eval" not in result and "research" not in result
    assert result["attempts"]["research"] == 1


def test_successful_research_invalidates_dependents(tmp_path):
    nodes, state, _, llm = setup_nodes(tmp_path, ["research"])
    fixture = make_state()
    state.update({key: fixture[key] for key in ("research", "market_eval", "domain_eval", "stakeholder_eval", "sources")})
    state.update(supervisor_node(state, llm=llm))
    result = worker_wrapper("research", nodes.research, tmp_path)(state)
    assert all(result[key] is None for key in ("market_eval", "domain_eval", "stakeholder_eval", "synthesis", "report", "quality"))


def test_changed_files_require_new_quality(tmp_path):
    _, state, _, llm = setup_nodes(tmp_path, ["report_quality"])
    state["report"] = artifact(tmp_path / "draft", make_state()["sources"])
    state["quality"] = quality(state["report"])
    state["report"].markdown_path.write_text("수정된 파일")
    result = supervisor_node(state, llm=llm)
    assert result["quality"] is None and result["route"] == "report_quality"


def test_sqlite_resume_with_new_graph_and_connection(tmp_path):
    nodes, state, calls, _ = setup_nodes(tmp_path, NORMAL)
    config = {"configurable": {"thread_id": state["trace_id"]}}
    database = tmp_path / "checkpoint.sqlite"
    with sqlite3.connect(database, check_same_thread=False) as connection:
        graph = build_supervisor_graph(nodes, checkpointer=SqliteSaver(connection, serde=app.checkpoint_serializer()), output_root=tmp_path)
        graph.invoke(state, config, interrupt_before=["market"])
        assert graph.get_state(config).next == ("market",)
    # 연결·그래프·모델 인스턴스를 새로 만든다. 진행 상태는 SQLite에서만 복원한다.
    resumed_nodes, _, _, _ = setup_nodes(tmp_path, NORMAL[2:], calls=calls)
    with sqlite3.connect(database, check_same_thread=False) as connection:
        graph = build_supervisor_graph(resumed_nodes, checkpointer=SqliteSaver(connection, serde=app.checkpoint_serializer()), output_root=tmp_path)
        loaded = graph.get_state(config).values
        assert isinstance(loaded["limits"], RunLimits)
        assert loaded["attempts"]["research"] == 1
        output = app.run_pipeline(graph=graph, resume=True, trace_id=state["trace_id"], output_root=tmp_path)
        assert output.status == "succeeded"
        assert calls["research"] == 1
        with pytest.raises(ValueError, match="미완료"):
            app.run_pipeline(graph=graph, resume=True, trace_id=state["trace_id"], output_root=tmp_path)


class PassingJudge:
    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        return SemanticQuality(checks=QualityChecks(
            groundedness=True, neutrality=True, bias_control=True,
            perspective_coverage=True, format=True,
        ), issues=[])


@pytest.mark.parametrize("pages,expected", [(1, True), (11, False)])
def test_quality_code_checks_override_llm(tmp_path, pages, expected):
    state = make_state()
    state["synthesis"] = make_synthesis()
    state["report"] = artifact(tmp_path, state["sources"], pages=pages)
    result = report_quality_node(state, llm=PassingJudge())["quality"]
    assert result.passed is expected
    assert result.page_count == pages


def test_quality_rejects_display_content_changes(tmp_path):
    state = make_state()
    state["synthesis"] = make_synthesis()
    state["report"] = artifact(tmp_path, state["sources"])
    state["report"].markdown_path.write_text("출처와 다른 내용")
    result = report_quality_node(state, llm=PassingJudge())["quality"]
    assert not result.checks.groundedness
    assert result.issues[0].suggested_action == "report"


def test_all_actual_nodes_generate_pdf_offline(tmp_path, monkeypatch):
    """외부 응답만 고정하고 실제 8개 노드·SQLite·PDF 렌더러를 모두 실행한다."""
    import importlib
    from types import SimpleNamespace
    from kvprism.graph.state import EvidenceClaim, TechnologyResearch, SynthesisResult
    from test_research_agent import FakeAsk, FakeRetrieve
    from test_agents import FakeChatOpenAI, make_web_source
    from test_output_nodes import StructuredLLM

    class ResearchAsk(FakeAsk):
        def __call__(self, schema, system, user):
            result = super().__call__(schema, system, user)
            if schema is TechnologyResearch:
                tech = "turboquant" if "turboquant" in result.overview else "itme"
                result.trl_signals = [EvidenceClaim(statement="실험실 프로토타입 검증", source_id=f"{tech}:p1:0")]
            return result

    retrieve = FakeRetrieve()
    monkeypatch.setattr("kvprism.agents.research.make_ask", lambda **kwargs: ResearchAsk())
    monkeypatch.setattr(importlib.import_module("kvprism.tools.rag_retrieve"), "rag_retrieve", retrieve)
    monkeypatch.setattr("kvprism.agents.domain.rag_retrieve", retrieve)
    monkeypatch.setattr("kvprism.agents.common.ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setattr("kvprism.agents.common.web_search", lambda query, max_results: [
        make_web_source("turboquant" if "TurboQuant" in query else "itme")])
    synthesized = make_synthesis().model_dump_json()
    for tech in ("turboquant", "itme"):
        synthesized = synthesized.replace(f"paper-{tech}", f"{tech}:p1:0")
    synthesized = SynthesisResult.model_validate_json(synthesized)
    monkeypatch.setattr("kvprism.agents.synthesize.ChatOpenAI", lambda **kwargs: StructuredLLM(synthesized))

    class ReportWriter:
        def invoke(self, messages):
            return SimpleNamespace(content="시장성·이해관계자·도메인 적용의 조건을 분석합니다 [turboquant:p1:0]")

    monkeypatch.setattr("kvprism.agents.report.ChatOpenAI", lambda **kwargs: ReportWriter())
    monkeypatch.setattr("kvprism.agents.supervisor.ChatOpenAI", lambda **kwargs: ScriptedSupervisorInstance)
    ScriptedSupervisorInstance = ScriptedSupervisor(NORMAL)
    monkeypatch.setattr("kvprism.agents.report_quality.ChatOpenAI", lambda **kwargs: PassingJudge())
    output = app.run_pipeline(make_state()["request"], output_root=tmp_path)
    assert output.status == "succeeded", output.termination_reason
    assert output.output.report.pdf_path.is_file()
    assert output.quality.passed and output.quality.page_count <= 10
    assert (tmp_path / "checkpoints.sqlite").exists()
