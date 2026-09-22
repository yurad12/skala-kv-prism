"""실제 관점 노드의 State 입출력 계약 테스트."""

from datetime import date

import pytest

from kvprism.agents import domain_node, market_node, stakeholder_node
from kvprism.graph.state import (
    EvidenceClaim,
    GraphState,
    PipelineInput,
    PerspectiveResult,
    ResearchResult,
    Source,
    Technology,
    TechnologyResearch,
)


class FakeChatOpenAI:
    """실제 관점 노드는 실행하되 외부 LLM 호출만 대체한다."""

    prompts: list[str] = []

    def __init__(self, *args, **kwargs):
        self.schema = None

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def invoke(self, messages):
        self.prompts.append(messages[-1].content)
        return self.schema.model_validate(
            {
                "evaluations": [
                    {
                        "technology_id": technology_id,
                        "summary": f"{technology_id} 평가",
                        "positive_evidence": [
                            {
                                "statement": "도입 효과가 있습니다.",
                                "source_id": source_id,
                            }
                        ],
                        "negative_evidence": [
                            {
                                "statement": "도입 제약이 있습니다.",
                                "source_id": source_id,
                            }
                        ],
                        "trl_signals": [
                            {
                                "statement": "검증 구현이 릴리스되었습니다.",
                                "source_id": source_id,
                            }
                        ],
                    }
                    for technology_id, source_id in (
                        ("turboquant", "web-turboquant"),
                        ("itme", "web-itme"),
                    )
                ]
            }
        )


def make_web_source(technology_id: str) -> Source:
    """관점 노드가 새로 발견하는 테스트용 웹 출처를 만든다."""

    return Source(
        source_id=f"web-{technology_id}",
        source_kind="web",
        title=f"{technology_id} 웹 자료",
        author_or_org="테스트 기관",
        published_at=date(2026, 1, 1),
        url_or_page=f"https://example.com/{technology_id}",
        excerpt=f"{technology_id} web deployment evidence",
    )


def make_research_profile(technology_id: str) -> TechnologyResearch:
    """관점별 프롬프트에 전달할 논문 기술 조사 결과를 만든다."""

    source_id = f"paper-{technology_id}"
    return TechnologyResearch(
        technology_id=technology_id,
        overview=f"{technology_id} 논문 개요",
        performance_metrics=[
            EvidenceClaim(statement="처리량이 개선되었습니다.", source_id=source_id)
        ],
        experimental_conditions=[
            EvidenceClaim(
                statement="128 concurrent requests에서 측정했습니다.",
                source_id=source_id,
            )
        ],
        limitations=[
            EvidenceClaim(statement="특정 하드웨어 조건이 필요합니다.", source_id=source_id)
        ],
        trl_signals=[
            EvidenceClaim(statement="프로토타입에서 검증했습니다.", source_id=source_id)
        ],
    )


@pytest.fixture
def graph_state() -> GraphState:
    """실제 관점 노드 세 개가 공통으로 사용하는 입력 State를 만든다."""

    technologies = [
        Technology(
            technology_id="turboquant",
            name="TurboQuant",
            kind="software",
            paper_doc_id="turboquant",
            paper_title="TurboQuant",
        ),
        Technology(
            technology_id="itme",
            name="ITME",
            kind="hardware",
            paper_doc_id="itme",
            paper_title="ITME",
        ),
    ]
    return {
        "request": PipelineInput(
            domain="데이터센터",
            scenario="장문맥 추론",
            technologies=technologies,
        ),
        "research": ResearchResult(
            technologies=[
                make_research_profile("turboquant"),
                make_research_profile("itme"),
            ]
        ),
        "sources": [
            Source(
                source_id=f"paper-{technology_id}",
                source_kind="paper",
                title=f"{technology_id} 논문",
                author_or_org="연구팀",
                url_or_page="p.1",
                excerpt="논문 원문",
                doc_id=technology_id,
            )
            for technology_id in ("turboquant", "itme")
        ],
        "retry_count": 0,
        "retry_targets": [],
    }


@pytest.fixture(autouse=True)
def replace_external_calls(monkeypatch):
    """웹 검색과 LLM만 고정 응답으로 교체해 노드 계약을 재현 가능하게 검사한다."""

    FakeChatOpenAI.prompts = []

    def fake_web_search(query: str, max_results: int):
        technology_id = "turboquant" if "TurboQuant" in query else "itme"
        return [make_web_source(technology_id)]

    def fake_rag_retrieve(doc_id: str, query: str, k: int | None = None):
        return [
            Source(
                source_id=f"rag-{doc_id}",
                source_kind="paper",
                title=f"{doc_id} 논문",
                author_or_org="연구팀",
                url_or_page="[TQ p.9]",
                excerpt=f"{doc_id} paper experimental setup excerpt",
                doc_id=doc_id,
            )
        ]

    monkeypatch.setattr("kvprism.agents.common.web_search", fake_web_search)
    monkeypatch.setattr("kvprism.agents.common.ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setattr("kvprism.agents.domain.rag_retrieve", fake_rag_retrieve)


@pytest.mark.parametrize(
    ("node", "result_key", "perspective", "extra_source_ids"),
    [
        (market_node, "market_eval", "market", set()),
        (stakeholder_node, "stakeholder_eval", "stakeholder", set()),
        # 도메인 노드는 논문 RAG 청크(설계서 2.1)도 출처로 등록한다
        (domain_node, "domain_eval", "domain", {"rag-turboquant", "rag-itme"}),
    ],
)
def test_perspective_node_returns_only_its_result_and_new_sources(
    graph_state, node, result_key, perspective, extra_source_ids
) -> None:
    output = node(graph_state)

    assert set(output) == {result_key, "sources"}
    assert isinstance(output[result_key], PerspectiveResult)
    assert output[result_key].perspective == perspective
    assert {source.source_id for source in output["sources"]} == {
        "web-turboquant",
        "web-itme",
        *extra_source_ids,
    }


def test_domain_node_retrieves_paper_chunks_for_request_technologies(graph_state, monkeypatch) -> None:
    """도메인 노드는 request.technologies 의 paper_doc_id 로 rag_retrieve 를 부르고 그 발췌문을 프롬프트에 넣는다."""

    calls: list[str] = []

    def spy_rag_retrieve(doc_id: str, query: str, k: int | None = None):
        calls.append(doc_id)
        return [
            Source(source_id=f"rag-{doc_id}", source_kind="paper", title="논문", author_or_org="연구팀",
                   url_or_page="[TQ p.9]", excerpt=f"{doc_id} rag excerpt", doc_id=doc_id)
        ]

    monkeypatch.setattr("kvprism.agents.domain.rag_retrieve", spy_rag_retrieve)
    domain_node(graph_state)

    assert set(calls) == {"turboquant", "itme"}
    assert "turboquant rag excerpt" in FakeChatOpenAI.prompts[-1]


def test_perspective_nodes_use_required_research_context(graph_state) -> None:
    market_node(graph_state)
    stakeholder_node(graph_state)
    domain_node(graph_state)

    market_prompt, stakeholder_prompt, domain_prompt = FakeChatOpenAI.prompts
    assert "turboquant 논문 개요" in market_prompt
    assert "특정 하드웨어 조건이 필요합니다." in stakeholder_prompt
    assert "128 concurrent requests에서 측정했습니다." in domain_prompt
    assert "turboquant web deployment evidence" in domain_prompt


def test_perspective_node_requires_research_result(graph_state) -> None:
    del graph_state["research"]

    with pytest.raises(ValueError, match="research 결과가 필요합니다"):
        market_node(graph_state)
