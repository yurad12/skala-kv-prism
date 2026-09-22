"""기술 조사 노드 테스트: 검색기와 LLM 을 가짜로 바꿔 흐름(재질의, 인용 검증, 출처 등록)을 확인한다."""

from kvprism.agents.research import (
    RelevanceVerdict,
    SearchQueries,
    collect_sources,
    extract_profile,
    research_node,
    validate_claims,
)
from kvprism.graph.state import EvidenceClaim, PipelineInput, ResearchResult, Source, Technology, TechnologyResearch


def make_source(doc: str, page: int, n: int, text: str) -> Source:
    tag = {"turboquant": "TQ", "itme": "ITME"}[doc]
    return Source(source_id=f"{doc}:p{page}:{n}", source_kind="paper", title=doc, author_or_org="A. et al.",
                  url_or_page=f"[{tag} p.{page}]", excerpt=text, doc_id=doc)


def make_techs() -> list[Technology]:
    return [
        Technology(technology_id="turboquant", name="TurboQuant", kind="software", paper_doc_id="turboquant",
                   paper_title="TurboQuant"),
        Technology(technology_id="itme", name="ITME", kind="hardware", paper_doc_id="itme", paper_title="ITME"),
    ]


class FakeRetrieve:
    """검색기 대역. 'REVISED' 가 든 질의(재질의)에는 다른 청크를 돌려준다."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def __call__(self, doc_id, query, k=None):
        self.calls.append((doc_id, query))
        if "REVISED" in query:
            return [make_source(doc_id, 9, 0, f"{doc_id} revised passage")]
        return [make_source(doc_id, 1, 0, f"{doc_id} passage A"), make_source(doc_id, 2, 0, f"{doc_id} passage B")]


class FakeAsk:
    """LLM 대역. 관련성 판정은 '성능 수치' 항목에서만 부족, 추출은 고정 응답 (지어낸 인용 하나 포함)."""

    def __init__(self):
        self.relevance_calls = 0

    def __call__(self, schema, system, user):
        if schema is SearchQueries:  # 항목 5개 × 2개 = 영어 질의 10개
            return SearchQueries(queries=[f"query {i}" for i in range(10)])
        if schema is RelevanceVerdict:
            self.relevance_calls += 1
            if "성능 수치" in user:
                return RelevanceVerdict(sufficient=False, revised_query="REVISED throughput numbers")
            return RelevanceVerdict(sufficient=True)
        tech_id = "turboquant" if "turboquant" in user else "itme"
        return TechnologyResearch(
            technology_id="wrong-id-from-llm",
            overview=f"{tech_id} 개요",
            mechanism=[EvidenceClaim(statement="동작 방식", source_id=f"{tech_id}:p1:0")],
            performance_metrics=[
                EvidenceClaim(statement="재질의로 찾은 수치", source_id=f"{tech_id}:p9:0", stance="positive"),
                EvidenceClaim(statement="지어낸 인용", source_id=f"{tech_id}:p99:0", stance="positive"),
            ],
            experimental_conditions=[EvidenceClaim(statement="실험 조건", source_id=f"{tech_id}:p1:0")],
            limitations=[EvidenceClaim(statement="한계", source_id=f"{tech_id}:p2:0", stance="negative")],
        )


def test_research_node_requeries_once_validates_citations_and_registers_sources():
    retrieve, ask = FakeRetrieve(), FakeAsk()
    request = PipelineInput(domain="데이터센터", scenario="장문맥 멀티테넌트", technologies=make_techs())
    out = research_node({"request": request, "retry_count": 0, "retry_targets": [], "sources": []}, retrieve, ask)

    assert set(out) == {"research", "sources"}  # 자기 키만 돌려준다
    result = out["research"]
    assert isinstance(result, ResearchResult)
    assert [t.technology_id for t in result.technologies] == ["turboquant", "itme"]  # LLM 이 준 id 는 덮어씀
    # 항목 5개 × 기술 2개 = 관련성 판정 10회, 그중 성능 수치 항목만 재질의 1회씩
    assert ask.relevance_calls == 10
    assert sum("REVISED" in q for _, q in retrieve.calls) == 2
    # 검색 결과에 없는 source_id 를 인용한 statement 는 제거
    assert [c.source_id for c in result.technologies[0].performance_metrics] == ["turboquant:p9:0"]
    # 인용된 출처만 등록, 중복 없음
    ids = sorted(s.source_id for s in out["sources"])
    assert ids == sorted({"turboquant:p1:0", "turboquant:p9:0", "turboquant:p2:0", "itme:p1:0", "itme:p9:0", "itme:p2:0"})


def test_validate_claims_moves_or_drops_claims_whose_numbers_are_not_in_excerpt():
    tech = make_techs()[1]
    sources = {
        "itme:p9:4": make_source("itme", 9, 4, "0 5 10 15 20 25 Turn 1.00 1.25 1.50 TTFT Speedup"),
        "itme:p2:4": make_source("itme", 2, 4, "ITME achieves a 1.80x throughput improvement over NVMe-oF with 1000 tokens"),
    }
    profile = TechnologyResearch(
        technology_id="itme", overview="ITME 개요 [itme:p2:4] 끝",
        experimental_conditions=[EvidenceClaim(statement="조건", source_id="itme:p2:4")],
        limitations=[EvidenceClaim(statement="한계", source_id="itme:p2:4", stance="negative")],
        performance_metrics=[
            EvidenceClaim(statement="ITME는 1.8× 처리량 향상 (1,000 tokens)", source_id="itme:p9:4", stance="positive"),  # 이동 (1.8 == 1.80)
            EvidenceClaim(statement="turn 5에서 1.81× speedup", source_id="itme:p9:4", stance="positive"),  # 제거
            EvidenceClaim(statement="Turn 25까지 측정", source_id="itme:p9:4"),  # 유지
        ],
    ).model_dump()
    out = validate_claims(profile, tech, sources)
    assert [(c["statement"][:4], c["source_id"]) for c in out["performance_metrics"]] == [("ITME", "itme:p2:4"), ("Turn", "itme:p9:4")]
    assert out["overview"] == "ITME 개요 끝"  # overview 에 섞인 source_id 제거


def test_extract_profile_retries_once_when_required_field_is_empty():
    tech = make_techs()[0]
    sources = {"turboquant:p1:0": make_source("turboquant", 1, 0, "passage")}
    calls = []

    def ask(schema, system, user):
        calls.append(user)
        # 1회차는 limitations 가 지어낸 인용뿐이라 검증에서 전부 제거되고, 2회차에 제대로 된 인용이 온다
        limitation = "turboquant:p77:0" if len(calls) == 1 else "turboquant:p1:0"
        return TechnologyResearch(
            technology_id="turboquant", overview="o",
            performance_metrics=[EvidenceClaim(statement="p", source_id="turboquant:p1:0")],
            experimental_conditions=[EvidenceClaim(statement="c", source_id="turboquant:p1:0")],
            limitations=[EvidenceClaim(statement="한계", source_id=limitation, stance="negative")],
        )

    profile = extract_profile(tech, sources, ask)
    assert len(calls) == 2 and "[재요청]" in calls[1] and "limitations" in calls[1]
    assert [c.statement for c in profile.limitations] == ["한계"]


def test_collect_sources_skips_llm_when_nothing_retrieved():
    calls = []

    def ask(schema, system, user):
        calls.append(schema)
        if schema is SearchQueries:
            return SearchQueries(queries=[f"q{i}" for i in range(10)])
        return RelevanceVerdict(sufficient=True)

    assert collect_sources(make_techs()[0], lambda d, q, k: [], ask) == {}
    assert calls == [SearchQueries]  # 질의 생성 1회뿐, 관련성 판정은 부르지 않음
