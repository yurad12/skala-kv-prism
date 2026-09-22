"""기술 조사(research) 노드 (설계 문서 4.1, 4.3).

흐름:
  항목별 영어 질의로 검색 → 검색 결과로 답할 수 있는지 LLM 이 판정 → 부족하면 질의를 한 번 고쳐 재검색
  → 모은 청크만 근거로 TechnologyResearch 추출 → 인용 검증 (없는 source_id, 청크에 없는 숫자 제거)

읽는 키: request        쓰는 키: research, sources (이 노드가 인용한 논문 청크만)
"""

import hashlib
import logging
import os
import re

from pydantic import BaseModel, Field

from kvprism.graph.state import GraphState, ResearchResult, Source, Technology, TechnologyResearch
from kvprism.prompts import research as P
from kvprism.rag.config import ROOT

log = logging.getLogger(__name__)

CLAIM_FIELDS = ("mechanism", "performance_metrics", "experimental_conditions", "limitations", "trl_signals")
REQUIRED_FIELDS = ("performance_metrics", "experimental_conditions", "limitations")  # State 계약: 최소 1건
CACHE_DIR = ROOT / "outputs" / "cache" / "research"


class SearchQueries(BaseModel):
    """LLM 이 쓴 영어 검색 질의 목록. 항목 순서대로 2개씩."""

    queries: list[str]


class RelevanceVerdict(BaseModel):
    """관련성 판정 결과. 부족하면 고친 영어 질의를 함께 돌려준다."""

    sufficient: bool
    revised_query: str | None = Field(default=None, description="부족할 때 한 번 고친 영어 질의")


# ---------------------------------------------------------------------------
# LLM 호출 (설계 문서 2.4: Generator 는 reasoning effort low, temperature 미사용)
# ---------------------------------------------------------------------------


def make_ask(replay: bool = False):
    """ask(schema, system, user) 함수를 만든다. 결과는 파일에 저장하고 replay=True 면 저장된 것을 재사용한다."""
    from langchain_openai import ChatOpenAI

    model = os.environ.get("GENERATOR_MODEL", "gpt-5.6-luna")
    llm = ChatOpenAI(model=model, reasoning_effort="low")

    def ask(schema: type[BaseModel], system: str, user: str) -> BaseModel:
        key = hashlib.sha256(f"{model}\n{schema.__name__}\n{system}\n{user}".encode()).hexdigest()[:24]
        path = CACHE_DIR / f"{key}.json"
        if replay and path.exists():
            return schema.model_validate_json(path.read_text())
        result = llm.with_structured_output(schema).invoke([("system", system), ("user", user)])
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(result.model_dump_json(indent=2))
        return result

    return ask


def format_context(sources: list[Source]) -> str:
    """프롬프트에 넣을 발췌문. 태그와 source_id 를 같이 보여 LLM 이 인용에 쓰게 한다."""
    return "\n\n".join(f"{s.url_or_page} (source_id: {s.source_id})\n{s.excerpt}" for s in sources)


# ---------------------------------------------------------------------------
# 1) 검색: LLM 이 항목별 영어 질의를 쓰고, 검색 결과가 부족하면 질의를 한 번 고쳐 다시 검색
# ---------------------------------------------------------------------------

QUERIES_PER_ASPECT = 2


def write_queries(tech: Technology, ask) -> dict[str, list[str]]:
    """한국어 조사 항목을 보고 LLM 이 항목마다 영어 질의 2개를 쓴다 (설계서 2.2.2: 에이전트가 질의를 영어로 쓴다)."""
    aspects = list(P.ASPECT_QUESTIONS_KO)
    items = "\n".join(f"{i}. {P.ASPECT_QUESTIONS_KO[a]}" for i, a in enumerate(aspects, start=1))
    result = ask(SearchQueries, P.QUERY_SYSTEM, P.QUERY_USER.format(
        title=tech.paper_title, name=tech.name, n=len(aspects) * QUERIES_PER_ASPECT, items=items))
    queries = result.queries[: len(aspects) * QUERIES_PER_ASPECT]
    by_aspect = {a: queries[i * QUERIES_PER_ASPECT:(i + 1) * QUERIES_PER_ASPECT] for i, a in enumerate(aspects)}
    for a in aspects:  # LLM 이 개수를 못 맞추면 항목 이름을 넣은 기본 질의로 채운다
        by_aspect[a] = by_aspect[a] or [f"{tech.name} {a.replace('_', ' ')}"]
    return by_aspect


def collect_sources(tech: Technology, retrieve, ask, k: int | None = None) -> dict[str, Source]:
    found: dict[str, Source] = {}  # source_id → Source
    for aspect, queries in write_queries(tech, ask).items():
        aspect_sources: dict[str, Source] = {}
        for query in queries:
            for s in retrieve(tech.paper_doc_id, query, k):
                aspect_sources.setdefault(s.source_id, s)
        if not aspect_sources:
            log.warning("[%s/%s] 검색 결과 없음", tech.technology_id, aspect)
            continue

        # 관련성 판정: 이 발췌문으로 항목 질문에 답할 수 있는가?
        verdict = ask(RelevanceVerdict, P.RELEVANCE_SYSTEM, P.RELEVANCE_USER.format(
            title=tech.paper_title, question=P.ASPECT_QUESTIONS_KO[aspect], query=" | ".join(queries),
            context=format_context(list(aspect_sources.values()))))
        if not verdict.sufficient and verdict.revised_query:
            log.info("[%s/%s] 재질의: %s", tech.technology_id, aspect, verdict.revised_query)
            for s in retrieve(tech.paper_doc_id, verdict.revised_query, k):
                aspect_sources.setdefault(s.source_id, s)
        found.update(aspect_sources)
    return found


# ---------------------------------------------------------------------------
# 2) 인용 검증: 근거 없는 인용과 수치를 걸러낸다 (설계 문서 4.1 "근거 청크가 없는 수치는 쓰지 않는다")
# ---------------------------------------------------------------------------

# 문장에서 수치를 뽑는 정규식. 글자·하이픈·점 바로 뒤의 숫자(Llama-3.1, v0.17.0)는 이름의 일부라 제외한다
NUMBER_RE = re.compile(r"(?<![A-Za-z\-.\d−])\d+(?:\.\d+)?")
THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}\b)")  # 1,000 → 1000
SOURCE_ID_RE = re.compile(r"\[?\b[a-z0-9_]+:p\d+:\d+\]?")  # overview 에 섞여 들어온 source_id


def numbers_in(text: str) -> set[float]:
    """비교용 숫자 값. 값이 같으면 표기가 달라도 같게 본다 ("1.80" == "1.8")."""
    return {float(n) for n in NUMBER_RE.findall(THOUSANDS_RE.sub("", text))}


def validate_claims(profile: dict, tech: Technology, sources: dict[str, Source]) -> dict:
    """1) source_id 가 검색 결과에 없으면 버린다.
    2) statement 의 숫자가 인용한 청크에 없으면, 숫자를 모두 담은 다른 검색 청크로 인용을 옮기고 없으면 버린다.
    """
    profile["technology_id"] = tech.technology_id
    profile["overview"] = re.sub(r"\s{2,}", " ", SOURCE_ID_RE.sub("", profile["overview"])).strip()
    for field in CLAIM_FIELDS:
        kept = []
        for claim in profile[field]:
            if claim["source_id"] not in sources:
                log.warning("[%s] 근거 없는 인용 제거 (%s): %s", tech.technology_id, claim["source_id"], claim["statement"])
                continue
            nums = numbers_in(claim["statement"])
            if nums and not nums <= numbers_in(sources[claim["source_id"]].excerpt):
                alt = next((sid for sid, s in sources.items() if nums <= numbers_in(s.excerpt)), None)
                if alt is None:
                    log.warning("[%s] 발췌문에 없는 수치로 제거: %s", tech.technology_id, claim["statement"])
                    continue
                log.info("[%s] 인용 이동 %s → %s", tech.technology_id, claim["source_id"], alt)
                claim["source_id"] = alt
            kept.append(claim)
        profile[field] = kept
    return profile


# ---------------------------------------------------------------------------
# 3) 추출: 모은 청크만 근거로 TechnologyResearch 를 만든다
# ---------------------------------------------------------------------------


def extract_profile(tech: Technology, sources: dict[str, Source], ask) -> TechnologyResearch:
    user = P.EXTRACT_USER.format(name=tech.name, technology_id=tech.technology_id, title=tech.paper_title,
                                 context=format_context(list(sources.values())))
    profile = validate_claims(ask(TechnologyResearch, P.EXTRACT_SYSTEM, user).model_dump(), tech, sources)

    # 검증에서 필수 항목이 비면 사유를 붙여 한 번 다시 추출한다
    empty = [f for f in REQUIRED_FIELDS if not profile[f]]
    if empty:
        log.warning("[%s] 필수 항목 비어 있음 %s → 재추출", tech.technology_id, empty)
        retry_user = user + P.EXTRACT_RETRY.format(fields=", ".join(empty))
        profile = validate_claims(ask(TechnologyResearch, P.EXTRACT_SYSTEM, retry_user).model_dump(), tech, sources)
    return TechnologyResearch.model_validate(profile)


# ---------------------------------------------------------------------------
# LangGraph 노드
# ---------------------------------------------------------------------------


def research_node(state: GraphState, retrieve=None, ask=None) -> dict:
    """새로 인용한 출처만 sources 로 돌려준다 (merge_sources 리듀서가 합침). retrieve/ask 는 바꿔 끼울 수 있다."""
    if retrieve is None:
        from kvprism.tools.rag_retrieve import rag_retrieve

        retrieve = rag_retrieve
    ask = ask or make_ask(replay=state["request"].replay)

    profiles = []
    cited: dict[str, Source] = {}
    for tech in state["request"].technologies:
        sources = collect_sources(tech, retrieve, ask)
        profile = extract_profile(tech, sources, ask)
        profiles.append(profile)
        for field in CLAIM_FIELDS:
            for claim in getattr(profile, field):
                cited.setdefault(claim.source_id, sources[claim.source_id])
    return {"research": ResearchResult(technologies=profiles), "sources": list(cited.values())}
