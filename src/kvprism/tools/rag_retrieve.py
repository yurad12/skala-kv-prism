"""논문 RAG 검색 도구 (설계 문서 2.4 rag_retrieve).

dense(bge-m3) 검색과 BM25 검색을 각각 수행해 순위를 융합하고, doc_id 로 거른 뒤 상위 k 개를
State 의 Source 목록으로 돌려준다.

- source_id   : 청크 id ("turboquant:p7:0"). 같은 청크는 항상 같은 id 라 여러 노드가 인용해도 merge_sources 가 하나로 합친다
- url_or_page : "[TQ p.7]" 형식의 인용 태그. 프롬프트와 보고서 본문에는 이 태그를 쓴다
- excerpt     : 검색된 청크 본문 그대로 (LLM 요약 아님). 검증 노드가 근거 문장 포함 여부를 여기서 확인한다
"""

from langchain_core.documents import Document

from kvprism.graph.state import Source
from kvprism.rag import load_config, load_or_build_index, search

_index: dict | None = None  # 임베딩 모델 로드가 느려서 프로세스당 한 번만 만든다


def get_index() -> dict:
    global _index
    if _index is None:
        _index = load_or_build_index(load_config())
    return _index


def resolve_doc_id(doc_id: str, cfg: dict, available: list[str]) -> str:
    """doc_id 는 PDF 파일명(turboquant, itme)이지만 태그(TQ, ITME)나 대소문자 차이도 받아준다."""
    key = doc_id.strip().lower()
    for cand in available:
        if cand.lower() == key or cfg["docs"].get(cand, {}).get("tag", "").lower() == key:
            return cand
    raise ValueError(f"인덱스에 없는 doc_id: {doc_id!r} (있는 문서: {available})")


def doc_to_source(doc: Document, cfg: dict) -> Source:
    """검색된 청크 Document 를 State 의 Source 로 바꾼다."""
    meta = cfg["docs"].get(doc.metadata["doc_id"], {})
    return Source(
        source_id=doc.metadata["chunk_id"],
        source_kind="paper",
        title=meta.get("title", doc.metadata["doc_id"]),
        author_or_org=meta.get("authors", "Unknown"),
        published_at=meta.get("published"),  # rag.yaml 의 날짜는 date 로 파싱된다
        url_or_page=doc.metadata["tag"],
        excerpt=doc.page_content,
        doc_id=doc.metadata["doc_id"],
    )


def rag_retrieve(doc_id: str, query: str, k: int | None = None) -> list[Source]:
    """논문 청크를 검색해 인용 태그가 붙은 Source 목록을 돌려준다.

    Args:
        doc_id: 검색 대상 논문 (Technology.paper_doc_id). PDF 파일명("turboquant", "itme") 또는 태그("TQ", "ITME").
        query: 검색 질의. 문서가 영어 논문이므로 영어로 받는다.
        k: 돌려줄 청크 수. 없으면 configs/rag.yaml 의 top_k.
    """
    index = get_index()
    doc_id = resolve_doc_id(doc_id, index["cfg"], index["doc_ids"])
    return [doc_to_source(d, index["cfg"]) for d in search(index, query, doc_id, k)]
