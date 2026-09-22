"""논문 RAG 검색 도구."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Source는 State 정의에 있다.
    from ..graph.state import Source

DEFAULT_K = 5


def rag_retrieve(doc_id: str, query: str, k: int = DEFAULT_K) -> list[Source]:
    """논문 청크를 검색해 인용 태그가 붙은 Source 목록을 돌려준다.

    dense(bge-m3) 검색과 BM25 검색을 각각 수행해 순위를 융합하고,
    doc_id로 거른 뒤 상위 k개를 돌려준다.
    Source의 excerpt에는 청크 본문을 그대로 담고,
    url_or_page에는 `[TQ p.7]` 형식의 페이지 태그를 담는다.

    Args:
        doc_id: 검색 대상 논문. 기술 하나에 논문 하나가 대응한다.
        query: 검색 질의. 문서가 영어 논문이므로 영어로 받는다.
        k: 돌려줄 청크 수.
    """
    raise NotImplementedError
