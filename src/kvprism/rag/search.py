"""검색: dense 와 BM25 순위를 RRF 로 융합하고 doc_id 로 걸러 상위 k 개 (설계 문서 2.2, 2.4).

순위 함수들은 index["chunks"] 의 위치(정수) 목록을 돌려준다. search() 가 그것을 Document 로 바꾸고
metadata["tag"] 에 "[TQ p.7]" 형식의 인용 태그를 붙인다.
"""

import numpy as np
from langchain_core.documents import Document

from kvprism.rag.config import tag_of
from kvprism.rag.index import tokenize


def rrf(*rank_lists: list[int], k: int = 60, n: int = 10) -> list[int]:
    """Reciprocal Rank Fusion. 각 목록에서 r 위인 항목에 1/(k+r) 점을 주고 합산해 정렬한다."""
    scores: dict[int, float] = {}
    for ranks in rank_lists:
        for r, idx in enumerate(ranks, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + r)
    return sorted(scores, key=scores.__getitem__, reverse=True)[:n]


def dense_rank(index: dict, query: str, doc_id: str | None = None, n: int = 10) -> list[int]:
    docs = index["vectorstore"].similarity_search(query, k=n, filter={"doc_id": doc_id} if doc_id else None)
    return [index["pos"][d.metadata["chunk_id"]] for d in docs]


def bm25_rank(index: dict, query: str, doc_id: str | None = None, n: int = 10) -> list[int]:
    scores = np.asarray(index["bm25"].get_scores(tokenize(query)), dtype=float)
    if doc_id is not None:  # 다른 문서의 청크는 제외
        doc_ids = np.array([c.metadata["doc_id"] for c in index["chunks"]])
        scores[doc_ids != doc_id] = -np.inf
    order = np.argsort(-scores)[:n]
    # 질의 토큰이 하나도 없는 청크(0점)는 순위에 넣지 않는다. 넣으면 RRF 에서 근거 없는 점수를 받는다
    return [int(i) for i in order if scores[i] > 0]


def rank(index: dict, query: str, doc_id: str | None = None, mode: str = "hybrid") -> list[int]:
    """mode: hybrid(채택) / dense / bm25. 평가 스크립트가 방식별 비교에 쓴다."""
    cfg = index["cfg"]
    n = cfg["candidate_k"]
    if mode == "dense":
        return dense_rank(index, query, doc_id, n)
    if mode == "bm25":
        return bm25_rank(index, query, doc_id, n)
    return rrf(dense_rank(index, query, doc_id, n), bm25_rank(index, query, doc_id, n), k=cfg["rrf_k"], n=n)


def search(index: dict, query: str, doc_id: str | None = None, k: int | None = None) -> list[Document]:
    """채택 방식(영어 질의, 순위 융합)으로 상위 k 개 청크. metadata["tag"] 에 인용 태그를 붙여 돌려준다."""
    cfg = index["cfg"]
    results = []
    for i in rank(index, query, doc_id)[: k or cfg["top_k"]]:
        chunk = index["chunks"][i]
        tag = f"[{tag_of(cfg, chunk.metadata['doc_id'])} p.{chunk.metadata['page']}]"
        results.append(Document(page_content=chunk.page_content, metadata={**chunk.metadata, "tag": tag}))
    return results
