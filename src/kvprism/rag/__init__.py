"""RAG 파이프라인: PDF 로딩 → 청킹 → 인덱싱(Chroma + BM25) → 순위 융합 검색 → 인용 태그.

에이전트(rag_retrieve 도구)와 검색 평가 스크립트가 같은 함수를 쓴다.
"""

from kvprism.rag.config import ROOT, load_config, tag_of
from kvprism.rag.documents import clean, load_pages, split_pages
from kvprism.rag.index import build_index, index_exists, load_index, load_or_build_index, tokenize
from kvprism.rag.search import bm25_rank, dense_rank, rank, rrf, search

__all__ = ["ROOT", "load_config", "tag_of", "clean", "load_pages", "split_pages", "build_index", "index_exists",
           "load_index", "load_or_build_index", "tokenize", "bm25_rank", "dense_rank", "rank", "rrf", "search"]
