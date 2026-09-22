"""인덱스 만들기/불러오기 (설계 문서 2.2, 2.4).

- dense: bge-m3 임베딩을 Chroma 에 저장 (outputs/index/chroma)
- BM25 : rank_bm25 인덱스를 pickle 로 저장 (outputs/index/bm25.pkl)
- 청크 원문은 chunks.json 에 저장. 세 파일이 있으면 다시 만들지 않는다.

index 는 dict 하나로 다룬다: {"cfg", "chunks", "vectorstore", "bm25", "pos"}
"""

import json
import pickle
import re
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

from kvprism.rag.documents import load_pages, split_pages

# BM25 토큰화: 영숫자 덩어리 / 한글 덩어리. 소수점 붙은 숫자("45.29")가 한 토큰이 되게 한다
TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?|[가-힣]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def get_embeddings(cfg: dict) -> HuggingFaceEmbeddings:
    # normalize_embeddings=True 면 내적이 코사인 유사도가 된다
    return HuggingFaceEmbeddings(model=cfg["embedding_model"], encode_kwargs={"normalize_embeddings": True})


def index_exists(index_dir: Path) -> bool:
    return all((index_dir / name).exists() for name in ("chroma", "bm25.pkl", "chunks.json", "meta.json"))


def build_index(cfg: dict, papers_dir: Path, index_dir: Path) -> dict:
    """PDF 를 읽어 청킹하고 Chroma + BM25 인덱스를 index_dir 에 저장한다."""
    pages = load_pages(papers_dir)
    if not pages:
        raise FileNotFoundError(f"PDF 가 없습니다: {papers_dir}. scripts/download_papers.py 를 먼저 실행하세요.")
    chunks = split_pages(pages, cfg["chunk_size"], cfg["chunk_overlap"])
    if not chunks:
        raise ValueError("텍스트가 없는 PDF 라 청크를 만들 수 없습니다")

    # 옛 인덱스 위에 덧붙이지 않게 인덱스 파일만 지우고 시작 (.gitkeep 은 남긴다)
    index_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(index_dir / "chroma", ignore_errors=True)
    for name in ("bm25.pkl", "chunks.json", "meta.json"):
        (index_dir / name).unlink(missing_ok=True)
    vectorstore = Chroma.from_documents(
        chunks, get_embeddings(cfg), ids=[c.metadata["chunk_id"] for c in chunks],
        collection_name="papers", persist_directory=str(index_dir / "chroma"),
        # search_ef 를 넉넉히 두어 수백 개 규모에서 완전 탐색과 같은 순위가 나오게 한다
        collection_metadata={"hnsw:space": "cosine", "hnsw:search_ef": 200, "hnsw:construction_ef": 200},
    )
    bm25 = BM25Okapi([tokenize(c.page_content) for c in chunks])
    (index_dir / "bm25.pkl").write_bytes(pickle.dumps(bm25))
    (index_dir / "chunks.json").write_text(json.dumps(
        [{"page_content": c.page_content, "metadata": c.metadata} for c in chunks], ensure_ascii=False))
    (index_dir / "meta.json").write_text(json.dumps(
        {"embedding_model": cfg["embedding_model"], "n_chunks": len(chunks),
         "docs": sorted({c.metadata["doc_id"] for c in chunks})}, ensure_ascii=False, indent=2))
    return _make_index(cfg, chunks, vectorstore, bm25)


def load_index(cfg: dict, index_dir: Path) -> dict:
    meta = json.loads((index_dir / "meta.json").read_text())
    if meta["embedding_model"] != cfg["embedding_model"]:
        raise ValueError(f"인덱스의 임베딩 모델({meta['embedding_model']})이 설정({cfg['embedding_model']})과 다릅니다. "
                         "scripts/build_index.py --force 로 다시 만드세요.")
    chunks = [Document(**c) for c in json.loads((index_dir / "chunks.json").read_text())]
    vectorstore = Chroma(collection_name="papers", embedding_function=get_embeddings(cfg),
                         persist_directory=str(index_dir / "chroma"))
    bm25 = pickle.loads((index_dir / "bm25.pkl").read_bytes())
    return _make_index(cfg, chunks, vectorstore, bm25)


def load_or_build_index(cfg: dict, papers_dir: Path | None = None, index_dir: Path | None = None,
                        force: bool = False) -> dict:
    """있으면 불러오고 없으면 만든다. app.py 첫 실행과 스크립트가 같이 쓴다."""
    papers_dir = papers_dir or cfg["papers_dir"]
    index_dir = index_dir or cfg["index_dir"]
    if index_exists(index_dir) and not force:
        return load_index(cfg, index_dir)
    return build_index(cfg, papers_dir, index_dir)


def _make_index(cfg, chunks, vectorstore, bm25) -> dict:
    return {
        "cfg": cfg,
        "chunks": chunks,
        "vectorstore": vectorstore,
        "bm25": bm25,
        "pos": {c.metadata["chunk_id"]: i for i, c in enumerate(chunks)},  # chunk_id → chunks 의 위치
        "doc_ids": sorted({c.metadata["doc_id"] for c in chunks}),
    }
