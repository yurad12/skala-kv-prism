"""RAG 파이프라인 단위 테스트 (임베딩 모델·인덱스 없이 돌아가는 부분)."""

from datetime import date

from langchain_core.documents import Document

from kvprism.graph.state import Source
from kvprism.rag import clean, load_config, rrf, split_pages, tokenize
from kvprism.rag.metrics import gold_sets, metrics
from kvprism.tools.rag_retrieve import doc_to_source, resolve_doc_id


def page(doc_id, page_no, text):
    return Document(page_content=text, metadata={"doc_id": doc_id, "page": page_no})


def test_clean_normalizes_ligatures_hyphens_and_whitespace():
    assert clean("eﬃcient   quanti-\nzation\n of  KV") == "efficient quantization of KV"


def test_chunks_respect_size_overlap_and_page():
    text = " ".join(f"w{i}" for i in range(600))  # 약 3,000자
    chunks = split_pages([page("tq", 3, text)], chunk_size=800, chunk_overlap=100)
    assert len(chunks) >= 4
    assert all(len(c.page_content) <= 800 for c in chunks)
    assert all(c.metadata["doc_id"] == "tq" and c.metadata["page"] == 3 for c in chunks)
    assert [c.metadata["chunk_id"] for c in chunks][:2] == ["tq:p3:0", "tq:p3:1"]
    # 겹침: 앞 청크의 마지막 단어가 다음 청크 앞부분에 단어 그대로 다시 나타난다
    last_word = chunks[0].page_content.split()[-1]
    assert last_word in chunks[1].page_content[:200].split()


def test_empty_page_is_skipped():
    assert split_pages([page("tq", 1, "")]) == []


def test_tokenize_keeps_decimals_and_hangul():
    assert tokenize("KV-cache 45.29 GB/s 압축 quantization") == ["kv", "cache", "45.29", "gb", "s", "압축", "quantization"]


def test_rrf_prefers_items_ranked_high_in_both_lists():
    fused = rrf([1, 2, 3], [3, 4, 1], k=60, n=10)
    assert set(fused[:2]) == {1, 3}
    assert len(rrf([1, 2, 3], [4, 5, 6], n=4)) == 4


def test_metrics_hit_and_mrr():
    rankings = [[5, 1, 9], [7, 8, 9], [2, 3, 4]]
    golds = [{1}, {9}, {0}]
    m = metrics(rankings, golds, ks=(1, 3, 5), top_k=10)
    assert m == {"n": 3, "mrr": (0.5 + 1 / 3 + 0) / 3, "hit@1": 0, "hit@3": 2, "hit@5": 2}


def test_gold_sets_allow_split_numbers():
    chunks = [page("tq", 1, "Full Cache 16 45 .29 45.16"), page("x", 1, "45.29")]
    assert gold_sets([{"doc": "tq", "evidence": r"45\.29"}], chunks) == [{0}]


def test_doc_to_source_builds_state_source():
    cfg = load_config()
    doc = Document(page_content="KV cache to 3.5 bits",
                   metadata={"doc_id": "turboquant", "page": 7, "chunk_id": "turboquant:p7:2", "tag": "[TQ p.7]"})
    src = doc_to_source(doc, cfg)
    assert isinstance(src, Source)
    assert src.source_id == "turboquant:p7:2" and src.url_or_page == "[TQ p.7]"
    assert src.source_kind == "paper" and src.doc_id == "turboquant"
    assert src.published_at == date(2025, 4, 28) and src.excerpt == "KV cache to 3.5 bits"


def test_resolve_doc_id_accepts_tag_and_case():
    cfg = {"docs": {"turboquant": {"tag": "TQ"}, "itme": {"tag": "ITME"}}}
    available = ["itme", "turboquant"]
    assert resolve_doc_id("TQ", cfg, available) == "turboquant"
    assert resolve_doc_id("Itme", cfg, available) == "itme"
    assert resolve_doc_id("turboquant", cfg, available) == "turboquant"
    try:
        resolve_doc_id("kivi", cfg, available)
    except ValueError as e:
        assert "kivi" in str(e)
    else:
        raise AssertionError("없는 문서는 ValueError")
