"""검색 평가 지표: 골든 근거 세트 기반 Hit@K, MRR@10 (설계 문서 2.2.1).

골든 세트 문항: {"id", "doc", "type", "en", "ko", "evidence"(정규식)}
정답 청크 = 정답 문서(doc)에서 나왔고 evidence 정규식과 일치하는 청크.
청크 id 가 아니라 근거 문자열로 판정하므로 청킹이 바뀌어도 골든 세트를 다시 만들 필요가 없다.
"""

import json
import re
from pathlib import Path

from langchain_core.documents import Document


def load_qa(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text())


def gold_sets(qas: list[dict], chunks: list[Document]) -> list[set[int]]:
    """문항마다 정답 청크의 위치 집합."""
    out = []
    for qa in qas:
        pattern = re.compile(qa["evidence"].replace("\\.", "\\s?\\."), re.I)  # "45 .29" 처럼 갈라진 숫자 허용
        out.append({i for i, c in enumerate(chunks)
                    if c.metadata["doc_id"] == qa["doc"] and pattern.search(c.page_content)})
    return out


def metrics(rankings: list[list[int]], golds: list[set[int]], ks=(1, 3, 5), top_k: int = 10) -> dict:
    """질의별 이진 판정. 상위 K 안에 정답 청크가 하나라도 있으면 hit. MRR 은 top_k 밖이면 0."""
    firsts = []  # 문항별 첫 정답 순위 (없으면 None)
    for ranked, gold in zip(rankings, golds, strict=True):
        firsts.append(next((r for r, idx in enumerate(ranked[:top_k], start=1) if idx in gold), None))
    n = len(firsts)
    out = {"n": n, "mrr": sum(1 / f for f in firsts if f) / n if n else 0.0}
    for k in ks:
        out[f"hit@{k}"] = sum(1 for f in firsts if f and f <= k)
    return out


def cell(m: dict, key: str) -> str:
    """표에 넣을 문자열. "22/24 (0.917)" 또는 MRR "0.727"."""
    if key == "mrr":
        return f"{m['mrr']:.3f}"
    return f"{m[key]}/{m['n']} ({m[key] / m['n']:.3f})" if m["n"] else "-"
