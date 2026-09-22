"""골든 근거 세트로 검색 품질(Hit@1/3/5, MRR@10)을 잰다.

  uv run python scripts/eval_retrieval.py              # 운영 인덱스(선정 2편). 해당 논문 문항만 (dev 8, val 4)
  uv run python scripts/eval_retrieval.py --benchmark  # 벤치마크 코퍼스(후보 6편). dev 24 / val 12 = 설계 문서 2.2.2 조건

운영 코드(kvprism.rag)의 검색기를 그대로 써서 설계 문서의 실측치가 현재 코드에서 재현되는지 확인한다.
벤치마크는 doc_id 필터 없음(설계서 2.2.2 표 조건)과 필터 적용(운영 조건) 두 조건을 모두 출력한다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from kvprism.rag import load_config, load_or_build_index, rank
from kvprism.rag.metrics import cell, gold_sets, load_qa, metrics

SYSTEMS = [  # (표시 이름, 검색 모드, 질의 언어) — 설계서 2.2.2 표와 같은 행 구성
    ("dense(영)", "dense", "en"),
    ("dense(한)", "dense", "ko"),
    ("BM25(영)", "bm25", "en"),
    ("BM25(한)", "bm25", "ko"),
    ("RRF dense(영)+BM25(영) — 채택", "hybrid", "en"),
    ("RRF dense(한)+BM25(한)", "hybrid", "ko"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--out", type=Path, default=None, help="결과 Markdown 저장 경로")
    args = ap.parse_args()
    cfg = load_config()

    if args.benchmark:
        index = load_or_build_index(cfg, cfg["eval_papers_dir"], cfg["eval_index_dir"])
        title = "벤치마크 코퍼스 (후보 6편)"
    else:
        index = load_or_build_index(cfg)
        title = "운영 인덱스 (선정 2편)"
    docs = set(index["doc_ids"])

    lines = [f"# 검색 평가 — {title}", "",
             f"- 문서: {', '.join(sorted(docs))} ({len(index['chunks'])} 청크)",
             f"- 조건: 청크 {cfg['chunk_size']}자/겹침 {cfg['chunk_overlap']}자, {cfg['embedding_model']}, "
             f"RRF k={cfg['rrf_k']}, 검색기별 상위 {cfg['candidate_k']}개 융합, doc_id 필터, MRR@{cfg['candidate_k']}",
             "- 판정: 질의당 이진. 상위 K 안에 정답 청크(정답 문서 + 근거 정규식 일치)가 있으면 hit", ""]
    # 운영 인덱스는 문서가 2편뿐이라 필터 없는 조건이 의미가 없으므로 필터 적용만 낸다
    conditions = [(True, "doc_id 필터 적용 (운영 조건)")]
    if args.benchmark:
        conditions.insert(0, (False, "doc_id 필터 없음 (설계서 2.2.2 표 조건)"))
    for set_name, qa_path in (("dev", cfg["qa_dev"]), ("val", cfg["qa_val"])):
        qas = [q for q in load_qa(qa_path) if q["doc"] in docs]
        golds = gold_sets(qas, index["chunks"])
        found = sum(bool(g) for g in golds)
        lines += [f"## {set_name} (n={len(qas)}, 골든 근거 발견 {found}/{len(qas)})", ""]
        for use_filter, cond_label in conditions:
            lines += [f"**{cond_label}**", "", "| 검색 방식 | Hit@1 | Hit@3 | Hit@5 | MRR@10 |", "|---|---|---|---|---|"]
            for label, mode, lang in SYSTEMS:
                ranks = [rank(index, q[lang], doc_id=q["doc"] if use_filter else None, mode=mode) for q in qas]
                m = metrics(ranks, golds, top_k=cfg["candidate_k"])
                lines.append(f"| {label} | {cell(m, 'hit@1')} | {cell(m, 'hit@3')} | {cell(m, 'hit@5')} | {cell(m, 'mrr')} |")
            lines.append("")
        if missing := [q["id"] for q, g in zip(qas, golds, strict=True) if not g]:
            lines += [f"골든 근거를 찾지 못한 문항: {missing}", ""]

    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
