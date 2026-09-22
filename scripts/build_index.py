"""운영 인덱스(Chroma + BM25)를 만든다. 이미 있으면 건너뛴다.

  uv run python scripts/build_index.py            # data/papers → outputs/index
  uv run python scripts/build_index.py --force    # 다시 만들기
  uv run python scripts/build_index.py --eval     # 벤치마크 코퍼스 data/papers/eval → outputs/index_eval
"""

import argparse
import time

from kvprism.rag import load_config, load_or_build_index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--eval", action="store_true", help="벤치마크용 6편 코퍼스를 인덱싱한다")
    args = ap.parse_args()
    cfg = load_config()
    papers_dir, index_dir = (cfg["eval_papers_dir"], cfg["eval_index_dir"]) if args.eval else (cfg["papers_dir"], cfg["index_dir"])
    t0 = time.time()
    index = load_or_build_index(cfg, papers_dir, index_dir, force=args.force)
    print(f"{index_dir}: 문서 {index['doc_ids']}, 청크 {len(index['chunks'])}개, {time.time() - t0:.1f}초")


if __name__ == "__main__":
    main()
