"""arXiv 에서 문서 풀 PDF 를 내려받는다.

  uv run python scripts/download_papers.py           # 운영 2편 (data/papers/)
  uv run python scripts/download_papers.py --all     # 벤치마크 6편 (data/papers/eval/) 까지

PDF 는 git 에 올리지 않는다 (.gitignore).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import httpx

from kvprism.rag import load_config

OPERATIONAL = ("turboquant", "itme")


def download(arxiv_id: str, dest: Path) -> None:
    if dest.exists():
        print(f"있음      {dest}")
        return
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    with httpx.stream("GET", url, follow_redirects=True, timeout=60) as r:
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    print(f"내려받음  {dest}  ({dest.stat().st_size // 1024} KB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="벤치마크용 6편까지 내려받는다")
    args = ap.parse_args()
    cfg = load_config()
    targets = list(cfg["docs"]) if args.all else list(OPERATIONAL)
    for doc_id in targets:
        base = cfg["eval_papers_dir"] if args.all else cfg["papers_dir"]
        download(cfg["docs"][doc_id]["arxiv"], base / f"{doc_id}.pdf")
    if args.all:
        for doc_id in OPERATIONAL:
            src, dst = cfg["eval_papers_dir"] / f"{doc_id}.pdf", cfg["papers_dir"] / f"{doc_id}.pdf"
            if not dst.exists():
                shutil.copy(src, dst)
                print(f"복사      {dst}")


if __name__ == "__main__":
    main()
