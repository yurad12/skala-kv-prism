"""configs/rag.yaml 을 읽는다. 경로 값은 레포 루트 기준 절대 경로로 바꿔 준다."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]  # 레포 루트
PATH_KEYS = ("papers_dir", "index_dir", "eval_papers_dir", "eval_index_dir", "qa_dev", "qa_val")


def load_config() -> dict:
    cfg = yaml.safe_load((ROOT / "configs" / "rag.yaml").read_text())
    for key in PATH_KEYS:
        cfg[key] = ROOT / cfg[key]
    return cfg


def tag_of(cfg: dict, doc_id: str) -> str:
    """인용 태그 접두어. rag.yaml 에 없는 문서는 doc_id 대문자 (예: kivi → KIVI)."""
    return cfg["docs"].get(doc_id, {}).get("tag", doc_id.upper())
