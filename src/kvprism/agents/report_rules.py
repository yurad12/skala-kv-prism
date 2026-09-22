"""
@desc   : 종합 결과와 보고서에 공통으로 쓰는 표현·인용 검사와 REFERENCE 작성
"""

import re
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlparse

import yaml

from ..graph.state import Source

'''
상수 정의
'''
# 3.4의 추천·우열 금지 표현
FORBIDDEN = re.compile(
    r"더\s*우수|우수하|우수한|권장|추천|우위|우월|열등|더\s*낫|더\s*나은"
    r"|\b(?:recommend\w*|superior|inferior|better\s+than)\b",
    re.IGNORECASE,
)
# 인용 태그. 링크 표기 [제목](url)는 제외
CITATION = re.compile(r"\[([^\]\n]+)\](?!\()")
# 논문 서지 정보 출처. doc_id 기준
RAG_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "rag.yaml"
PATENT_HOSTS = ("patents.google.com", "patents.uspto.gov", "kipris.or.kr")


def check_forbidden(text: str, sources: Sequence[Source]) -> None:
    """금지 표현 검사. 원문 그대로의 이해관계자 발언만 예외."""
    for number, line in enumerate(text.splitlines(), start=1):
        # 강조 기호로 표현이 쪼개지는 경우 대비
        if not FORBIDDEN.search(re.sub(r"[*_`]", "", line)):
            continue
        if _is_quote(line, sources):
            continue
        raise ValueError(f"{number}행: 금지 표현이 있습니다 - {line.strip()}")


def cited_sources(text: str, sources: Sequence[Source]) -> list[Source]:
    """본문이 인용한 출처 목록. 등장 순서, 중복 제거."""
    by_tag: dict[str, Source] = {}
    for source in sources:
        by_tag.setdefault(source.source_id, source)
        by_tag.setdefault(source.url_or_page, source)  # [TQ p.7] 형식의 페이지 태그

    used: dict[str, Source] = {}
    for tag in CITATION.findall(text):
        source = by_tag.get(tag)
        if source is None:
            raise ValueError(f"출처 목록에 없는 인용입니다: [{tag}]")
        used.setdefault(source.source_id, source)
    return list(used.values())


def build_references(sources: Sequence[Source]) -> str:
    """REFERENCE 블록 생성. 과제 표기 형식을 출처 종류별로 적용."""
    papers = _paper_docs()
    lines = ["# REFERENCE", ""]
    for source in sources:
        if source.source_kind == "paper":
            lines.append(_paper_line(source, papers.get(source.doc_id or "", {})))
        elif any(host in source.url_or_page for host in PATENT_HOSTS):
            lines.append(_patent_line(source))
        else:
            lines.append(_web_line(source))
    return "\n".join(lines)


def _paper_docs() -> dict:
    """configs/rag.yaml의 논문 서지. 파일이 없으면 빈 값."""
    if not RAG_CONFIG.exists():
        return {}
    return yaml.safe_load(RAG_CONFIG.read_text(encoding="utf-8")).get("docs") or {}


def _paper_line(source: Source, meta: dict) -> str:
    # 논문 : 저자(YYYY). 논문제목. *학술지/학회명*, 번호.
    # 저자 뒤 소속 괄호는 연도 괄호와 겹치므로 제거
    author = re.sub(r"\s*\([^)]*\)$", "", meta.get("authors") or source.author_or_org)
    year = str(meta.get("published") or source.published_at or "")[:4] or "연도 미상"
    title = _plain(meta.get("title") or source.title)
    tail = f" *arXiv*, {meta['arxiv']}." if meta.get("arxiv") else f" {source.url_or_page}"
    return f"- [{source.source_id}] {author} ({year}). {title}.{tail}"


def _web_line(source: Source) -> str:
    # 기타 : 기관명 또는 작성자(YYYY-MM-DD). *제목*. 사이트명, URL
    date = source.published_at or "날짜 미상"
    site = urlparse(source.url_or_page).netloc.removeprefix("www.") or source.url_or_page
    return (
        f"- [{source.source_id}] {source.author_or_org} ({date}). "
        f"*{_plain(source.title)}*. {site}, {source.url_or_page}"
    )


def _patent_line(source: Source) -> str:
    # 특허 : 출원인(YYYY-MM). 특허명, 번호, URL. 번호 칸이 State에 없어 제목에 실린 값을 그대로 둔다
    date = str(source.published_at)[:7] if source.published_at else "날짜 미상"
    return f"- [{source.source_id}] {source.author_or_org} ({date}). *{_plain(source.title)}*, {source.url_or_page}"


def _plain(value: str) -> str:
    return " ".join(value.split())


def _is_quote(line: str, sources: Sequence[Source]) -> bool:
    """원문 인용 여부. `> 발언 [source_id]` 한 줄이면서 발췌에 포함될 때만 참."""
    if not line.startswith("> "):
        return False
    tag = CITATION.search(line)
    if tag is None:
        return False
    quote = line[2:tag.start()].strip()
    return any(
        source.source_id == tag.group(1) and quote and quote in source.excerpt
        for source in sources
    )
