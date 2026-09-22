"""
@file   : report_rules.py
@writer : 오승민
@date   : 2026-09-22
@desc   : 종합 결과와 보고서에 공통으로 쓰는 표현·인용 검사와 REFERENCE 작성
"""

import re
from collections.abc import Sequence

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
    """REFERENCE 블록 생성."""
    lines = ["# REFERENCE", ""]
    for source in sources:
        date = source.published_at or "날짜 미상"
        title = " ".join(source.title.split())
        lines.append(
            f"- [{source.source_id}] {source.author_or_org} ({date}). {title}. {source.url_or_page}"
        )
    return "\n".join(lines)


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
