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
        # [TQ p.7] 형식의 페이지 태그. 조회 키는 대괄호를 뺀 값이다
        by_tag.setdefault(source.url_or_page.strip("[]"), source)

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
    listed_pages: set[tuple[str | None, str]] = set()
    listed_urls: set[str] = set()
    for source in sources:
        if source.source_kind == "paper":
            # 본문에서 인용한 페이지마다 서지를 남기고, 같은 페이지의 청크만 합친다.
            page_key = (source.doc_id, source.url_or_page)
            if page_key in listed_pages:
                continue
            listed_pages.add(page_key)
            lines.append(_paper_line(source, papers.get(source.doc_id or "", {})))
        elif source.url_or_page in listed_urls:
            continue
        elif any(host in source.url_or_page for host in PATENT_HOSTS):
            lines.append(_patent_line(source))
        else:
            lines.append(_web_line(source))
        if source.source_kind != "paper":
            listed_urls.add(source.url_or_page)
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
    # 본문과 같은 페이지 태그를 사용해 인용 위치를 바로 확인할 수 있게 한다.
    label = source.url_or_page.strip("[]") or source.source_id
    return f"- [{label}] {author} ({year}). {title}.{tail}"


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
    # 검색 결과 제목 끝에 붙는 말줄임표를 없앤다
    return " ".join(value.split()).rstrip(". ").removesuffix("..").strip()


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


# 원문 인용을 보존하는 Supervisor 보고서 계약
def build_source_references(sources: Sequence[Source]) -> str:
    """검증용 원본에는 페이지로 합치기 전 모든 인용 ID를 남긴다."""
    return "# REFERENCE\n\n" + "\n".join(
        f"- [{s.source_id}] {s.author_or_org}. {s.title}. {s.url_or_page}"
        for s in sources
    ) + "\n"


def display_report(source_markdown: str, sources: Sequence[Source]) -> str:
    """같은 본문에 표시용 페이지 태그와 중복 제거된 서지만 적용한다."""
    body = source_markdown.split("# REFERENCE", 1)[0]
    references = cited_sources(body, sources)
    web_labels = {}
    for source in references:
        if source.source_kind == "paper":
            body = body.replace(f"[{source.source_id}]", source.url_or_page)
        else:
            # 같은 URL의 여러 발췌문은 참고문헌에 남는 첫 ID로 연결한다.
            label = web_labels.setdefault(source.url_or_page, source.source_id)
            body = body.replace(f"[{source.source_id}]", f"[{label}]")
    return body + build_references(references) + "\n"


def validate_report_artifact(report, sources: Sequence[Source]) -> int:
    """인용 ID·참고문헌·표시용 변환을 확인하고 실제 PDF 페이지 수를 반환한다."""
    import pymupdf

    original = report.source_markdown_path.read_text(encoding="utf-8")
    if original.count("# REFERENCE") != 1 or "# SUMMARY" not in original:
        raise ValueError("SUMMARY와 REFERENCE가 필요합니다")
    body, refs = original.split("# REFERENCE", 1)
    registered = {s.source_id for s in sources}
    tags = set(CITATION.findall(body))
    if not tags or tags - registered:
        raise ValueError("본문에는 등록된 원문 source_id 인용이 필요합니다")
    ref_ids = set(re.findall(r"(?m)^- \[([^\]]+)\]", refs))
    if tags != set(report.reference_source_ids) or tags != ref_ids:
        raise ValueError("본문 인용·REFERENCE·reference_source_ids가 일치하지 않습니다")
    if report.markdown_path.read_text(encoding="utf-8") != display_report(original, sources):
        raise ValueError("표시용 Markdown이 근거 원본과 다릅니다")
    with pymupdf.open(report.pdf_path) as pdf:
        if not len(pdf) or not any(page.get_text().strip() for page in pdf):
            raise ValueError("PDF 내용이 비어 있습니다")
        return len(pdf)


# 보고서 작성과 품질 평가에서 공유하는 방법론
REPORT_METHODOLOGY = (
    "구현 절차: Supervisor는 등록된 발췌문과 관점 결과로 다음 작업을 선택한다. "
    "종합·보고서는 등록된 출처 ID와 작성자의 우열 표현을 코드로 검사한다. "
    "보고서 작성 후 별도 품질 노드가 근거성·중립성·편향·관점 충족·형식을 검사한다. "
    "이는 검사 절차 설명이며, 현재 보고서의 통과나 모든 주장에 대한 보증은 아니다."
)
TRL_GUIDE = (
    "TRL은 출처에 보고된 점수가 아니라 공개 근거를 다음 기준에 대응한 작성자의 추정이다. "
    "1~2: 개념·원리, 3: 논문 내 실험·프로토타입, 4: 외부 재현·통합 테스트, "
    "5: 관련 환경 구성요소 검증, 6: 관련 환경 시스템 시연, 7: 서비스·파일럿, "
    "8: 양산 적합성 검증, 9: 상용 운용. "
    "근거가 없는 항목은 판정 불가로 표시하며, 제안 시스템과 기반 구성요소의 성숙도를 구분한다."
)
