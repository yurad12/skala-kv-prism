"""웹 검색 도구."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import date
from email.utils import parsedate_to_datetime
from typing import Any
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv(override=True)

from ..graph.state import Source

CACHE_DIR = Path("outputs/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# 주요 테크 도메인 기관명 매핑 테이블
KNOWN_DOMAINS: dict[str, str] = {
    "huggingface.co": "Hugging Face",
    "github.com": "GitHub",
    "arxiv.org": "arXiv",
    "qdrant.tech": "Qdrant",
    "vllm.ai": "vLLM",
    "openai.com": "OpenAI",
    "google.com": "Google",
    "research.google": "Google Research",
    "microsoft.com": "Microsoft",
    "meta.com": "Meta",
    "nvidia.com": "NVIDIA",
    "amd.com": "AMD",
    "intel.com": "Intel",
    "semianalysis.com": "SemiAnalysis",
    "anandtech.com": "AnandTech",
    "tomshardware.com": "Tom's Hardware",
    "medium.com": "Medium",
}


def _get_cache_path(query: str, max_results: int) -> Path:
    key=f"{query.strip()}_{max_results}"
    query_hash = hashlib.md5(key.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"search_{query_hash}.json"

def _extract_author_or_org(url: str) -> str:
    """URL에서 올바른 기관명 또는 루트 도메인을 추출합니다."""
    netloc = urlparse(url).netloc.lower().replace("www.", "").replace("m.", "")
    for domain_key, org_name in KNOWN_DOMAINS.items():
        if domain_key in netloc:
            return org_name
    
    parts = netloc.split(".")
    if len(parts) >= 2:
        return f"{parts[-2]}.{parts[-1]}"
    return netloc or "Web"


def _parse_published_date(raw_date: Any, url:str) -> str | None:
    """Tavily 날짜 문자열을 YYYY-MM-DD 포맷으로 정제합니다."""
    if raw_date and isinstance(raw_date, str):
        # 1. RFC 2822 형식 (예: "Wed, 02 Oct 2002 13:00:00 GMT")
        try:
            dt = parsedate_to_datetime(raw_date)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            pass

        # 2. ISO 8601 형식 (예: "2023-08-15T14:30:00Z") 및 정규식 매칭
        match = re.search(r"(\d{4})[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])", raw_date)
        if match:
            return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"

    # 3. Fallback: URL 경로 내 날짜 패턴 (/2026/03/26/ 등)
    url_match = re.search(r"/(202[0-9])[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])", url)
    if url_match:
        return f"{url_match.group(1)}-{url_match.group(2)}-{url_match.group(3)}"

    return None


def web_search(query: str, max_results: int = 5, topic:str="general") -> list[Source]:
    """Tavily로 웹을 검색해 결과마다 Source를 만들어 돌려준다. 로컬 캐시를 우선 확인한다."""
    cache_path = _get_cache_path(query, max_results)

    # 1. 로컬 캐시 조회
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
        return [Source(**item) for item in cached_data]

    # 2. Tavily 검색 실행
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return []
    client = TavilyClient(api_key=api_key)
    try:
        response = client.search(
            query=query,
            max_results=max_results,
            topic=topic,
            include_published_date=True,
            include_raw_content=False,
        )
    except Exception:
        return []

    # 3. Source 인스턴스 생성 및 캐시 저장
    sources: list[Source] = []
    cache_items: list[dict] = []

    for item in response.get("results", []):
        raw_url = item.get("url", "").strip()
        if not raw_url:
            continue

        url_hash = hashlib.md5(raw_url.encode("utf-8")).hexdigest()[:8]
        source_id = f"web_{url_hash}"

        title = item.get("title", "").strip() or "Untitled Web Source"
        excerpt = item.get("content", "").strip() or title
        
        # 수정: 기관명 추출 함수 및 날짜 파싱 함수 정상 연동
        author_or_org = _extract_author_or_org(raw_url)
        published_at = _parse_published_date(item.get("published_date"), raw_url)

        data = {
            "source_id": source_id,
            "source_kind": "web",
            "title": title,
            "author_or_org": author_or_org,
            "published_at": published_at,
            "url_or_page": raw_url,
            "excerpt": excerpt,
        }

        # Pydantic 엄격 검증 통과 후 추가
        source_obj = Source(**data)
        sources.append(source_obj)
        cache_items.append(data)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_items, f, ensure_ascii=False, indent=2)

    return sources