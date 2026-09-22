"""웹 검색 도구."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from pathlib import Path
from urllib.parse import urlparse
from tavily import TavilyClient

from ..graph.state import Source


from dotenv import load_dotenv

load_dotenv(override=True)

DEFAULT_MAX_RESULTS = 5
CACHE_DIR = Path("outputs/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _get_cache_path(query: str) -> Path:
    query_hash = hashlib.md5(query.strip().lower().encode("utf-8")).hexdigest()
    return CACHE_DIR / f"search_{query_hash}.json"


def _parse_published_date(date_str: str | None) -> str | None:
    """Tavily 날짜 문자열을 YYYY-MM-DD 포맷으로 정제합니다."""
    if not date_str:
        return None
    try:
        # ISO 형식(YYYY-MM-DD...)에서 날짜 부분만 추출
        clean_date = date_str[:10]
        date.fromisoformat(clean_date)
        return clean_date
    except Exception:
        return None


def web_search(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> list[Source]:
    """Tavily로 웹을 검색해 결과마다 Source를 만들어 돌려준다."""
    cache_path = _get_cache_path(query)

    # 1. 로컬 캐시 조회
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
        return [Source(**item) for item in cached_data]

    # 2. Tavily 검색 실행
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = client.search(
        query=query,
        max_results=max_results,
        include_raw_content=False,
    )

    # 3. Source 인스턴스 생성 및 캐시 저장
    sources: list[Source] = []
    cache_items: list[dict] = []

    for item in response.get("results", []):
        raw_url = item.get("url", "").strip()
        if not raw_url:
            continue

        domain = urlparse(raw_url).netloc.replace("www.", "").replace("m.", "")
        
        # URL 고유 해시로 ID 생성 (동일 URL에 대한 일관성 보장)
        url_hash = hashlib.md5(raw_url.encode("utf-8")).hexdigest()[:8]
        source_id = f"web_{url_hash}"

        # min_length=1 제약 조건 방어
        title = item.get("title", "").strip() or "Untitled Web Source"
        excerpt = item.get("content", "").strip() or title
        author_or_org = domain.split(".")[0].capitalize() if domain else "Web"
        published_at = _parse_published_date(item.get("published_date"))

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