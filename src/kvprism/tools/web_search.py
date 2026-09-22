"""웹 검색 도구."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlparse
from tavily import TavilyClient

from ..graph.state import Source

DEFAULT_MAX_RESULTS = 5
CACHE_DIR = Path("outputs/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _get_cache_path(query: str) -> Path:
    query_hash = hashlib.md5(query.strip().lower().encode("utf-8")).hexdigest()
    return CACHE_DIR / f"search_{query_hash}.json"


def web_search(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> list[Source]:
    """Tavily로 웹을 검색해 결과마다 Source를 만들어 돌려준다."""
    cache_path = _get_cache_path(query)

    # 1. 캐시 확인
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
        return [Source(**item) for item in cached_data]

    # 2. Tavily 검색 실행
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = client.search(
        query=query,
        max_results=max_results,
        include_raw_content=False,  # 원문 HTML 제외, 발췌(content)만 수집
    )

    # 3. Source 인스턴스 생성 및 캐시 저장
    sources: list[Source] = []
    cache_items: list[dict] = []

    for idx, item in enumerate(response.get("results", []), start=1):
        url = item.get("url", "")
        domain = urlparse(url).netloc.replace("www.", "").replace("m.", "")
        source_id = f"web_{hashlib.md5(url.encode()).hexdigest()[:6]}_{idx}"

        data = {
            "source_id": source_id,
            "source_kind": "web",
            "title": item.get("title", ""),
            "author_or_org": domain.split(".")[0].capitalize() if domain else "Web",
            "url_or_page": url,
            "excerpt": item.get("content", ""),
        }
        sources.append(Source(**data))
        cache_items.append(data)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_items, f, ensure_ascii=False, indent=2)

    return sources


if __name__ == "__main__":
    res = web_search("TurboQuant vLLM", max_results=1)
    print("성공:", res)