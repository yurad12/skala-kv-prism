"""URL 본문 추출과 요약 도구."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import date
from pathlib import Path
from urllib.parse import urlparse
import urllib.request
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..graph.state import Source

load_dotenv(override=True)

CACHE_DIR = Path("outputs/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _get_cache_path(url: str) -> Path:
    url_hash = hashlib.md5(url.strip().encode("utf-8")).hexdigest()
    return CACHE_DIR / f"fetch_{url_hash}.json"


def _extract_page_info(url: str, timeout: int = 10) -> tuple[str, str, str | None, str]:
    """웹 페이지에서 (title, author_or_org, published_at, text_content)를 추출합니다."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    )
    
    with urllib.request.urlopen(req, timeout=timeout) as response:
        html = response.read().decode("utf-8", errors="ignore")

    soup = BeautifulSoup(html, "html.parser")

    # 1. 제목 추출
    title = soup.title.string.strip() if soup.title and soup.title.string else "Untitled Source"

    # 2. 기관/작성자 (도메인 기반)
    domain = urlparse(url).netloc.replace("www.", "").replace("m.", "")
    author_or_org = domain.split(".")[0].capitalize() if domain else "Web"

    # 3. 발행일 메타 태그 추출 시도
    published_at: str | None = None
    meta_date = (
        soup.find("meta", property="article:published_time")
        or soup.find("meta", attrs={"name": "date"})
        or soup.find("meta", attrs={"name": "pubdate"})
    )
    if meta_date and meta_date.get("content"):
        content_val = str(meta_date["content"])[:10]
        try:
            date.fromisoformat(content_val)
            published_at = content_val
        except Exception:
            published_at = None

    # 4. 불필요한 태그 제거 후 본문 텍스트 추출
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    raw_text = " ".join(soup.stripped_strings)
    # 연속 공백 정리
    cleaned_text = re.sub(r"\s+", " ", raw_text).strip()

    return title, author_or_org, published_at, cleaned_text


def fetch_and_summarize(url: str) -> tuple[str, Source]:
    """페이지 본문을 추출해 요약문과 Source를 돌려준다.

    Source에는 기관명 또는 작성자, 날짜, 제목, URL을 담아 REFERENCE 형식을 맞춘다.
    excerpt에는 요약문이 아니라 추출한 원문의 일부를 그대로 담는다.
    """
    cache_path = _get_cache_path(url)

    # 1. 로컬 캐시 조회 (Cache-Aside)
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["summary"], Source(**data["source"])

    # 2. 본문 및 메타데이터 스크래핑
    try:
        title, author_or_org, published_at, full_text = _extract_page_info(url)
    except Exception as e:
        # 스크래핑 실패 시 최소 데이터로 방어
        title = "Failed to fetch page"
        author_or_org = "Web"
        published_at = None
        full_text = f"Error extracting page: {e}"

    # excerpt: 요약문이 아니라 추출한 원문의 일부를 담음 (최대 1000자)
    excerpt = full_text[:1000].strip() or title
    url_hash = hashlib.md5(url.strip().encode("utf-8")).hexdigest()[:8]
    source_id = f"web_{url_hash}"

    source = Source(
        source_id=source_id,
        source_kind="web",
        title=title,
        author_or_org=author_or_org,
        published_at=published_at,
        url_or_page=url,
        excerpt=excerpt,
    )

    # 3. LLM 요약문 생성
    model_name = os.getenv("OPENAI_MODEL_NAME") or os.getenv("GENERATOR_MODEL") or "gpt-4o-mini"
    llm = ChatOpenAI(model=model_name, temperature=0)

    # 프롬프트 변수 오작동 방지를 위한 메시지 객체 직접 주입
    context_text = full_text[:4000]
    messages = [
        SystemMessage(content=(
            "당신은 기술 보고서 분석 전문가입니다. "
            "주어진 본문에서 핵심 기술 내용, 주장, 실측 지표, 장단점을 3줄 이내로 명확하게 요약하세요."
        )),
        HumanMessage(content=f"웹페이지 본문:\n{context_text}"),
    ]

    try:
        res = llm.invoke(messages)
        summary = str(res.content).strip()
    except Exception:
        summary = excerpt[:200]

    # 4. 캐시 저장
    cache_data = {
        "summary": summary,
        "source": source.model_dump(),
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)

    return summary, source