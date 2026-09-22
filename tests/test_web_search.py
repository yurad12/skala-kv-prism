"""웹 출처 식별자와 캐시 호환성 테스트."""

import json
from importlib import import_module


web_search_module = import_module("kvprism.tools.web_search")


def test_source_id_is_stable_for_same_url_and_excerpt() -> None:
    first = web_search_module._make_source_id("https://example.com/a", "같은 발췌문")
    second = web_search_module._make_source_id("https://example.com/a", "같은 발췌문")

    assert first == second


def test_source_id_differs_when_same_url_has_different_excerpt() -> None:
    first = web_search_module._make_source_id("https://example.com/a", "첫 번째 발췌문")
    second = web_search_module._make_source_id("https://example.com/a", "두 번째 발췌문")

    assert first != second


def test_cached_source_id_is_normalized_before_return(tmp_path, monkeypatch) -> None:
    cache_path = tmp_path / "cached.json"
    cache_path.write_text(
        json.dumps(
            [
                {
                    "source_id": "web_old",
                    "source_kind": "web",
                    "title": "캐시 출처",
                    "author_or_org": "Example",
                    "published_at": None,
                    "url_or_page": "https://example.com/a",
                    "excerpt": "캐시에 저장된 발췌문",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(web_search_module, "_get_cache_path", lambda query: cache_path)

    sources = web_search_module.web_search("cached query")

    assert sources[0].source_id == web_search_module._make_source_id(
        sources[0].url_or_page,
        sources[0].excerpt,
    )
