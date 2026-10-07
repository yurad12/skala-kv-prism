"""캐시 적중 시 외부 검색을 반복하지 않는지 검증한다."""

from importlib import import_module
from unittest.mock import Mock


def test_second_search_uses_cache(tmp_path, monkeypatch):
    module = import_module("kvprism.tools.web_search")
    client = Mock()
    client.search.return_value = {"results": [{
        "url": "https://example.com/paper", "title": "공개 자료", "content": "원문 근거",
    }]}
    monkeypatch.setenv("TAVILY_API_KEY", "test-only")
    monkeypatch.setattr(module, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(module, "TavilyClient", lambda **kwargs: client)
    first = module.web_search("테스트 질의", max_results=2)
    second = module.web_search("테스트 질의", max_results=2)
    assert first == second
    client.search.assert_called_once()
