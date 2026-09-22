"""웹 검색 도구."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Source는 State 정의에 있다.
    from ..graph.state import Source

DEFAULT_MAX_RESULTS = 5


def web_search(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> list[Source]:
    """Tavily로 웹을 검색해 결과마다 Source를 만들어 돌려준다.

    검색 결과의 제목, 기관명 또는 작성자, 날짜, URL을 Source에 담는다.
    본문 발췌는 요약하지 않고 원문 그대로 excerpt에 담는다.

    Args:
        query: 검색 질의.
        max_results: 돌려줄 검색 결과 수.
    """
    raise NotImplementedError
