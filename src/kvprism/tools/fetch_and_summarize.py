"""URL 본문 추출과 요약 도구."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Source는 State 정의에 있다.
    from ..graph.state import Source


def fetch_and_summarize(url: str) -> tuple[str, Source]:
    """페이지 본문을 추출해 요약문과 Source를 돌려준다.

    Source에는 기관명 또는 작성자, 날짜, 제목, URL을 담아 REFERENCE 형식을 맞춘다.
    excerpt에는 요약문이 아니라 추출한 원문의 일부를 그대로 담는다.

    Args:
        url: 본문을 가져올 페이지 주소.

    Returns:
        (요약문, Source) 한 쌍.
    """
    raise NotImplementedError
