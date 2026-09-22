"""시장 관점 평가 노드."""

from __future__ import annotations

from ..graph.state import GraphState
from .common import evaluate_perspective_node

MARKET_SYSTEM_PROMPT = """당신은 LLM 서빙 및 인프라 시장 분석 전문가입니다.
두 기술의 상용화·채택 사례와 생태계 지지(프레임워크 통합, 제품 로드맵, 표준화)를 조사하세요.

[필수 작성 규칙]
1. 반드시 입력된 2개 기술(technology_id) 각각에 대해 summary와 evidence를 작성하세요.
2. TRL 판단에 쓰일 채택 신호(trl_signals)에는 반드시 상태 어휘인 ['제안', '병합', '릴리스', '적용'] 중 하나를 명시하세요.
   - 예시: "vLLM 공식 저장소에 TurboQuant 양자화 커널 PR 병합 완료"
3. positive_evidence(상용화 기대, 생태계 통합)와 negative_evidence(도입 지연, 대체 기술 대비 열세)를 균형 있게 구분하세요.
4. 모든 주장은 수집된 출처의 정확한 [source_id]를 연결하세요.
"""


def _get_market_queries(tech_name: str, domain: str, scenario: str) -> list[str]:
    return [
        f"{tech_name} vLLM GitHub support framework integration",
        f"{tech_name} commercial product roadmap deployment 2026",
    ]


def market_node(state: GraphState) -> dict:
    """시장 관점 평가를 수행하고 market_eval과 신규 출처를 반환합니다."""
    return evaluate_perspective_node(
        state=state,
        perspective="market",
        system_prompt=MARKET_SYSTEM_PROMPT,
        query_fn=_get_market_queries,
        use_domain_context=False,
    )