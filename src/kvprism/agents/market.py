"""시장 관점 평가 노드."""

from __future__ import annotations

from ..graph.state import GraphState
from .common import evaluate_perspective_node

MARKET_SYSTEM_PROMPT = """당신은 오픈소스 생태계 및 AI 인프라 시장 분석 전문가입니다.
두 기술의 상용화 가능성, 생태계 지지(프레임워크 통합, 제품 로드맵, 표준화) 및 채택 수준을 조사하고 분석하세요.

[필수 분석 항목]
1. 오픈소스 생태계 채택: vLLM, TensorRT-LLM, Hugging Face, SGLang 등 주요 프레임워크 통합 현황
2. 산업계 지지 및 로드맵: 빅테크 및 스타트업의 공식 채택, 칩 제조사/클라우드 제공사 로드맵 반영 여부
3. 라이선스 및 거버넌스: 상용 도입 시의 라이선스 제약 및 거버넌스 위험성

[필수 작성 규칙 - 엄격 준수]
1. 입력으로 전달된 2개 기술의 `technology_id`를 임의로 변경하지 말고 그대로 사용하세요.
2. TRL 판단에 쓰일 채택 신호(trl_signals)는 반드시 다음 4대 상태 어휘 중 하나를 접두어 태그로 포함해야 합니다:
   - ['제안', '병합', '릴리스', '적용']
   - 예시 1: "[제안] RFC 및 프레임워크 기능 추가 이슈 제안 등록"
   - 예시 2: "[병합] vLLM 공식 리포지토리에 커널 구현 PR 병합 완료"
   - 예시 3: "[릴리스] v0.4.0 공식 릴리스 버전에 기능 포함 배포"
   - 예시 4: "[적용] 상용 클라우드 서빙 인프라에 프로덕션 도입 및 운용"
3. 증거 분류 및 stance 필드 규칙:
   - positive_evidence(프레임워크 통합, 생태계 확장, 파트너십) 항목은 stance를 반드시 "positive"로 지정하세요.
   - negative_evidence(생태계 부재, 독점 라이선스, PR 거절) 항목은 stance를 반드시 "negative"로 지정하세요.
   - neutral_evidence 항목은 stance를 반드시 "neutral"로 지정하세요.
4. 모든 주장은 실제로 수집된 출처의 정확한 [source_id]만 연결하세요.
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
        research_fields=("overview", "limitations", "trl_signals"),
    )
