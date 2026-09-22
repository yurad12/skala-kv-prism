"""이해관계자 관점 평가 노드."""

from __future__ import annotations

from ..graph.state import GraphState
from .common import evaluate_perspective_node

STAKEHOLDER_SYSTEM_PROMPT = """당신은 AI/반도체 산업 투자자 및 경쟁사 관점의 이해관계자 분석 전문가입니다.
두 기술에 대한 투자 업계, 경쟁 진영, 도입 기업의 입장을 긍정·부정·유보 관점으로 나누어 분석하세요.

[필수 분석 항목]
1. 투자 및 자본 시장: 해당 기술 또는 관련 스타트업에 대한 VC/빅테크 투자, 펀딩, 밸류에이션 평가
2. 경쟁 진영의 대응: 대체 기술 진영(예: 타 양자화 방식, HBM 진영 등)의 비판, 반론, 벤치마크 반박
3. 도입 의사결정권자(엔지니어링 리드): 도입 위험 대비 기대 효익에 대한 유보적/회의적 시각

[필수 작성 규칙 - 엄격 준수]
1. 입력으로 전달된 2개 기술의 `technology_id`를 임의로 변경하지 말고 그대로 사용하세요.
2. TRL 판단에 쓰일 채택 신호(trl_signals)는 반드시 다음 4대 상태 어휘 중 하나를 접두어 태그로 포함해야 합니다:
   - ['제안', '병합', '릴리스', '적용']
   - 예시 1: "[제안] 투자 유치 제안서 및 기술 백서 공개"
   - 예시 2: "[병합] 파트너십 기반 공동 솔루션 브랜치 병합"
   - 예시 3: "[릴리스] 엔터프라이즈 솔루션 파트너사 대상 베타 릴리스"
   - 예시 4: "[적용] 글로벌 빅테크 데이터센터 프로덕션 환경 실제 적용"
3. 증거 분류 및 stance 필드 규칙:
   - positive_evidence 항목은 stance를 반드시 "positive"로 지정하세요.
   - negative_evidence 항목은 stance를 반드시 "negative"로 지정하세요.
   - neutral_evidence 항목은 stance를 반드시 "neutral"로 지정하세요.
4. 모든 주장은 실제로 수집된 출처의 정확한 [source_id]만 연결하세요.
"""


def _get_stakeholder_queries(tech_name: str, domain: str, scenario: str) -> list[str]:
    return [
        f"{tech_name} developer adoption feedback benchmark challenges",
        f"{tech_name} competitors alternative solutions industry response",
        f"{tech_name} hardware investment analyst TCO evaluation",
    ]


def stakeholder_node(state: GraphState) -> dict:
    """이해관계자 관점 평가를 수행하고 stakeholder_eval과 신규 출처를 반환합니다."""
    return evaluate_perspective_node(
        state=state,
        perspective="stakeholder",
        system_prompt=STAKEHOLDER_SYSTEM_PROMPT,
        query_fn=_get_stakeholder_queries,
        use_domain_context=False,
        research_fields=("overview", "limitations"),
    )
