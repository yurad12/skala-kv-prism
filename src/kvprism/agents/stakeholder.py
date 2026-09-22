"""이해관계자 관점 평가 노드."""

from __future__ import annotations

from ..graph.state import GraphState
from .common import evaluate_perspective_node

STAKEHOLDER_SYSTEM_PROMPT = """당신은 LLM 인프라 및 가속 시스템 산업 전략 분석 전문가입니다.
두 기술에 대해 3대 핵심 이해관계자(경쟁 기술 진영, 도입 기업 및 개발자, 투자 업계)의 반응과 입장을 심층 조사하세요.

[필수 분석 항목]
1. 경쟁 기술 진영: 경쟁사/대체 기술 진영의 반응 및 대응 기술 동향 (예: GPU 벤더, 타 압축/오프로딩 기법 진영)
2. 도입 기업 및 개발자: 실무 서빙 엔지니어 및 개발사의 도입 선호도, 기존 시스템 통합 시 직면하는 기술적/운영적 진입 장벽
3. 투자 업계: 반도체/인프라 펀드의 투자 동향, 테크 애널리스트 및 전문 미디어의 상업적 생존 가능성 및 TCO 평가

[필수 작성 규칙]
1. 반드시 입력된 2개 기술(technology_id) 각각에 대해 summary와 evidence를 작성하세요.
2. positive_evidence(개발 편의성, 생태계 호평, 투자 유치)와 negative_evidence(도입 장벽, 고비용 설비 투자, 경쟁 기술 대비 열세)를 균형 있게 구분하세요.
3. 이해관계자 간 상충 관계(예: 모델 코드를 수정하지 않으려는 개발자의 요구 vs 하드웨어 설비 투자(CAPEX)를 줄이려는 인프라 운영팀의 입장)를 명확히 짚어내세요.
4. 모든 주장은 수집된 출처의 정확한 [source_id]를 연결하세요.
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
    )