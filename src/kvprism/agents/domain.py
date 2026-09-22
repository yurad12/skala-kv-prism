"""도메인 적용 관점 평가 노드."""

from __future__ import annotations

from ..graph.state import GraphState, Source
from ..tools.rag_retrieve import rag_retrieve
from .common import evaluate_perspective_node

DOMAIN_SYSTEM_PROMPT = """당신은 LLM 인프라 배포 환경 및 도메인 적용성 분석 전문가입니다.
동일한 기술이라도 서비스가 배포되는 운영 환경(도메인)의 물리적 제약에 따라 실효성과 평가가 완전히 달라집니다.
두 기술에 대해 3대 핵심 도메인 환경에서의 적합성과 물리적 한계를 심층 조사하세요.

[필수 분석 도메인]
1. 데이터센터 / 클라우드: 대규모 동시 요청(Concurrent Batch/Throughput), 인프라 TCO 및 서빙 원가 민감도
2. On-Device AI: 극심한 메모리(LPDDR) 용량 한계, 모바일/엣지 기기의 저전력 및 발열 제약
3. 장문맥 처리 어플리케이션: 100k~1M+ 초장문 문맥 길이 처리, TTFT 및 생성 지연 시간 민감도

[필수 작성 규칙 - 엄격 준수]
1. 입력으로 전달된 2개 기술의 `technology_id`(예: 소문자 ID)를 임의로 대소문자나 명칭을 바꾸지 말고 그대로 일치시켜 작성하세요.
2. TRL 판단에 쓰일 채택 신호(trl_signals)에는 반드시 상태 어휘인 ['제안', '병합', '릴리스', '적용'] 중 하나를 명시하세요.
3. 증거 분류 및 stance 필드 규칙을 엄격히 준수하세요:
   - positive_evidence(도메인 적합성, 처리량 개선, 저전력/경량화 이점) 항목은 stance를 반드시 "positive"로 지정하세요.
   - negative_evidence(도메인 환경 제약에 따른 부적합, 지연 발생, 물리적 도입 불가) 항목은 stance를 반드시 "negative"로 지정하세요.
   - neutral_evidence 항목은 stance를 반드시 "neutral"로 지정하세요.
4. 특정 도메인에서는 최적이나 타 도메인에서는 물리적 적용이 불가능한 상충 관계(예: 온디바이스에서 CXL 적용 불가, 초장문맥에서 단순 양자화 노이즈 누적 등)를 명확히 기술하세요.
5. 모든 주장은 실제로 수집된 출처의 정확한 [source_id]만 연결해야 하며, 가상의 source_id를 생성하지 마세요.
"""


def _get_domain_queries(tech_name: str, domain: str, scenario: str) -> list[str]:
    queries = [
        f"{tech_name} datacenter cloud serving throughput TCO benchmark",
        f"{tech_name} on-device mobile edge memory power constraint",
        f"{tech_name} long-context 100k 1M tokens TTFT latency evaluation",
    ]
    if domain:
        queries.append(f"{tech_name} {domain} {scenario} performance comparison")
    return queries


def domain_node(state: GraphState) -> dict:
    """도메인 평가 노드: 논문 RAG 청크를 검색해 웹 검색 결과와 함께 평가합니다."""
    # 1. 도메인 평가에 필요한 논문 실험 조건 RAG 검색
    rag_sources: list[Source] = []
    for tech in state.get("selected_techs", []):
        doc_id = getattr(tech, "paper_doc_id", tech.technology_id)
        queries = [
            f"{tech.name} experimental setup hardware GPU memory batch size",
            f"{tech.name} throughput latency serving benchmark",
        ]
        for q in queries:
            try:
                rag_sources.extend(rag_retrieve(doc_id=doc_id, query=q, k=2))
            except Exception:
                pass

    # 2. 웹 검색 및 LLM 평가는 기존 common 함수에 위임 (extra_sources로 RAG 청크 전달)
    return evaluate_perspective_node(
        state=state,
        perspective="domain",
        system_prompt=DOMAIN_SYSTEM_PROMPT,
        query_fn=_get_domain_queries,
        extra_sources=rag_sources,
        use_domain_context=True,
    )