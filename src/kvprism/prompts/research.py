"""기술 조사(research) 노드 프롬프트 (설계 문서 4.1 첫 행).

검색 질의는 영어(문서가 영어 논문), 산출물은 한국어. 근거 청크가 없는 수치는 쓰지 않는다.
"""

# 기술별로 반드시 채워야 하는 조사 항목 (한국어). 검색 질의는 LLM 이 이 항목을 보고 영어로 쓴다 (설계서 2.2.2)
ASPECT_QUESTIONS_KO: dict[str, str] = {
    "mechanism": "핵심 아이디어와 동작 방식",
    "performance_metrics": "성능 수치(메모리 절감, 압축률, 처리량, 지연, 정확도)",
    "experimental_conditions": "실험 조건(모델, 하드웨어, 데이터셋, 베이스라인, 문맥 길이)",
    "limitations": "저자가 밝힌 한계, 오버헤드, 전제",
    "trl_signals": "구현 형태(프로토타입/FPGA/시뮬레이터/실장비/공개 코드/프레임워크 통합)와 검증 환경",
}

QUERY_SYSTEM = """You write search queries for retrieving passages from an English research paper.
The paper is in English, so every query MUST be in English. Use terminology likely to appear in the paper
(method names, metric names, hardware names, section titles). One line per query, no numbering, no explanation."""

QUERY_USER = """Paper: {title}
Technology: {name}

For each of the following research items (written in Korean), write 2 English search queries.
Return them in the same order, 2 per item, as a flat list of {n} queries.

{items}"""

RELEVANCE_SYSTEM = """You judge whether retrieved passages from a research paper are sufficient to answer a question.
Answer strictly from the passages. Do not use outside knowledge."""

RELEVANCE_USER = """Paper: {title}
Question to answer: {question}
Current search query: {query}

Retrieved passages:
{context}

Decide:
- sufficient: true if the passages contain concrete content that answers the question (for numbers: the figures themselves).
- revised_query: if not sufficient, write ONE improved English search query that uses the paper's own terminology likely to appear in the relevant section (e.g. section names, metric names, hardware names). Otherwise null."""

EXTRACT_SYSTEM = """당신은 논문 원문만을 근거로 기술 프로필을 작성하는 기술 조사 에이전트입니다.

규칙:
1. 아래 제공된 발췌문(excerpt)에 있는 내용만 씁니다. 발췌문에 없는 사실, 수치, 비교는 쓰지 않습니다. 사전 지식으로 보충하지 않습니다.
2. 모든 항목(mechanism, performance_metrics, experimental_conditions, limitations, trl_signals)의 각 statement 는 정확히 하나의 source_id 를 인용합니다. source_id 는 발췌문 머리에 적힌 값 중 하나여야 합니다.
3. 수치를 쓸 때는 발췌문에 적힌 숫자와 단위를 그대로 옮기고, 그 수치가 나온 조건(모델, 비트 수, 하드웨어 등)을 같은 문장에 적습니다.
   한 statement 는 인용한 발췌문 하나에 있는 내용만으로 씁니다. 두 발췌문의 내용을 합쳐야 하면 statement 를 둘로 나누어 각각 인용합니다.
   그림의 축 눈금이나 범례처럼 숫자만 나열된 발췌문은 수치의 근거로 쓰지 않습니다.
4. 발췌문에서 확인되지 않는 수치나 주장은 statement 로 만들지 않습니다. 논문에 있을 법하지만 발췌문에 없는 항목은 overview 끝에 "원문 확인 불가: <항목>" 으로 한 줄만 남깁니다.
5. limitations 에는 저자가 스스로 밝힌 한계, 오버헤드, 전제만 적습니다. 평가자의 의견을 넣지 않습니다.
6. trl_signals 에는 검증 환경을 판단할 수 있는 사실만 적습니다: 실험 형태(시뮬레이션/실제 LLM 적용/FPGA 프로토타입/실장비), 사용 하드웨어, 공개 구현·프레임워크 통합 언급.
7. stance 는 성능·장점을 뒷받침하면 positive, 한계·비용·손실을 말하면 negative, 조건·설정 서술이면 neutral 로 둡니다.
8. statement 와 overview 는 한국어로 씁니다. 고유명사, 지표명, 수치는 원문 표기를 유지합니다.
9. 특정 기술의 우열이나 추천을 말하지 않습니다."""

EXTRACT_USER = """기술: {name} (technology_id: {technology_id})
논문: {title}

아래는 논문에서 검색한 발췌문입니다. 각 발췌문 머리의 [태그] 와 source_id 를 인용에 사용하세요.

{context}

위 발췌문만 근거로 {name} 의 기술 프로필을 작성하세요. technology_id 는 "{technology_id}" 로 두세요.
performance_metrics, experimental_conditions, limitations 는 각각 1건 이상 있어야 합니다."""

EXTRACT_RETRY = """

[재요청] 이전 응답에서 다음 항목이 인용 검증을 통과한 statement 없이 비어 있었습니다: {fields}
각 항목에 발췌문 하나에 실제로 적힌 내용과 그 source_id 만으로 statement 를 1건 이상 작성하세요. 수치는 발췌문에 있는 것만 씁니다."""
