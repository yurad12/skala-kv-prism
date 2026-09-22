"""
@desc   : 보고서 생성 노드. 장별 본문에 표·REFERENCE를 붙여 Markdown·PDF 저장
"""

import json
import re
import os
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from ..graph.state import GraphState, ReportResult, Source, SynthesisResult
from ..tools.render_pdf import render_pdf
from .report_rules import CITATION, build_references, check_forbidden, cited_sources
from .synthesize import INPUT_KEYS

PROMPT = Path(__file__).resolve().parents[1] / "prompts" / "report.md"
PERSPECTIVE_LABELS = {"market": "시장성", "stakeholder": "이해관계자", "domain": "도메인 적용"}

# 설계서 5장 목차와 장별 작성 지침. 분량은 상한
CHAPTERS = (
    ("1. 분석 배경", "KV cache 병목과 비교 배경, 평가 도메인과 시나리오. 700자 이내."),
    ("2. 기술 선정", "선정 기술 2건과 두 접근의 비교 축. 선정 사유는 코드가 앞에 붙이므로 반복하지 않는다. 주어지지 않은 선정 점수는 쓰지 않는다. 600자 이내."),
    ("3. 기술 개요", "기술별 개요, 동작 방식, 성능 주장, 실험 조건, 한계. 2200자 이내."),
    ("4. 관점별 평가", "시장·이해관계자·도메인 관점의 평가. 뒤에 붙는 TRL 표와 매트릭스는 반복하지 않는다. 2500자 이내."),
    ("5. 시사점", "synthesis.neutral_summary를 바탕으로 관점 간 일치·보완 관계와 병용 조건, 확인되지 않은 부분. 상충 지점은 뒤에 붙으므로 반복하지 않는다. 900자 이내."),
    ("6. 한계점", "## 공개 정보 기반 추정의 한계 와 ## 확증편향 방지 조치 두 소제목으로 쓴다. "
     "앞에는 공개 자료·실험 조건의 한계와 judge를 통과하지 못한 관점을 쓴다. "
     "뒤에는 이 파이프라인이 실제로 적용한 조치를 쓴다: 관점별 긍정·부정 근거 균형 검사, "
     "주장과 출처 발췌문의 일치 판정, 미달 관점 1회 재실행, 등록된 출처 ID만 인용 허용, "
     "논문만 근거일 때 TRL 3 상한과 논문 기술·기반 기술 분리 추정, 우열 판정 표현 필터. "
     "이 조치를 설명할 때도 금지 표현 단어 자체는 쓰지 않는다. 900자 이내."),
)
# 2안(Human 기반) 선정 사유. 코드가 2장 앞에 그대로 붙인다
SELECTION_REASON = """## 선정 방식과 사유

Doc Pool의 SW·HW 진영에서 조가 직접 1건씩 선정했다(2안, Human 기반).

- **SW: TurboQuant** — 이미 만들어진 KV cache를 생성 중 온라인으로 양자화하는 사후 압축 기법으로, 모델 재학습 없이 기존 GPU 서빙에 적용 경로가 있다. 아키텍처 변경이 필요한 MLA(DeepSeek-V2)나 베이스라인 성격의 KIVI보다 "데이터를 작게 만드는" 접근의 현재 채택 흐름을 살피기에 적합하다고 판단했다.
- **HW: ITME** — CXL-hybrid memory와 RDMA로 GPU 서버 밖의 원격 메모리 계층을 확장하는 구조로, 평가 도메인인 데이터센터 장문맥 멀티테넌트 추론과 직접 맞닿는다. 단일 서버 호스트 오프로딩(InfiniGen)이나 전용 PIM 하드웨어보다 "담을 공간을 넓히는" 접근의 데이터센터 적용 조건을 살피기에 적합하다고 판단했다."""

SUMMARY_GUIDE = (
    "이 요약만 읽는 사람을 위해 쓴다. 보고서 소개나 목차 안내를 쓰지 않는다. "
    "비교한 두 기술, 관점에 따라 평가가 갈리는 지점, 기술 성숙도 추정과 그 근거 범위, "
    "아직 확인되지 않아 판단 전에 확인해야 할 조건을 담는다. 800자 이내."
)


def report_node(state: GraphState, *, llm=None, output_dir: str | Path = "outputs") -> dict:
    """담당 키 report만 반환. 입력 State 불변."""
    synthesis = state["synthesis"]
    names = {tech.technology_id: tech.name for tech in state["request"].technologies}
    context = {key: state[key].model_dump(mode="json") for key in (*INPUT_KEYS, "synthesis")}
    context["sources"] = [source.model_dump(mode="json") for source in state["sources"]]
    if llm is None:
        load_dotenv()
        # 설계서 2.4의 Generator 설정. 추론 모델이라 temperature 미지정
        llm = ChatOpenAI(model=os.getenv("GENERATOR_MODEL") or "gpt-5.6-luna", reasoning_effort="medium")
    prompt = PROMPT.read_text(encoding="utf-8")

    def write(chapter: str, guide: str, body: str = "") -> str:
        """장별 본문 생성. 검사에 걸리면 무엇이 틀렸는지 알려주고 한 번 다시 요청."""
        request = {
            "chapter": chapter, "guide": guide, "body": body, "context": context,
            # 인용에 쓸 수 있는 ID. 목록 밖의 태그는 검사에서 걸린다
            "source_ids": [source.source_id for source in state["sources"]],
        }
        for retried in (False, True):
            response = llm.invoke([
                ("system", prompt),
                ("human", json.dumps(request, ensure_ascii=False)),
            ])
            text = response.content
            try:
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("생성된 본문이 비어 있습니다")
                check_forbidden(text, state["sources"])
                cited_sources(text, state["sources"])
                return text.strip()
            except ValueError as error:
                if retried:
                    raise ValueError(f"{chapter}: {error}") from None
                request["fix"] = f"직전 초안의 문제: {error}. 같은 문제를 반복하지 않는다."

    chapters = []
    for chapter, guide in CHAPTERS:
        text = _strip_title(write(chapter, guide), chapter)
        if chapter.startswith("2."):
            text = SELECTION_REASON + "\n\n" + text
        elif chapter.startswith("4."):
            text += "\n\n" + _trl_table(synthesis, names) + "\n\n" + _matrix_table(synthesis, names)
        elif chapter.startswith("5."):
            text += "\n\n" + _conflicts(synthesis)
        chapters.append(f"# {chapter}\n\n{text}")

    body = "\n\n".join(chapters)
    request = state["request"]
    title = (
        "# KV cache 최적화 기술 다관점 평가 보고서\n\n"
        f"**{' vs '.join(names.values())}** · {request.domain} {request.scenario} · SKALA 9반 6조\n\n"
    )
    markdown = f"{title}# SUMMARY\n\n{write('SUMMARY', SUMMARY_GUIDE, body)}\n\n{body}\n\n"
    markdown = _page_tags(markdown, state["sources"])
    references = cited_sources(markdown, state["sources"])
    markdown += build_references(references) + "\n"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = render_pdf(markdown, output_dir / "RAG-Output.pdf")
    markdown_path = output_dir / "report.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    return {"report": ReportResult(
        markdown_path=markdown_path,
        pdf_path=pdf_path,
        reference_source_ids=[source.source_id for source in references],
    )}


def _trl_table(synthesis: SynthesisResult, names: dict[str, str]) -> str:
    lines = [
        "## 기술 성숙도", "",
        "| 기술 | 논문 기술 TRL | 기반 기술 TRL | 근거 |",
        "| --- | --- | --- | --- |",
    ]
    for item in synthesis.trl_estimates:
        # 논문 기술·기반 기술 두 줄을 한 칸 안에서 구분
        rationale = " / ".join(line for line in item.rationale.splitlines() if line.strip())
        lines.append(
            f"| {names[item.technology_id]} | {item.paper_trl} | {item.enabling_technology_trl} | "
            f"{_cell(rationale)} {item.disclosure} {_tags(item.source_ids, rationale)} |"
        )
    return "\n".join(lines)


def _matrix_table(synthesis: SynthesisResult, names: dict[str, str]) -> str:
    technology_ids = list(names)
    lines = [
        "## 관점 × 기술 매트릭스", "",
        "| 관점 | " + " | ".join(names[tech_id] for tech_id in technology_ids) + " |",
        "| --- | --- | --- |",
    ]
    cells = {(cell.technology_id, cell.perspective): cell for cell in synthesis.matrix}
    for perspective, label in PERSPECTIVE_LABELS.items():
        row = [label]
        for tech_id in technology_ids:
            cell = cells[tech_id, perspective]
            row.append(f"{_cell(cell.assessment)} {_tags(cell.source_ids, cell.assessment)}".rstrip())
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _conflicts(synthesis: SynthesisResult) -> str:
    return "\n\n".join(
        f"## 상충 지점 {number}: {item.title}\n\n{item.description} {_tags(item.source_ids, item.description)}".rstrip()
        for number, item in enumerate(synthesis.conflicts, start=1)
    )


def _page_tags(markdown: str, sources: Sequence[Source]) -> str:
    """논문 청크 ID 인용을 [TQ p.7] 페이지 태그로 바꾼다."""
    for source in sources:
        if source.source_kind == "paper":
            markdown = markdown.replace(f"[{source.source_id}]", source.url_or_page)
    return markdown


def _cell(text: str) -> str:
    """표 한 칸용 정리. 줄바꿈·칸 구분 기호 제거."""
    return " ".join(text.split()).replace("|", "/")


def _tags(source_ids: list[str], text: str = "") -> str:
    """본문에 아직 없는 출처만 인용 태그로 덧붙인다."""
    cited = set(CITATION.findall(text))
    return " ".join(f"[{source_id}]" for source_id in dict.fromkeys(source_ids) if source_id not in cited)


def _strip_title(text: str, chapter: str) -> str:
    """LLM이 본문 앞에 다시 쓴 장 제목 줄을 지운다. 예: '# 1. 분석 배경', '## 한계점'."""
    name = re.sub(r"^\d+\.\s*", "", chapter)
    lines = text.splitlines()
    while lines and (not lines[0].strip() or re.sub(r"^#+\s*(\d+\.\s*)?", "", lines[0].strip()) in (chapter, name)):
        lines.pop(0)
    return "\n".join(lines).strip()
