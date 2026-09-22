"""
@desc   : 보고서 생성 노드. 장별 본문에 표·REFERENCE를 붙여 Markdown·PDF 저장
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from ..graph.state import GraphState, ReportResult, SynthesisResult
from ..tools.render_pdf import render_pdf
from .report_rules import build_references, check_forbidden, cited_sources
from .synthesize import INPUT_KEYS

PROMPT = Path(__file__).resolve().parents[1] / "prompts" / "report.md"
PERSPECTIVE_LABELS = {"market": "시장성", "stakeholder": "이해관계자", "domain": "도메인 적용"}

# 설계서 5장 목차와 장별 작성 지침. 분량은 상한
CHAPTERS = (
    ("1. 분석 배경", "KV cache 병목과 비교 배경, 평가 도메인과 시나리오. 700자 이내."),
    ("2. 기술 선정", "선정 기술 2건과 두 접근의 비교 축. 주어지지 않은 선정 점수는 쓰지 않는다. 700자 이내."),
    ("3. 기술 개요", "기술별 개요, 동작 방식, 성능 주장, 실험 조건, 한계. 2200자 이내."),
    ("4. 관점별 평가", "시장·이해관계자·도메인 관점의 평가. 뒤에 붙는 TRL 표와 매트릭스는 반복하지 않는다. 2500자 이내."),
    ("5. 시사점", "관점 간 일치·보완 관계와 병용 조건. 상충 지점은 뒤에 붙으므로 반복하지 않는다. 700자 이내."),
    ("6. 한계점", "공개 자료의 한계, 실험 조건 차이, 검증에서 통과하지 못한 관점과 편향 방지 조치. 700자 이내."),
)
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
        text = write(chapter, guide)
        if chapter.startswith("4."):
            text += "\n\n" + _trl_table(synthesis, names) + "\n\n" + _matrix_table(synthesis, names)
        elif chapter.startswith("5."):
            text += "\n\n" + synthesis.neutral_summary + "\n\n" + _conflicts(synthesis)
        chapters.append(f"# {chapter}\n\n{text}")

    body = "\n\n".join(chapters)
    markdown = f"# SUMMARY\n\n{write('SUMMARY', SUMMARY_GUIDE, body)}\n\n{body}\n\n"
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
            f"{_cell(rationale)} {item.disclosure} {_tags(item.source_ids)} |"
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
            row.append(f"{_cell(cell.assessment)} {_tags(cell.source_ids)}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _conflicts(synthesis: SynthesisResult) -> str:
    return "\n\n".join(
        f"## 상충 지점 {number}: {item.title}\n\n{item.description}\n\n{_tags(item.source_ids)}"
        for number, item in enumerate(synthesis.conflicts, start=1)
    )


def _cell(text: str) -> str:
    """표 한 칸용 정리. 줄바꿈·칸 구분 기호 제거."""
    return " ".join(text.split()).replace("|", "/")


def _tags(source_ids: list[str]) -> str:
    return " ".join(f"[{source_id}]" for source_id in dict.fromkeys(source_ids))
