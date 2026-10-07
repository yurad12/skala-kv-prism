"""장별 본문과 표·출처를 조립해 Markdown과 PDF를 저장한다."""

import json
import re
import os
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from ..graph.state import GraphState, Source, SynthesisResult
from ..tools.render_pdf import render_pdf
from .report_rules import CITATION, check_forbidden, cited_sources
from ..graph.supervisor_state import ReportArtifact
from .report_rules import (
    build_source_references, display_report,
)

INPUT_KEYS = ("request", "research", "market_eval", "stakeholder_eval", "domain_eval")

PROMPT = Path(__file__).resolve().parents[1] / "prompts" / "report.md"
PERSPECTIVE_LABELS = {"market": "시장성", "stakeholder": "이해관계자", "domain": "도메인 적용"}

# 설계서 5장 목차와 장별 작성 지침. 분량은 상한
CHAPTERS = (
    ("1. 분석 배경", "KV cache 병목과 비교 배경, 평가 도메인과 시나리오. 700자 이내."),
    ("2. 기술 선정", "선정 기술 2건과 두 접근의 비교 축. 선정 방식은 코드가 앞에 붙인다. 선정 사유는 입력 research와 sources의 실제 근거를 인용해 작성한다. 주어지지 않은 선정 점수나 채택 사실은 쓰지 않는다. 600자 이내."),
    ("3. 기술 개요", "기술별 개요, 동작 방식, 성능 개선과 손실, 실험 조건, 한계. performance_metrics의 대표 개선 결과도 포함하고 배수·백분율마다 비교 기준을 적는다. 원문에서 확인한 조건만 함께 제시한다. 2200자 이내."),
    ("4. 관점별 평가", "시장: 공개 채택·제품화·생태계 신호와 그 확인 범위. 이해관계자: 공급자·운영자·사용자의 기대와 부담, 입장이 갈리는 이유. 도메인: 데이터센터 장문맥 시나리오에 실험 조건을 적용할 수 있는 범위와 운영 조건. 각 관점을 소제목으로 구분한다. 3장의 성능 수치와 동작 설명은 반복하지 않는다. 매트릭스와 같은 근거를 사용하되 본문에는 판단의 이유를 쓴다. 1600자 이내."),
    ("5. 시사점", "기술별로 '어떤 병목·운영 조건이면 검토 대상이 되는가 → 제공된 근거가 뒷받침하는 범위 → 적용 전에 확인할 조건'을 조건부 문장으로 쓴다. 병용은 상호작용 근거가 있을 때만 다룬다. 기술 간 우열이나 도입 결정을 내리지 않는다. 앞 장의 결과·수치·미검증 목록을 재요약하지 않는다. 상충 지점은 뒤에 붙으므로 반복하지 않는다. 700자 이내."),
    ("6. 한계점", "## 공개 정보 기반 판단의 한계 와 ## 해석 시 주의점 두 소제목으로 쓴다. 앞에는 결론을 제한하는 공통 미검증 조건을 한 번만 모은다. 뒤에는 논문 자기 평가와 외부 검증·채택 신호의 차이, 서로 다른 환경과 비교 기준을 섞지 않는 해석 원칙을 설명한다. 실제 수행 여부가 확인되지 않은 검증 조치를 수행했다고 쓰지 않는다. 프로그램 내부 절차는 설명하지 않는다. 700자 이내."),
)
# 선정 방식은 사용자 결정이며, 기술적 선정 근거는 조사 결과에서 작성한다.
SELECTION_REASON = """## 선정 방식

SW·HW 분야에서 조가 직접 1건씩 선정했다.
평가 범위는 데이터센터의 장문맥 멀티테넌트 LLM 추론이다."""

SUMMARY_GUIDE = (
    "이 요약만 읽는 사람을 위해 쓴다. 도입 문단이나 목차 안내 없이 다음 세 항목만 작성한다. "
    "각 항목은 '## 1. 분석 대상과 범위', '## 2. 핵심 판단', '## 3. 적용 전 확인사항'이라는 "
    "소제목과 그 아래 설명으로 구성한다. 소제목과 설명 사이, 항목 사이에 빈 줄을 둔다. "
    "1번에는 두 기술이 다루는 병목과 평가 시나리오를 2문장 이내로 쓴다. "
    "2번에는 기술별로 '**기술명**: 설명' 형식의 짧은 문단을 하나씩 쓴다. "
    "각 기술의 검토 조건과 공개 근거가 뒷받침하는 범위를 설명하고, 성숙도 근거가 부족하면 판정 불가라고 쓴다. "
    "3번에는 판단에 영향을 주는 미확인 조건을 최대 세 개의 짧은 글머리표로 쓴다. "
    "각 문단은 2문장 이내로 쓰고 상세 성능 수치와 본문의 한계 목록을 반복하지 않는다. "
    "기술 간 우열이나 도입 결정을 내리지 않는다. 전체 800자 이내."
)


def report_node(state: GraphState, *, llm=None, output_dir: str | Path = "outputs") -> dict:
    """담당 키 report만 반환. 입력 State 불변."""
    synthesis = state["synthesis"]
    names = {tech.technology_id: tech.name for tech in state["request"].technologies}
    context = {key: state[key].model_dump(mode="json") for key in (*INPUT_KEYS, "synthesis")}
    context["instruction"] = state["decision"].instruction if state.get("decision") else ""
    context["sources"] = [source.model_dump(mode="json") for source in state["sources"]]
    if llm is None:
        load_dotenv()
        # 설계서 2.4의 Generator 설정. 추론 모델이라 temperature 미지정
        llm = ChatOpenAI(model=os.getenv("GENERATOR_MODEL") or "gpt-5.6-luna", reasoning_effort="medium", timeout=120, max_retries=1)
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
        text = _strip_title(write(chapter, guide, "\n\n".join(chapters)), chapter)
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
    # 검증용 원문에는 청크 ID를 보존하고 표시용 문서에서만 페이지 태그를 쓴다.
    references = cited_sources(markdown, state["sources"])
    source_markdown = markdown + build_source_references(references)
    markdown = display_report(source_markdown, state["sources"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = output_dir / "report.source.md"
    markdown_path = output_dir / "report.md"
    source_path.write_text(source_markdown, encoding="utf-8")
    markdown_path.write_text(markdown, encoding="utf-8")
    pdf_path = render_pdf(markdown, output_dir / "RAG-OUTPUT.pdf")
    return {"report": ReportArtifact(
        source_markdown_path=source_path,
        markdown_path=markdown_path,
        pdf_path=pdf_path,
        reference_source_ids=[source.source_id for source in references],
    )}



def _trl_table(synthesis: SynthesisResult, names: dict[str, str]) -> str:
    lines = [
        "## 기술 성숙도", "",
        "TRL은 제공된 공개 근거에서 확인되는 검증 단계의 추정이다. 논문 기술과 기반 기술을 구분하며, 근거가 부족하면 판정 불가로 표시한다.", "",
        "| 기술 | 논문 기술 TRL | 기반 기술 TRL |",
        "| --- | --- | --- |",
    ]
    details = []
    for item in synthesis.trl_estimates:
        paper_basis, separator, enabling_basis = item.rationale.partition("기반 기술:")
        paper_basis = paper_basis.removeprefix("논문 기술:").strip()
        bases = (paper_basis, enabling_basis.strip() if separator else "판정 불가")
        scores = (item.paper_trl, item.enabling_technology_trl)
        labels = ("논문 기술", "기반 기술")
        displayed = []
        explanations = []
        for label, score, basis in zip(labels, scores, bases):
            # 근거 부족을 나타내는 내부 호환값은 독자에게 점수로 제시하지 않는다.
            unknown = any(term in basis for term in ("판정 불가", "스키마", "호환용 최소", "schema"))
            displayed.append("판정 불가" if unknown else str(score))
            if unknown:
                citations = " ".join(f"[{source_id}]" for source_id in CITATION.findall(basis))
                basis = f"판정 불가: 검증 단계를 판단할 근거가 부족하다. {citations}".strip()
            explanations.append(f"{label}: {basis}")
        lines.append(f"| {names[item.technology_id]} | {displayed[0]} | {displayed[1]} |")
        explanation = "\n\n".join(explanations)
        details.append(
            f"**{names[item.technology_id]} — 판단 근거**\n\n"
            f"{explanation} {_tags(item.source_ids, explanation)}"
        )
    return "\n".join(lines) + "\n\n" + "\n\n".join(details)


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
