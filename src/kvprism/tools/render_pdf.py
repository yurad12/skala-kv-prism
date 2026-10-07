"""표지와 본문을 분리해 A4 보고서 PDF를 만든다."""

import re
from pathlib import Path

import pymupdf
from markdown_pdf import MarkdownPdf, Section

FONT_DIR = Path(__file__).resolve().parents[1] / "assets/fonts"
FONT_CSS = """
@font-face { font-family: KVPrismSans; src: url(KVPrismSans-Regular.ttf); }
@font-face { font-family: KVPrismSans; src: url(KVPrismSans-Bold.ttf); font-weight: bold; }
"""
REPORT_CSS = FONT_CSS + """
body { font-family: KVPrismSans; font-size: 10pt; line-height: 1.55; color: #303236; }
h1 { font-size: 17pt; color: #25272a; margin: 22pt 0 11pt; page-break-after: avoid; }
h2 { font-size: 12pt; color: #414449; margin: 15pt 0 8pt; page-break-after: avoid; }
h3 { font-size: 10.5pt; margin: 12pt 0 6pt; page-break-after: avoid; }
p { margin: 7pt 0; }
li { margin: 5pt 0; }
table { width: 100%; border-collapse: collapse; font-size: 8.5pt; line-height: 1.45; margin: 10pt 0; }
th { font-weight: bold; color: #25272a; }
th, td { border: 0.5pt solid #d5d7da; padding: 7pt; vertical-align: top; }
blockquote { border-left: 2pt solid #a7abb0; padding-left: 10pt; color: #55585d; }
a { color: #414449; }
"""
COVER_CSS = FONT_CSS + """
body { font-family: KVPrismSans; color: #303236; font-size: 12pt; line-height: 1.65; }
h1 { font-size: 32pt; line-height: 1.3; margin: 0 0 28pt; }
h2 { font-size: 20pt; margin: 0 0 25pt; }
p { margin: 10pt 0; }
"""
MARGIN = 54


def number_web_references(report_md: str) -> str:
    """본문과 참고문헌의 웹 출처 ID를 같은 번호로 표시한다."""
    body, marker, references = report_md.partition("# REFERENCE")
    if not marker:
        raise ValueError("REFERENCE가 없습니다")
    pattern = r"\[(web_[0-9a-f]+)\]"
    used = set(re.findall(pattern, body))
    listed = list(dict.fromkeys(re.findall(pattern, references)))
    missing = used - set(listed)
    if missing:
        raise ValueError(f"본문 출처가 REFERENCE에 없습니다: {sorted(missing)}")
    labels = {source_id: str(index) for index, source_id in enumerate(listed, start=1)}
    return re.sub(pattern, lambda match: f"[{labels[match.group(1)]}]", report_md)


def validate_summary(summary: str) -> None:
    """현재 글꼴·여백에서 SUMMARY가 A4 반 페이지 안에 들어가는지 확인한다."""
    renderer = MarkdownPdf(toc_level=2)
    renderer.m_d.disable("html_block").disable("html_inline").disable("image")
    page = pymupdf.paper_rect("a4")
    story = pymupdf.Story(
        html=renderer.m_d.render(summary), user_css=REPORT_CSS, archive=str(FONT_DIR),
    )
    overflow, _ = story.place(pymupdf.Rect(MARGIN, MARGIN, page.width - MARGIN, page.height / 2))
    if overflow:
        raise ValueError("SUMMARY가 A4 반 페이지를 넘습니다. 문단 구성은 유지하고 각 문단을 한두 문장으로 줄이세요")


def render_pdf(report_md: str, output_path: str | Path = "outputs/RAG-Output.pdf") -> Path:
    """표지를 포함한 PDF를 저장하고 SUMMARY의 반 페이지 제한을 검사한다."""
    for name in ("KVPrismSans-Regular.ttf", "KVPrismSans-Bold.ttf"):
        if not (FONT_DIR / name).is_file():
            raise FileNotFoundError(f"보고서 글꼴이 없습니다: {name}")
    pdf = MarkdownPdf(toc_level=2)
    # 수집 자료에서 들어온 HTML과 외부 이미지는 렌더링하지 않는다.
    pdf.m_d.disable("html_block").disable("html_inline").disable("image")
    report_md = number_web_references(report_md)
    cover, marker, content = report_md.partition("# SUMMARY")
    if not marker:
        raise ValueError("SUMMARY가 없습니다")
    body, _, references = (marker + content).partition("# REFERENCE")
    summary = body.split("\n# ", 1)[0]
    validate_summary(summary)

    cover = re.sub(r"\*\*(.+?)\*\*", r"\n## \1\n", cover)
    cover = cover.replace(" · ", "\n\n").strip()
    pdf.meta["title"] = "KV cache 최적화 기술 다관점 평가"
    pdf.meta["author"] = "SKALA 9반 6조"
    pdf.add_section(
        Section(cover, toc=False, root=str(FONT_DIR), borders=(MARGIN, 172, -MARGIN, -MARGIN)),
        user_css=COVER_CSS,
    )
    if pdf.sections[0].page_count != 1:
        raise ValueError("표지 내용이 한 페이지를 넘습니다")
    pdf.add_section(
        Section(body, root=str(FONT_DIR), borders=(MARGIN, MARGIN, -MARGIN, -MARGIN)),
        user_css=REPORT_CSS,
    )
    pdf.add_section(
        Section("# REFERENCE" + references, root=str(FONT_DIR),
                borders=(MARGIN, MARGIN, -MARGIN, -MARGIN)),
        user_css=REPORT_CSS + "body { font-size: 8.5pt; }",
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 원본 파일과 교체하기 전 임시 PDF에 페이지 장식을 적용한다.
    temporary = output_path.with_suffix(".layout.pdf")
    pdf.save(temporary)
    try:
        with pymupdf.open(temporary) as document:
            charcoal = (0.18, 0.19, 0.20)
            light_gray = (0.82, 0.83, 0.85)
            for index, sheet in enumerate(document):
                if index == 0:
                    sheet.draw_rect(pymupdf.Rect(MARGIN, 103, MARGIN + 44, 108), color=charcoal, fill=charcoal)
                    sheet.insert_text((MARGIN, 87), "KV PRISM / TECHNICAL REPORT", fontsize=9, color=charcoal)
                    continue
                sheet.draw_line((MARGIN, 30), (sheet.rect.width - MARGIN, 30), color=light_gray, width=0.5)
                sheet.insert_text((MARGIN, 24), "KV PRISM / TURBOQUANT + ITME", fontsize=7, color=charcoal)
                sheet.insert_text((sheet.rect.width - MARGIN - 26, sheet.rect.height - 25),
                                  f"{index:02d}", fontsize=9, color=charcoal)
            document.save(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return output_path
