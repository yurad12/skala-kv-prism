"""
@desc   : 보고서 Markdown의 A4 PDF 변환
"""

from pathlib import Path

import pymupdf
from markdown_pdf import MarkdownPdf, Section

REPORT_CSS = """
body { font-family: sans-serif; font-size: 10pt; line-height: 1.5; }
h1 { font-size: 16pt; margin-top: 18pt; margin-bottom: 9pt; }
h2 { font-size: 12pt; margin-top: 12pt; margin-bottom: 6pt; }
p { margin: 6pt 0; }
table { width: 100%; border-collapse: collapse; font-size: 9pt; }
th, td { border: 0.5pt solid #aaa; padding: 5pt; }
blockquote { border-left: 2pt solid #aaa; padding-left: 8pt; }
a { color: #245680; }
"""
MARGIN = 42


def render_pdf(report_md: str, output_path: str | Path = "outputs/RAG-Output.pdf") -> Path:
    """A4 PDF 저장. SUMMARY가 반 페이지 초과 시 중단."""
    pdf = MarkdownPdf(toc_level=2)
    # 웹 발췌에 섞인 HTML·이미지 차단
    pdf.m_d.disable("html_block").disable("html_inline").disable("image")

    page = pymupdf.paper_rect("a4")
    summary = report_md.split("\n# ", 1)[0]
    story = pymupdf.Story(html=pdf.m_d.render(summary), user_css=REPORT_CSS)
    overflow, _ = story.place(pymupdf.Rect(MARGIN, MARGIN, page.width - MARGIN, page.height / 2))
    if overflow:
        raise ValueError("SUMMARY가 A4 반 페이지를 넘습니다")

    pdf.meta["title"] = "KV cache 최적화 기술 다관점 평가"
    pdf.meta["author"] = "SKALA 9반 6조"
    pdf.add_section(
        Section(report_md, paper_size="A4", borders=(MARGIN, MARGIN, -MARGIN, -MARGIN)),
        user_css=REPORT_CSS,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.save(output_path)
    return output_path
