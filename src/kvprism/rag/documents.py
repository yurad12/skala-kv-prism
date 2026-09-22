"""PDF → 페이지 Document → 800자 청크 Document (설계 문서 2.2).

Document.metadata 에는 doc_id(PDF 파일명), page(1부터), 청크에는 chunk_id("turboquant:p7:2") 가 들어간다.
"""

import re
import unicodedata
from pathlib import Path

import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean(text: str) -> str:
    """전처리: NFKC 정규화(합자 ﬁ 풀기), 줄 끝 하이픈 분철 복원, 공백 정리."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"-\n(?=[a-z])", "", text)
    return re.sub(r"\s+", " ", text).strip()


def load_pages(papers_dir: Path) -> list[Document]:
    """papers_dir 바로 아래의 PDF 를 PyMuPDF 로 읽어 페이지마다 Document 하나를 만든다."""
    pages = []
    for pdf in sorted(Path(papers_dir).glob("*.pdf")):
        with pymupdf.open(pdf) as doc:
            for page_no, page in enumerate(doc, start=1):
                pages.append(Document(page_content=clean(page.get_text()),
                                      metadata={"doc_id": pdf.stem, "page": page_no}))
    return pages


def split_pages(pages: list[Document], chunk_size: int = 800, chunk_overlap: int = 100) -> list[Document]:
    """페이지 안에서 800자/겹침 100자로 자른다. 페이지 경계를 넘지 않아 청크마다 page 가 하나다."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for page in pages:
        if not page.page_content:
            continue
        for n, piece in enumerate(splitter.split_text(page.page_content)):
            chunk_id = f"{page.metadata['doc_id']}:p{page.metadata['page']}:{n}"
            chunks.append(Document(page_content=piece, metadata={**page.metadata, "chunk_id": chunk_id}))
    return chunks
