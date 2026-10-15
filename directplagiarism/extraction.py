from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from typing import Iterable

from docx import Document as DocxDocument
from pypdf import PdfReader


def normalize_whitespace(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_text_file(path: str | os.PathLike[str]) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as file_obj:
        return file_obj.read()


def read_pdf_text(path: str | os.PathLike[str]) -> str:
    text_chunks: list[str] = []
    reader = PdfReader(str(path))
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text:
            text_chunks.append(page_text)
    return "\n".join(text_chunks)


def read_docx_text(path: str | os.PathLike[str]) -> str:
    doc = DocxDocument(str(path))
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n".join(paragraphs)


def extract_text_from_file(path: str | os.PathLike[str]) -> tuple[str, str]:
    file_path = Path(path)
    suffix = file_path.suffix.lower().lstrip(".")

    if suffix == "txt":
        content = read_text_file(file_path)
        return normalize_whitespace(content), "txt"
    if suffix == "pdf":
        try:
            content = read_pdf_text(file_path)
        except Exception as exc:  # pragma: no cover - validated at runtime
            raise ValueError(f"Unable to read PDF content: {exc}") from exc
        return normalize_whitespace(content), "pdf"
    if suffix in {"docx", "doc"}:
        if suffix == "doc":
            raise ValueError("Legacy .doc files are not supported in this build. Please upload a PDF, DOCX, or TXT file.")
        content = read_docx_text(file_path)
        return normalize_whitespace(content), "docx"

    raise ValueError(f"Unsupported file type: {suffix}")


def split_sentences(text: str, minimum_length: int = 3) -> list[str]:
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip() and len(sentence.split()) >= minimum_length]


def tokenize_words(text: str) -> list[str]:
    if not text:
        return []
    normalized = unicodedata.normalize("NFKC", text).lower()
    return re.findall(r"\b\w+\b", normalized)


def build_excerpt(text: str, max_chars: int = 180) -> str:
    clean = normalize_whitespace(text)
    if not clean:
        return ""
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3].rstrip() + "..."


def is_empty_or_weak_text(text: str, min_words: int = 20) -> bool:
    words = tokenize_words(text)
    return len(words) < min_words
