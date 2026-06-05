"""
Document ingestion service.
Handles PDF, TXT, and DOCX files.
Extracts text with metadata (source, page number) for context traceability.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import IO

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

from core.exceptions import DocumentProcessingError
from core.models import UploadedDocument
from utils.logger import logger

SUPPORTED_TYPES = {".pdf", ".txt", ".docx"}


def _load_pdf(path: str) -> list[Document]:
    loader = PyPDFLoader(path)
    return loader.load()


def _load_txt(path: str) -> list[Document]:
    loader = TextLoader(path, encoding="utf-8")
    return loader.load()


def _load_docx(path: str) -> list[Document]:
    loader = Docx2txtLoader(path)
    return loader.load()


def load_documents(file_path: str) -> list[Document]:
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext not in SUPPORTED_TYPES:
        raise DocumentProcessingError(
            f"Unsupported file type '{ext}'. Supported: {SUPPORTED_TYPES}"
        )

    try:
        if ext == ".pdf":
            docs = _load_pdf(file_path)
        elif ext == ".txt":
            docs = _load_txt(file_path)
        elif ext == ".docx":
            docs = _load_docx(file_path)
        else:
            raise DocumentProcessingError(f"Unhandled extension: {ext}")

        for doc in docs:
            doc.metadata.setdefault("source", path.name)

        logger.info(f"Loaded {len(docs)} pages/sections from '{path.name}'")
        return docs

    except DocumentProcessingError:
        raise
    except Exception as exc:
        raise DocumentProcessingError(
            f"Failed to process '{path.name}': {exc}"
        ) from exc


def save_uploaded_file(file_obj: IO[bytes], filename: str, upload_dir: str = "./data/uploads") -> str:
    os.makedirs(upload_dir, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex}_{Path(filename).name}"
    dest = os.path.join(upload_dir, safe_name)
    with open(dest, "wb") as f:
        f.write(file_obj.read() if hasattr(file_obj, "read") else file_obj)
    logger.info(f"Saved uploaded file to '{dest}'")
    return dest


def build_uploaded_document(
    doc_id: str,
    filename: str,
    ext: str,
    pages: list[Document],
    chunk_count: int,
) -> UploadedDocument:
    return UploadedDocument(
        doc_id=doc_id,
        filename=filename,
        file_type=ext.lstrip("."),
        page_count=len(pages),
        chunk_count=chunk_count,
    )
