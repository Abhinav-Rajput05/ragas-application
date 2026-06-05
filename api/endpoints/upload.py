from __future__ import annotations
import os
import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from core.models import UploadResponse
from core.exceptions import DocumentProcessingError
from services.rag_service import ingest_document
from utils.logger import logger

router = APIRouter(prefix="/upload", tags=["Upload"])

UPLOAD_DIR = "./data/uploads"
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}


@router.post("/", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload a document and build the initial RAG pipeline."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type {ext!r}. Allowed: {ALLOWED_EXTENSIONS}",
        )

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    save_path = os.path.join(UPLOAD_DIR, safe_name)

    try:
        contents = await file.read()
        with open(save_path, "wb") as fh:
            fh.write(contents)
        logger.info(f"Saved upload: {save_path}")

        uploaded_doc, pipeline_config = ingest_document(save_path)

        return UploadResponse(
            doc_id=uploaded_doc.doc_id,
            pipeline_id=pipeline_config.pipeline_id,
            message=f"Document ingested successfully.",
            chunk_count=uploaded_doc.chunk_count,
        )

    except DocumentProcessingError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")
