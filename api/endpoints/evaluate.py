from __future__ import annotations
from fastapi import APIRouter, HTTPException
from core.models import EvaluationRequest, EvaluationResult
from core.exceptions import PipelineNotFoundError, EvaluationError
from services.rag_service import get_pipeline_config
from services.document_service import load_documents
from agents.evaluation_agent import run_evaluation_agent
from utils.logger import logger
import glob, os

_evaluation_store: dict[str, EvaluationResult] = {}

router = APIRouter(prefix="/evaluate", tags=["Evaluate"])


def get_stored_evaluation(pipeline_id: str) -> EvaluationResult | None:
    return _evaluation_store.get(pipeline_id)


@router.post("/{pipeline_id}", response_model=EvaluationResult)
async def evaluate_pipeline(pipeline_id: str, request: EvaluationRequest):
    try:
        get_pipeline_config(pipeline_id)
    except PipelineNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    files = glob.glob("./data/uploads/*")
    doc_text = "No document text available."
    if files:
        latest = max(files, key=os.path.getmtime)
        try:
            pages = load_documents(latest)
            doc_text = " ".join([p.page_content for p in pages])
        except Exception:
            pass

    try:
        result = run_evaluation_agent(
            pipeline_id=pipeline_id,
            document_text=doc_text,
            num_questions=request.num_questions,
        )
        _evaluation_store[pipeline_id] = result
        return result
    except EvaluationError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation error: {e}")
