from __future__ import annotations
from fastapi import APIRouter, HTTPException
from core.models import DiagnosisReport
from agents.retrieval_inspector import run_retrieval_inspector
from agents.hallucination_detector import run_hallucination_detector
from api.endpoints.evaluate import get_stored_evaluation
from utils.logger import logger

_diagnosis_store: dict[str, DiagnosisReport] = {}

router = APIRouter(prefix="/diagnose", tags=["Diagnose"])


def get_stored_diagnosis(pipeline_id: str) -> DiagnosisReport | None:
    return _diagnosis_store.get(pipeline_id)


@router.post("/{pipeline_id}", response_model=DiagnosisReport)
async def diagnose_pipeline(pipeline_id: str):
    evaluation = get_stored_evaluation(pipeline_id)
    if not evaluation:
        raise HTTPException(status_code=404, detail=f"No evaluation for '{pipeline_id}'. Run /evaluate first.")

    try:
        chunking, retrieval, ranking, context_util = run_retrieval_inspector(evaluation)
        hallucination = run_hallucination_detector(evaluation.per_query_results)

        report = DiagnosisReport(
            pipeline_id=pipeline_id,
            chunking=chunking,
            retrieval=retrieval,
            ranking=ranking,
            context_utilization=context_util,
            hallucination=hallucination,
        )
        _diagnosis_store[pipeline_id] = report
        return report
    except Exception as e:
        logger.error(f"Diagnosis failed for '{pipeline_id}': {e}")
        raise HTTPException(status_code=500, detail=f"Diagnosis error: {e}")
