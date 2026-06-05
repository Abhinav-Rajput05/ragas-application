from __future__ import annotations
from fastapi import APIRouter, HTTPException
from core.models import FinalReport
from agents.report_generator import build_final_report
from api.endpoints.evaluate import get_stored_evaluation
from api.endpoints.diagnose import get_stored_diagnosis
from api.endpoints.optimize import get_stored_optimization, get_stored_cost_accuracy
from utils.logger import logger

router = APIRouter(prefix="/report", tags=["Report"])

_report_store: dict[str, FinalReport] = {}


def get_stored_report(pipeline_id: str) -> FinalReport | None:
    return _report_store.get(pipeline_id)


@router.post("/{pipeline_id}", response_model=FinalReport)
async def generate_report(pipeline_id: str, pipeline_name: str = "RAG Pipeline"):
    evaluation = get_stored_evaluation(pipeline_id)
    if not evaluation:
        raise HTTPException(status_code=404, detail="Run /evaluate first.")

    diagnosis = get_stored_diagnosis(pipeline_id)
    if not diagnosis:
        raise HTTPException(status_code=404, detail="Run /diagnose first.")

    optimization = get_stored_optimization(pipeline_id)
    if not optimization:
        raise HTTPException(status_code=404, detail="Run /optimize first.")

    cost_accuracy = get_stored_cost_accuracy(pipeline_id)
    if not cost_accuracy:
        raise HTTPException(status_code=404, detail="Cost accuracy data missing.")

    try:
        report = build_final_report(
            pipeline_id=pipeline_id,
            pipeline_name=pipeline_name,
            evaluation=evaluation,
            diagnosis=diagnosis,
            optimization=optimization,
            cost_accuracy=cost_accuracy,
        )
        _report_store[pipeline_id] = report
        return report
    except Exception as e:
        logger.error(f"Report generation failed for '{pipeline_id}': {e}")
        raise HTTPException(status_code=500, detail=f"Report error: {e}")


@router.get("/{pipeline_id}", response_model=FinalReport)
async def get_report(pipeline_id: str):
    report = get_stored_report(pipeline_id)
    if not report:
        raise HTTPException(status_code=404, detail=f"No report found for '{pipeline_id}'.")
    return report
