from __future__ import annotations
import glob, os
from fastapi import APIRouter, HTTPException
from core.models import OptimizationResult
from api.endpoints.evaluate import get_stored_evaluation
from utils.logger import logger

_optimization_store: dict[str, OptimizationResult] = {}
_cost_accuracy_store: dict = {}

router = APIRouter(prefix="/optimize", tags=["Optimize"])


def get_stored_optimization(pipeline_id: str):
    return _optimization_store.get(pipeline_id)


def get_stored_cost_accuracy(pipeline_id: str):
    return _cost_accuracy_store.get(pipeline_id)


@router.post("/{pipeline_id}", response_model=OptimizationResult)
async def optimize_pipeline(pipeline_id: str):
    evaluation = get_stored_evaluation(pipeline_id)
    if not evaluation:
        raise HTTPException(status_code=404, detail=f"No evaluation for '{pipeline_id}'. Run /evaluate first.")

    files = glob.glob("./data/uploads/*")
    if not files:
        raise HTTPException(status_code=404, detail="No uploaded file found.")
    file_path = max(files, key=os.path.getmtime)

    try:
        from agents.optimization_agent import run_optimization_agent
        opt_result, cost_report = run_optimization_agent(
            pipeline_id=pipeline_id,
            file_path=file_path,
            baseline_evaluation=evaluation,
            num_questions=3,
        )
        _optimization_store[pipeline_id] = opt_result
        _cost_accuracy_store[pipeline_id] = cost_report
        return opt_result
    except Exception as e:
        logger.error(f"Optimization failed for '{pipeline_id}': {e}")
        raise HTTPException(status_code=500, detail=f"Optimization error: {e}")
