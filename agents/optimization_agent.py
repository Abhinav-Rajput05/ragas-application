"""
Agent 4: Optimization Agent
"""
from __future__ import annotations
import time
from itertools import product

from core.config import get_settings
from core.models import (
    OptimizationConfig, OptimizationResult, CostAccuracyPoint,
    CostAccuracyReport, EvaluationResult, RAGASMetrics,
)
from services.rag_service import build_pipeline_for_config, get_raw_pages
from agents.evaluation_agent import generate_eval_dataset
from services.evaluation_service import collect_rag_outputs_parallel as collect_rag_outputs
from services.evaluation_service import run_ragas_evaluation
from utils.health_score import compute_health_score
from utils.logger import logger

_COST_PER_1K_INPUT = 0.001
_COST_PER_1K_OUTPUT = 0.002


def _estimate_cost(chunk_size: int, top_k: int, num_questions: int) -> float:
    input_tokens = chunk_size * top_k * num_questions
    output_tokens = 200 * num_questions
    cost = (input_tokens / 1000) * _COST_PER_1K_INPUT + (output_tokens / 1000) * _COST_PER_1K_OUTPUT
    return round(cost, 4)


def _compute_pareto_frontier(points: list[CostAccuracyPoint]) -> list[CostAccuracyPoint]:
    for p in points:
        dominated = False
        for q in points:
            if q.config_id == p.config_id:
                continue
            if q.token_cost <= p.token_cost and q.health_score >= p.health_score:
                if q.token_cost < p.token_cost or q.health_score > p.health_score:
                    dominated = True
                    break
        p.is_pareto_optimal = not dominated
    return points


def run_optimization_agent(
    pipeline_id: str,
    file_path: str,
    baseline_evaluation: EvaluationResult,
    num_questions: int = 3,
) -> tuple[OptimizationResult, CostAccuracyReport]:
    settings = get_settings()
    baseline_config = baseline_evaluation.config
    baseline_score = baseline_evaluation.health_score.score

    logger.info(f"Starting optimization grid search for pipeline '{pipeline_id}'")

    raw_pages = get_raw_pages(file_path)
    document_text = " ".join([p.page_content for p in raw_pages])
    eval_dataset = generate_eval_dataset(pipeline_id, document_text, num_questions)

    grid = list(product(
        settings.opt_chunk_sizes,
        settings.opt_chunk_overlaps[:2],
        settings.opt_top_k_values[:2],
        settings.opt_embedding_models[:2],
    ))
    grid = grid[:settings.max_optimization_configs]

    tested_configs: list[OptimizationConfig] = []
    cost_points: list[CostAccuracyPoint] = []

    for i, (chunk_size, chunk_overlap, top_k, emb_model) in enumerate(grid):
        config_id = f"cfg_{i+1:03d}"
        logger.info(
            f"Testing config {i+1}/{len(grid)}: "
            f"chunk={chunk_size}, overlap={chunk_overlap}, k={top_k}, model={emb_model}"
        )
        t0 = time.time()
        try:
            new_config = build_pipeline_for_config(
                doc_id=baseline_config.doc_id,
                original_pipeline_id=pipeline_id,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                top_k=top_k,
                embedding_model=emb_model,
                raw_pages=raw_pages,
            )
            per_query = collect_rag_outputs(new_config.pipeline_id, eval_dataset.questions)
            eval_result = run_ragas_evaluation(new_config, eval_dataset.questions, per_query)
            latency_ms = (time.time() - t0) * 1000
            token_cost = _estimate_cost(chunk_size, top_k, num_questions)

            opt_config = OptimizationConfig(
                config_id=config_id,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                top_k=top_k,
                embedding_model=emb_model,
                health_score=eval_result.health_score.score,
                metrics=eval_result.aggregate_metrics,
                token_cost=token_cost,
                latency_ms=round(latency_ms, 1),
            )
            tested_configs.append(opt_config)
            cost_points.append(CostAccuracyPoint(
                config_id=config_id,
                token_cost=token_cost,
                latency_ms=round(latency_ms, 1),
                health_score=eval_result.health_score.score,
                embedding_model=emb_model,
                chunk_size=chunk_size,
                top_k=top_k,
            ))
        except Exception as exc:
            logger.warning(f"Config {config_id} failed: {exc}")
            continue

    if not tested_configs:
        logger.error("All optimization configs failed — returning baseline.")
        best = OptimizationConfig(
            config_id="baseline",
            chunk_size=baseline_config.chunk_size,
            chunk_overlap=baseline_config.chunk_overlap,
            top_k=baseline_config.top_k,
            embedding_model=baseline_config.embedding_model,
            health_score=baseline_score,
            metrics=baseline_evaluation.aggregate_metrics,
        )
        return (
            OptimizationResult(
                pipeline_id=pipeline_id, tested_configs=[best], best_config=best,
                baseline_health_score=baseline_score, optimized_health_score=baseline_score, improvement=0.0,
            ),
            CostAccuracyReport(
                pipeline_id=pipeline_id, points=[],
                recommended_config_id="baseline", recommendation_reason="No valid configs tested.",
            ),
        )

    best = max(tested_configs, key=lambda c: c.health_score)
    improvement = round(best.health_score - baseline_score, 1)
    logger.info(
        f"Optimization complete. Best: chunk={best.chunk_size}, k={best.top_k}, "
        f"model={best.embedding_model} -> score={best.health_score} (delta={improvement:+.1f})"
    )

    opt_result = OptimizationResult(
        pipeline_id=pipeline_id, tested_configs=tested_configs, best_config=best,
        baseline_health_score=baseline_score, optimized_health_score=best.health_score, improvement=improvement,
    )

    cost_points = _compute_pareto_frontier(cost_points)
    pareto_points = [p for p in cost_points if p.is_pareto_optimal]
    recommended = max(pareto_points, key=lambda p: p.health_score) if pareto_points else cost_points[0]

    cost_report = CostAccuracyReport(
        pipeline_id=pipeline_id, points=cost_points,
        recommended_config_id=recommended.config_id,
        recommendation_reason=(
            f"Pareto-optimal config with best accuracy ({recommended.health_score}/100) "
            f"at cost ${recommended.token_cost:.4f} per evaluation run."
        ),
    )
    return opt_result, cost_report
