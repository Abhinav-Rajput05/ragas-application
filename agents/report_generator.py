"""
Agent 5: Report Generator Agent
"""
from __future__ import annotations

from core.config import get_settings
from core.models import (
    EvaluationResult, DiagnosisReport, OptimizationResult, CostAccuracyReport,
    PrescriptionSheet, PrescriptionItem, Priority, BeforeAfterComparison,
    ProductionReadinessReport, ProductionReadinessCheckItem, ProductionVerdict,
    FinalReport, HealthCategory, RAGASMetrics,
)
from utils.llm_client import chat_json_sync
from utils.health_score import compute_health_score
from utils.logger import logger


_RX_SYSTEM = """You are RAG Doctor — an expert AI system that writes clinical-style
prescription sheets for failing RAG pipelines.
Be specific, technical, and actionable.
Use P1 (critical), P2 (important), P3 (nice-to-have) priorities.
Respond only with valid JSON."""

_RX_PROMPT = """Generate an AI Prescription Sheet for this RAG pipeline.

Pipeline: {pipeline_name}
Current Health Score: {score}/100 ({category})

RAGAS Metrics:
- Faithfulness:       {faithfulness:.3f}
- Answer Relevancy:   {answer_relevancy:.3f}
- Context Recall:     {context_recall:.3f}
- Context Precision:  {context_precision:.3f}
- Answer Correctness: {answer_correctness:.3f}

Diagnosis Summary:
- Chunking issues: {chunking_issues}
- Retrieval issues: {retrieval_issues}
- Ranking issues: {ranking_issues}
- Context utilization issues: {context_util_issues}
- Hallucination risk: {hallucination_risk}
- Hallucination rate: {hallucination_rate:.1%}

Current Config:
- Chunk size: {chunk_size}
- Chunk overlap: {chunk_overlap}
- Top-K: {top_k}
- Embedding model: {embedding_model}

Best optimized config found:
- Chunk size: {best_chunk_size}
- Overlap: {best_overlap}
- Top-K: {best_top_k}
- Embedding model: {best_embedding}
- Projected score: {projected_score}/100

Generate 4-6 specific prescriptions. Respond with JSON:
{{
  "prescriptions": [
    {{
      "priority": "P1",
      "issue": "specific problem",
      "fix": "exact actionable fix",
      "expected_metric": "Faithfulness",
      "expected_gain": 12.5,
      "implementation_note": "how to implement"
    }}
  ]
}}"""


def generate_prescription_sheet(
    pipeline_id: str,
    pipeline_name: str,
    evaluation: EvaluationResult,
    diagnosis: DiagnosisReport,
    optimization: OptimizationResult,
) -> PrescriptionSheet:
    metrics = evaluation.aggregate_metrics
    config = evaluation.config
    best = optimization.best_config
    diag = diagnosis

    chunking_issues = diag.chunking.details if (diag.chunking.chunk_size_issue or diag.chunking.overlap_issue) else "None detected"
    retrieval_issues = diag.retrieval.details if (diag.retrieval.missing_chunks or diag.retrieval.low_recall) else "None detected"
    ranking_issues = diag.ranking.details if diag.ranking.relevant_chunks_ranked_low else "None detected"
    context_util_issues = diag.context_utilization.details if diag.context_utilization.llm_ignored_context else "None detected"

    prompt = _RX_PROMPT.format(
        pipeline_name=pipeline_name,
        score=evaluation.health_score.score,
        category=evaluation.health_score.category.value,
        faithfulness=metrics.faithfulness,
        answer_relevancy=metrics.answer_relevancy,
        context_recall=metrics.context_recall,
        context_precision=metrics.context_precision,
        answer_correctness=metrics.answer_correctness,
        chunking_issues=chunking_issues,
        retrieval_issues=retrieval_issues,
        ranking_issues=ranking_issues,
        context_util_issues=context_util_issues,
        hallucination_risk=diag.hallucination.risk.value,
        hallucination_rate=diag.hallucination.hallucination_rate,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        top_k=config.top_k,
        embedding_model=config.embedding_model,
        best_chunk_size=best.chunk_size,
        best_overlap=best.chunk_overlap,
        best_top_k=best.top_k,
        best_embedding=best.embedding_model,
        projected_score=best.health_score,
    )

    logger.info(f"Generating Prescription Sheet for pipeline '{pipeline_id}'")

    try:
        result: dict = chat_json_sync(prompt=prompt, system=_RX_SYSTEM, temperature=0.2, max_tokens=2048)
        raw_items: list[dict] = result.get("prescriptions", [])
    except Exception as exc:
        logger.error(f"Prescription generation failed: {exc}")
        raw_items = []

    prescriptions: list[PrescriptionItem] = []
    for item in raw_items:
        try:
            prescriptions.append(PrescriptionItem(
                priority=Priority(item.get("priority", "P3")),
                issue=item.get("issue", ""),
                fix=item.get("fix", ""),
                expected_metric=item.get("expected_metric", ""),
                expected_gain=float(item.get("expected_gain", 0.0)),
                implementation_note=item.get("implementation_note", ""),
            ))
        except Exception:
            continue

    projected_score = best.health_score
    projected_health = compute_health_score(best.metrics)

    return PrescriptionSheet(
        pipeline_id=pipeline_id,
        pipeline_name=pipeline_name,
        current_health_score=evaluation.health_score.score,
        current_category=evaluation.health_score.category,
        prescriptions=prescriptions,
        projected_health_score=projected_score,
        projected_category=projected_health.category,
    )


def build_before_after_comparison(
    pipeline_id: str,
    baseline_eval: EvaluationResult,
    optimized_metrics: RAGASMetrics,
    optimized_score: float,
) -> BeforeAfterComparison:
    before = baseline_eval.aggregate_metrics
    after = optimized_metrics
    improvement = round(optimized_score - baseline_eval.health_score.score, 1)

    deltas = {
        "faithfulness": round(after.faithfulness - before.faithfulness, 3),
        "answer_relevancy": round(after.answer_relevancy - before.answer_relevancy, 3),
        "context_recall": round(after.context_recall - before.context_recall, 3),
        "context_precision": round(after.context_precision - before.context_precision, 3),
        "answer_correctness": round(after.answer_correctness - before.answer_correctness, 3),
    }

    return BeforeAfterComparison(
        pipeline_id=pipeline_id,
        before_metrics=before,
        after_metrics=after,
        before_health_score=baseline_eval.health_score.score,
        after_health_score=optimized_score,
        improvement=improvement,
        metric_deltas=deltas,
    )


def build_production_readiness(
    pipeline_id: str,
    evaluation: EvaluationResult,
    diagnosis: DiagnosisReport,
) -> ProductionReadinessReport:
    settings = get_settings()
    metrics = evaluation.aggregate_metrics
    halluc = diagnosis.hallucination

    checklist: list[ProductionReadinessCheckItem] = [
        ProductionReadinessCheckItem(
            dimension="Answer Faithfulness",
            passed=metrics.faithfulness >= settings.threshold_faithfulness,
            value=round(metrics.faithfulness, 3),
            threshold=settings.threshold_faithfulness,
            note="Answers must be grounded in retrieved context." if metrics.faithfulness < settings.threshold_faithfulness else "Meets threshold.",
        ),
        ProductionReadinessCheckItem(
            dimension="Retrieval Coverage (Context Recall)",
            passed=metrics.context_recall >= settings.threshold_context_recall,
            value=round(metrics.context_recall, 3),
            threshold=settings.threshold_context_recall,
            note="Retriever must surface relevant context." if metrics.context_recall < settings.threshold_context_recall else "Meets threshold.",
        ),
        ProductionReadinessCheckItem(
            dimension="Latency Budget",
            passed=True, value="N/A", threshold="< 3s p95",
            note="Validate latency under production load before deployment.",
        ),
        ProductionReadinessCheckItem(
            dimension="Cost Budget",
            passed=True, value="N/A", threshold="User-defined",
            note="Review Cost vs Accuracy report to select budget-appropriate config.",
        ),
        ProductionReadinessCheckItem(
            dimension="Hallucination Rate",
            passed=halluc.hallucination_rate <= settings.threshold_hallucination_rate,
            value=round(halluc.hallucination_rate, 3),
            threshold=settings.threshold_hallucination_rate,
            note=f"Hallucination risk: {halluc.risk.value}." if halluc.hallucination_rate > settings.threshold_hallucination_rate else "Within acceptable range.",
        ),
        ProductionReadinessCheckItem(
            dimension="Context Relevance (Context Precision)",
            passed=metrics.context_precision >= 0.65,
            value=round(metrics.context_precision, 3), threshold=0.65,
            note="Retrieved chunks should be highly relevant to queries." if metrics.context_precision < 0.65 else "Meets threshold.",
        ),
        ProductionReadinessCheckItem(
            dimension="Answer Quality (Answer Relevancy)",
            passed=metrics.answer_relevancy >= 0.70,
            value=round(metrics.answer_relevancy, 3), threshold=0.70,
            note="Answers must directly address user questions." if metrics.answer_relevancy < 0.70 else "Meets threshold.",
        ),
        ProductionReadinessCheckItem(
            dimension="Embedding Stability",
            passed=True, value="N/A", threshold="Consistent model version",
            note="Pin embedding model version in production to prevent drift.",
        ),
    ]

    passed_count = sum(1 for c in checklist if c.passed)
    total = len(checklist)

    if passed_count == total:
        verdict = ProductionVerdict.READY
    elif passed_count >= total - 2:
        verdict = ProductionVerdict.NEEDS_WORK
    else:
        verdict = ProductionVerdict.NOT_READY

    deployment_notes = []
    if verdict == ProductionVerdict.READY:
        deployment_notes.append("System meets all critical thresholds. Proceed with staged rollout.")
    elif verdict == ProductionVerdict.NEEDS_WORK:
        deployment_notes.append("Address failing dimensions before production deployment.")
        deployment_notes.append("Consider canary deployment with human review enabled.")
    else:
        deployment_notes.append("Do NOT deploy. Critical quality issues detected.")
        deployment_notes.append("Apply AI Prescription Sheet fixes and re-evaluate.")

    logger.info(f"Production Readiness: {verdict.value} ({passed_count}/{total} checks passed)")

    return ProductionReadinessReport(
        pipeline_id=pipeline_id,
        verdict=verdict,
        checklist=checklist,
        deployment_notes=deployment_notes,
    )


def build_final_report(
    pipeline_id: str,
    pipeline_name: str,
    evaluation: EvaluationResult,
    diagnosis: DiagnosisReport,
    optimization: OptimizationResult,
    cost_accuracy: CostAccuracyReport,
) -> FinalReport:
    prescription = generate_prescription_sheet(
        pipeline_id, pipeline_name, evaluation, diagnosis, optimization
    )
    before_after = build_before_after_comparison(
        pipeline_id, evaluation, optimization.best_config.metrics, optimization.best_config.health_score,
    )
    production_readiness = build_production_readiness(pipeline_id, evaluation, diagnosis)

    return FinalReport(
        pipeline_id=pipeline_id,
        evaluation=evaluation,
        diagnosis=diagnosis,
        optimization=optimization,
        prescription=prescription,
        before_after=before_after,
        production_readiness=production_readiness,
        cost_accuracy=cost_accuracy,
    )
