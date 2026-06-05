"""
Agent 2: Retrieval Inspector Agent
"""
from __future__ import annotations

from core.models import (
    EvaluationResult, ChunkingDiagnosis, RetrievalDiagnosis,
    RankingDiagnosis, ContextUtilizationDiagnosis, PerQueryResult,
)
from utils.llm_client import chat_json_sync
from utils.logger import logger


_RETRIEVAL_ANALYSIS_SYSTEM = """You are a RAG pipeline diagnostics expert.
Analyze the retrieval quality data and identify specific failure patterns.
Be precise and technical. Respond only with valid JSON."""

_RETRIEVAL_ANALYSIS_PROMPT = """Analyze this RAG evaluation data and diagnose retrieval failures.

Pipeline Config:
- Chunk size: {chunk_size}
- Chunk overlap: {chunk_overlap}
- Top-K: {top_k}
- Embedding model: {embedding_model}

Aggregate Metrics:
- Context Recall: {context_recall:.3f}
- Context Precision: {context_precision:.3f}
- Faithfulness: {faithfulness:.3f}
- Answer Relevancy: {answer_relevancy:.3f}

Sample failed queries (where context_recall < 0.6):
{failed_queries}

Respond with JSON:
{{
  "chunking": {{
    "chunk_size_issue": true,
    "overlap_issue": false,
    "details": "explanation"
  }},
  "retrieval": {{
    "missing_chunks": true,
    "wrong_chunks": false,
    "low_recall": true,
    "details": "explanation"
  }},
  "ranking": {{
    "relevant_chunks_ranked_low": false,
    "details": "explanation"
  }},
  "context_utilization": {{
    "llm_ignored_context": false,
    "details": "explanation"
  }}
}}"""


def _format_failed_queries(per_query: list[PerQueryResult], max_items: int = 3) -> str:
    failed = [r for r in per_query if r.context_recall < 0.6]
    failed = sorted(failed, key=lambda r: r.context_recall)[:max_items]

    if not failed:
        return "No significantly failing queries detected."

    lines = []
    for r in failed:
        lines.append(
            f"Q: {r.question}\n"
            f"  Context Recall: {r.context_recall:.2f} | Faithfulness: {r.faithfulness:.2f}\n"
            f"  Retrieved contexts: {len(r.retrieved_contexts)} chunks\n"
            f"  First context snippet: {r.retrieved_contexts[0][:120] if r.retrieved_contexts else 'NONE'}"
        )
    return "\n\n".join(lines)


def run_retrieval_inspector(
    evaluation: EvaluationResult,
) -> tuple[ChunkingDiagnosis, RetrievalDiagnosis, RankingDiagnosis, ContextUtilizationDiagnosis]:
    config = evaluation.config
    metrics = evaluation.aggregate_metrics
    per_query = evaluation.per_query_results

    logger.info(f"Running Retrieval Inspector for pipeline '{config.pipeline_id}'")

    prompt = _RETRIEVAL_ANALYSIS_PROMPT.format(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        top_k=config.top_k,
        embedding_model=config.embedding_model,
        context_recall=metrics.context_recall,
        context_precision=metrics.context_precision,
        faithfulness=metrics.faithfulness,
        answer_relevancy=metrics.answer_relevancy,
        failed_queries=_format_failed_queries(per_query),
    )

    try:
        result: dict = chat_json_sync(
            prompt=prompt,
            system=_RETRIEVAL_ANALYSIS_SYSTEM,
            temperature=0.1,
            max_tokens=1024,
        )
    except Exception as exc:
        logger.error(f"Retrieval Inspector LLM call failed: {exc}")
        return (
            ChunkingDiagnosis(details="Analysis unavailable."),
            RetrievalDiagnosis(details="Analysis unavailable."),
            RankingDiagnosis(details="Analysis unavailable."),
            ContextUtilizationDiagnosis(details="Analysis unavailable."),
        )

    chunking = ChunkingDiagnosis(
        chunk_size_issue=result.get("chunking", {}).get("chunk_size_issue", False),
        overlap_issue=result.get("chunking", {}).get("overlap_issue", False),
        details=result.get("chunking", {}).get("details", ""),
    )
    retrieval = RetrievalDiagnosis(
        missing_chunks=result.get("retrieval", {}).get("missing_chunks", False),
        wrong_chunks=result.get("retrieval", {}).get("wrong_chunks", False),
        low_recall=result.get("retrieval", {}).get("low_recall", False),
        details=result.get("retrieval", {}).get("details", ""),
    )
    ranking = RankingDiagnosis(
        relevant_chunks_ranked_low=result.get("ranking", {}).get("relevant_chunks_ranked_low", False),
        details=result.get("ranking", {}).get("details", ""),
    )
    context_util = ContextUtilizationDiagnosis(
        llm_ignored_context=result.get("context_utilization", {}).get("llm_ignored_context", False),
        details=result.get("context_utilization", {}).get("details", ""),
    )

    logger.info("Retrieval Inspector analysis complete.")
    return chunking, retrieval, ranking, context_util
