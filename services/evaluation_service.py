from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_correctness,
)

from core.config import get_settings
from core.exceptions import EvaluationError
from core.models import RAGASMetrics, PerQueryResult, EvaluationResult, EvalQuestion, PipelineConfig
from utils.health_score import compute_health_score
from utils.logger import logger


def _get_ragas_llm():
    from langchain_openai import ChatOpenAI
    s = get_settings()
    return ChatOpenAI(
        model=s.nexus_model,
        api_key=s.nexus_api_key,
        base_url=s.nexus_base_url,
        temperature=0.0,
        max_tokens=256,
        max_retries=1,
        request_timeout=30,
    )


def _get_ragas_embeddings(model_name: str):
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )


def collect_rag_outputs_parallel(
    pipeline_id: str,
    eval_questions: list[EvalQuestion],
    max_workers: int = 5,
) -> list[PerQueryResult]:
    from services.rag_service import answer_question

    results: list[PerQueryResult | None] = [None] * len(eval_questions)

    def _query(idx: int, eq: EvalQuestion) -> tuple[int, PerQueryResult]:
        try:
            answer, contexts = answer_question(pipeline_id, eq.question)
        except Exception as exc:
            logger.warning(f"Query {idx+1} failed: {exc}")
            answer, contexts = "", []
        return idx, PerQueryResult(
            question=eq.question,
            ground_truth=eq.ground_truth,
            generated_answer=answer,
            retrieved_contexts=contexts,
            faithfulness=0.0,
            answer_relevancy=0.0,
            context_recall=0.0,
            context_precision=0.0,
            answer_correctness=0.0,
        )

    logger.info(f"Collecting {len(eval_questions)} answers in parallel (workers={max_workers})...")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_query, i, eq): i for i, eq in enumerate(eval_questions)}
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result

    return [r for r in results if r is not None]


def run_ragas_evaluation(
    pipeline_config: PipelineConfig,
    eval_questions: list[EvalQuestion],
    per_query_results: list[PerQueryResult],
) -> EvaluationResult:
    valid = [r for r in per_query_results if r.generated_answer.strip() and r.retrieved_contexts]
    if not valid:
        raise EvaluationError("No valid answers to evaluate. Check the RAG pipeline.")

    logger.info(f"Running RAGAS on {len(valid)} questions...")

    data = {
        "question": [r.question for r in valid],
        "answer": [r.generated_answer for r in valid],
        "contexts": [r.retrieved_contexts for r in valid],
        "ground_truth": [r.ground_truth for r in valid],
    }

    try:
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        ragas_llm = LangchainLLMWrapper(_get_ragas_llm())
        ragas_emb = LangchainEmbeddingsWrapper(_get_ragas_embeddings(pipeline_config.embedding_model))

        metrics_list = [faithfulness, answer_relevancy, context_recall, context_precision, answer_correctness]

        for m in metrics_list:
            m.llm = ragas_llm
            if hasattr(m, "embeddings"):
                m.embeddings = ragas_emb

        dataset = Dataset.from_dict(data)
        result = evaluate(
            dataset=dataset,
            metrics=metrics_list,
            raise_exceptions=False,
            is_async=True,
        )

        scores_df = result.to_pandas()

        def safe_mean(col: str) -> float:
            if col in scores_df.columns:
                return float(scores_df[col].fillna(0.0).mean())
            return 0.0

        aggregate = RAGASMetrics(
            faithfulness=safe_mean("faithfulness"),
            answer_relevancy=safe_mean("answer_relevancy"),
            context_recall=safe_mean("context_recall"),
            context_precision=safe_mean("context_precision"),
            answer_correctness=safe_mean("answer_correctness"),
        )

        updated: list[PerQueryResult] = []
        for i, r in enumerate(valid):
            row = scores_df.iloc[i] if i < len(scores_df) else {}
            updated.append(PerQueryResult(
                question=r.question,
                ground_truth=r.ground_truth,
                generated_answer=r.generated_answer,
                retrieved_contexts=r.retrieved_contexts,
                faithfulness=float(row.get("faithfulness") or 0.0),
                answer_relevancy=float(row.get("answer_relevancy") or 0.0),
                context_recall=float(row.get("context_recall") or 0.0),
                context_precision=float(row.get("context_precision") or 0.0),
                answer_correctness=float(row.get("answer_correctness") or 0.0),
            ))

        health = compute_health_score(aggregate)
        logger.info(
            f"RAGAS done — Score: {health.score}/100 Grade {health.grade.value} "
            f"| F:{aggregate.faithfulness:.2f} AR:{aggregate.answer_relevancy:.2f} "
            f"CR:{aggregate.context_recall:.2f} CP:{aggregate.context_precision:.2f} "
            f"AC:{aggregate.answer_correctness:.2f}"
        )

        return EvaluationResult(
            pipeline_id=pipeline_config.pipeline_id,
            config=pipeline_config,
            per_query_results=updated,
            aggregate_metrics=aggregate,
            health_score=health,
        )

    except EvaluationError:
        raise
    except Exception as exc:
        raise EvaluationError(f"RAGAS evaluation failed: {exc}") from exc
