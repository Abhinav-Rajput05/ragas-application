from __future__ import annotations
from core.models import EvalQuestion, EvalDataset, EvaluationResult
from services.rag_service import get_pipeline_config
from services.evaluation_service import run_ragas_evaluation, collect_rag_outputs_parallel
from utils.llm_client import chat_json_sync
from utils.logger import logger


_SYSTEM = (
    "You are an expert RAG evaluation engineer. "
    "Generate diverse evaluation questions from the document text. "
    "Questions must be answerable from the text only. "
    "Respond with valid JSON only."
)

_PROMPT = """Generate exactly {n} evaluation question-answer pairs from this document.

Document:
\"\"\"{text}\"\"\"

Respond with a JSON array only:
[
  {{"question": "...", "ground_truth": "..."}},
  ...
]

Rules:
- Questions must be specific and answerable from the text.
- Ground truth: factual, 1-3 sentences.
- Cover different parts of the document.
- No yes/no questions."""


def generate_eval_dataset(pipeline_id: str, document_text: str, num_questions: int = 5) -> EvalDataset:
    logger.info(f"Generating {num_questions} eval questions for pipeline '{pipeline_id}'")
    text_sample = document_text[:6000]

    raw: list[dict] = chat_json_sync(
        prompt=_PROMPT.format(n=num_questions, text=text_sample),
        system=_SYSTEM,
        temperature=0.3,
        max_tokens=1500,
    )

    questions = [
        EvalQuestion(question=str(item["question"]), ground_truth=str(item["ground_truth"]))
        for item in raw
        if isinstance(item, dict) and "question" in item and "ground_truth" in item
    ]
    logger.info(f"Generated {len(questions)} questions.")
    return EvalDataset(pipeline_id=pipeline_id, questions=questions)


def run_evaluation_agent(pipeline_id: str, document_text: str, num_questions: int = 5) -> EvaluationResult:
    config = get_pipeline_config(pipeline_id)
    eval_dataset = generate_eval_dataset(pipeline_id, document_text, num_questions)
    per_query = collect_rag_outputs_parallel(
        pipeline_id=pipeline_id,
        eval_questions=eval_dataset.questions,
        max_workers=min(num_questions, 8),
    )
    return run_ragas_evaluation(config, eval_dataset.questions, per_query)
