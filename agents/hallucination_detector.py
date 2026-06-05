"""
Agent 3: Hallucination Detector
Uses only the LLM (Nexus API) for per-claim hallucination detection.
"""

from __future__ import annotations

import re
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.models import HallucinationDiagnosis, HallucinationRisk, PerQueryResult
from utils.llm_client import chat_json_sync
from utils.logger import logger


def _split_claims(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def _llm_check(question: str, context: str, answer: str) -> dict[str, Any]:
    system = (
        "You are a hallucination detection expert for RAG systems. "
        "Identify claims in the answer that are NOT supported by the context. "
        "Respond only with valid JSON."
    )
    prompt = f"""Question: {question}

Context (retrieved from the document):
{context[:1500]}

Generated Answer:
{answer}

Identify which specific sentences or claims in the answer are NOT supported by the context above.

Respond with JSON:
{{
  "unsupported_claims": ["exact claim 1", "exact claim 2"],
  "hallucination_risk": "Low",
  "explanation": "one sentence reason"
}}

hallucination_risk must be one of: Low, Medium, High"""
    try:
        return chat_json_sync(prompt=prompt, system=system, temperature=0.0, max_tokens=400)
    except Exception as exc:
        logger.error(f"LLM hallucination check failed: {exc}")
        return {"unsupported_claims": [], "hallucination_risk": "Low", "explanation": "unavailable"}


def _check_one(qr: PerQueryResult) -> dict[str, Any]:
    answer = qr.generated_answer
    context = " ".join(qr.retrieved_contexts)

    if not answer or not context:
        return {"unsupported_claims": [], "risk": HallucinationRisk.LOW, "hallucination_rate": 0.0, "explanation": ""}

    result = _llm_check(qr.question, context, answer)
    unsupported = result.get("unsupported_claims", [])
    claims = _split_claims(answer)
    total = max(len(claims), 1)
    rate = len(unsupported) / total
    risk_str = result.get("hallucination_risk", "Low")

    if rate > 0.4 or risk_str == "High":
        risk = HallucinationRisk.HIGH
    elif rate > 0.15 or risk_str == "Medium":
        risk = HallucinationRisk.MEDIUM
    else:
        risk = HallucinationRisk.LOW

    return {
        "unsupported_claims": unsupported,
        "risk": risk,
        "hallucination_rate": round(rate, 3),
        "explanation": result.get("explanation", ""),
    }


def run_hallucination_detector(
    per_query_results: list[PerQueryResult],
) -> HallucinationDiagnosis:
    n = len(per_query_results)
    logger.info(f"Hallucination Detector: checking {n} answers in parallel ...")

    results_map: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=min(n, 5)) as pool:
        futures = {pool.submit(_check_one, qr): i for i, qr in enumerate(per_query_results)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results_map[i] = fut.result()
            except Exception as exc:
                logger.warning(f"Answer {i+1} check failed: {exc}")
                results_map[i] = {"unsupported_claims": [], "risk": HallucinationRisk.LOW, "hallucination_rate": 0.0}

    all_unsupported: list[str] = []
    all_traces: list[dict] = []
    rates: list[float] = []

    for i, qr in enumerate(per_query_results):
        r = results_map.get(i, {})
        all_unsupported.extend(r.get("unsupported_claims", []))
        all_traces.append({
            "question": qr.question,
            "risk": r.get("risk", HallucinationRisk.LOW).value if hasattr(r.get("risk"), "value") else "Low",
            "hallucination_rate": r.get("hallucination_rate", 0.0),
            "explanation": r.get("explanation", ""),
        })
        rates.append(r.get("hallucination_rate", 0.0))

    avg_rate = sum(rates) / len(rates) if rates else 0.0

    if avg_rate > 0.4:
        overall_risk = HallucinationRisk.HIGH
    elif avg_rate > 0.15:
        overall_risk = HallucinationRisk.MEDIUM
    else:
        overall_risk = HallucinationRisk.LOW

    logger.info(f"Hallucination complete — risk: {overall_risk.value}, rate: {avg_rate:.3f}")

    return HallucinationDiagnosis(
        unsupported_claims=list(set(all_unsupported)),
        risk=overall_risk,
        hallucination_rate=round(avg_rate, 3),
        per_claim_trace=all_traces,
    )
