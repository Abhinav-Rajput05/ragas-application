"""
RAG Health Score calculator.
Formula: 0.30*Faithfulness + 0.25*AnswerRelevancy + 0.25*ContextRecall + 0.20*ContextPrecision
All inputs are 0.0–1.0; output is 0–100.
"""

from core.config import get_settings
from core.models import HealthGrade, HealthCategory, HealthScore, RAGASMetrics


def compute_health_score(metrics: RAGASMetrics) -> HealthScore:
    """
    Compute the weighted RAG Health Score from RAGAS metrics.

    Args:
        metrics: RAGASMetrics with values in [0.0, 1.0]

    Returns:
        HealthScore with numeric score (0–100), letter grade, and category.
    """
    s = get_settings()

    raw = (
        s.weight_faithfulness * metrics.faithfulness
        + s.weight_answer_relevancy * metrics.answer_relevancy
        + s.weight_context_recall * metrics.context_recall
        + s.weight_context_precision * metrics.context_precision
    )
    score = round(raw * 100, 1)

    grade = HealthScore.compute_grade(score)
    category = HealthScore.compute_category(score)

    return HealthScore(score=score, grade=grade, category=category, metrics=metrics)


def score_to_color(score: float) -> str:
    """Return a CSS/hex color string based on health score severity."""
    if score >= 80:
        return "#22c55e"
    elif score >= 60:
        return "#f59e0b"
    elif score >= 40:
        return "#ef4444"
    return "#7f1d1d"
