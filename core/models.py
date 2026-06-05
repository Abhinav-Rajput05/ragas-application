from __future__ import annotations
from enum import Enum
from typing import Any
from datetime import datetime

from pydantic import BaseModel, Field


class HealthGrade(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    F = "F"


class HealthCategory(str, Enum):
    PRODUCTION_READY = "Production Ready"
    NEEDS_OPTIMIZATION = "Needs Optimization"
    NOT_READY = "Not Ready"
    CRITICAL = "Critical"


class HallucinationRisk(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class Priority(str, Enum):
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


class ProductionVerdict(str, Enum):
    READY = "Ready"
    NEEDS_WORK = "Needs Work"
    NOT_READY = "Not Ready"


class UploadedDocument(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    page_count: int
    chunk_count: int
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)


class PipelineConfig(BaseModel):
    pipeline_id: str
    doc_id: str
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    embedding_model: str = "all-MiniLM-L6-v2"
    created_at: datetime = Field(default_factory=datetime.utcnow)


class EvalQuestion(BaseModel):
    question: str
    ground_truth: str


class EvalDataset(BaseModel):
    pipeline_id: str
    questions: list[EvalQuestion]
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class RAGASMetrics(BaseModel):
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_recall: float = 0.0
    context_precision: float = 0.0
    answer_correctness: float = 0.0


class HealthScore(BaseModel):
    score: float
    grade: HealthGrade
    category: HealthCategory
    metrics: RAGASMetrics

    @classmethod
    def compute_grade(cls, score: float) -> HealthGrade:
        if score >= 90:
            return HealthGrade.A
        if score >= 80:
            return HealthGrade.B
        if score >= 70:
            return HealthGrade.C
        if score >= 60:
            return HealthGrade.D
        return HealthGrade.F

    @classmethod
    def compute_category(cls, score: float) -> HealthCategory:
        if score >= 80:
            return HealthCategory.PRODUCTION_READY
        if score >= 60:
            return HealthCategory.NEEDS_OPTIMIZATION
        if score >= 40:
            return HealthCategory.NOT_READY
        return HealthCategory.CRITICAL


class PerQueryResult(BaseModel):
    question: str
    ground_truth: str
    generated_answer: str
    retrieved_contexts: list[str]
    faithfulness: float
    answer_relevancy: float
    context_recall: float
    context_precision: float
    answer_correctness: float


class EvaluationResult(BaseModel):
    pipeline_id: str
    config: PipelineConfig
    per_query_results: list[PerQueryResult]
    aggregate_metrics: RAGASMetrics
    health_score: HealthScore
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)


class ChunkingDiagnosis(BaseModel):
    chunk_size_issue: bool = False
    overlap_issue: bool = False
    details: str = ""


class RetrievalDiagnosis(BaseModel):
    missing_chunks: bool = False
    wrong_chunks: bool = False
    low_recall: bool = False
    details: str = ""


class RankingDiagnosis(BaseModel):
    relevant_chunks_ranked_low: bool = False
    details: str = ""


class ContextUtilizationDiagnosis(BaseModel):
    model_config = {"protected_namespaces": ()}
    llm_ignored_context: bool = False
    details: str = ""


class HallucinationDiagnosis(BaseModel):
    unsupported_claims: list[str] = Field(default_factory=list)
    risk: HallucinationRisk = HallucinationRisk.LOW
    hallucination_rate: float = 0.0
    per_claim_trace: list[dict[str, Any]] = Field(default_factory=list)


class DiagnosisReport(BaseModel):
    pipeline_id: str
    chunking: ChunkingDiagnosis
    retrieval: RetrievalDiagnosis
    ranking: RankingDiagnosis
    context_utilization: ContextUtilizationDiagnosis
    hallucination: HallucinationDiagnosis
    diagnosed_at: datetime = Field(default_factory=datetime.utcnow)


class OptimizationConfig(BaseModel):
    config_id: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    embedding_model: str
    health_score: float = 0.0
    metrics: RAGASMetrics = Field(default_factory=RAGASMetrics)
    token_cost: float = 0.0
    latency_ms: float = 0.0


class OptimizationResult(BaseModel):
    pipeline_id: str
    tested_configs: list[OptimizationConfig]
    best_config: OptimizationConfig
    baseline_health_score: float
    optimized_health_score: float
    improvement: float
    optimized_at: datetime = Field(default_factory=datetime.utcnow)


class PrescriptionItem(BaseModel):
    priority: Priority
    issue: str
    fix: str
    expected_metric: str
    expected_gain: float
    implementation_note: str = ""


class PrescriptionSheet(BaseModel):
    pipeline_id: str
    pipeline_name: str = "RAG Pipeline"
    evaluated_year: int = 2026
    current_health_score: float
    current_category: HealthCategory
    prescriptions: list[PrescriptionItem]
    projected_health_score: float
    projected_category: HealthCategory
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class BeforeAfterComparison(BaseModel):
    pipeline_id: str
    before_metrics: RAGASMetrics
    after_metrics: RAGASMetrics
    before_health_score: float
    after_health_score: float
    improvement: float
    metric_deltas: dict[str, float]


class ProductionReadinessCheckItem(BaseModel):
    dimension: str
    passed: bool
    value: float | str
    threshold: float | str
    note: str = ""


class ProductionReadinessReport(BaseModel):
    pipeline_id: str
    verdict: ProductionVerdict
    checklist: list[ProductionReadinessCheckItem]
    deployment_notes: list[str]
    assessed_at: datetime = Field(default_factory=datetime.utcnow)


class CostAccuracyPoint(BaseModel):
    config_id: str
    token_cost: float
    latency_ms: float
    health_score: float
    embedding_model: str
    chunk_size: int
    top_k: int
    is_pareto_optimal: bool = False


class CostAccuracyReport(BaseModel):
    pipeline_id: str
    points: list[CostAccuracyPoint]
    recommended_config_id: str
    recommendation_reason: str


class FinalReport(BaseModel):
    pipeline_id: str
    evaluation: EvaluationResult
    diagnosis: DiagnosisReport
    optimization: OptimizationResult
    prescription: PrescriptionSheet
    before_after: BeforeAfterComparison
    production_readiness: ProductionReadinessReport
    cost_accuracy: CostAccuracyReport
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class UploadResponse(BaseModel):
    doc_id: str
    pipeline_id: str
    message: str
    chunk_count: int


class EvaluationRequest(BaseModel):
    pipeline_id: str = ""
    num_questions: int = 5


class OptimizationRequest(BaseModel):
    pipeline_id: str
    baseline_evaluation_id: str | None = None


class ReportRequest(BaseModel):
    pipeline_id: str


class ErrorResponse(BaseModel):
    error: str
    detail: str = ""
