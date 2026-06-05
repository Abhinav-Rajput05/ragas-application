class RAGDoctorError(Exception):
    pass

class DocumentProcessingError(RAGDoctorError):
    pass

class PipelineNotFoundError(RAGDoctorError):
    pass

class EvaluationError(RAGDoctorError):
    pass

class LLMError(RAGDoctorError):
    pass

class OptimizationError(RAGDoctorError):
    pass

class VectorStoreError(RAGDoctorError):
    pass
