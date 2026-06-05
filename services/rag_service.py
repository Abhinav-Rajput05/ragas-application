from __future__ import annotations
import uuid
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document

from core.config import get_settings
from core.exceptions import PipelineNotFoundError
from core.models import PipelineConfig, UploadedDocument
from services.document_service import build_uploaded_document, load_documents
from services.vector_store_service import build_vector_store, load_vector_store
from utils.logger import logger

_pipeline_registry: dict[str, PipelineConfig] = {}
_document_registry: dict[str, UploadedDocument] = {}


def _get_llm():
    from langchain_openai import ChatOpenAI

    settings = get_settings()
    return ChatOpenAI(
        model=settings.nexus_model,
        api_key=settings.nexus_api_key,
        base_url=settings.nexus_base_url,
        temperature=0.0,
        max_tokens=512,
    )


def ingest_document(
    file_path: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    embedding_model: str | None = None,
) -> tuple[UploadedDocument, PipelineConfig]:
    settings = get_settings()
    chunk_size = chunk_size or settings.default_chunk_size
    chunk_overlap = chunk_overlap or settings.default_chunk_overlap
    embedding_model = embedding_model or settings.default_embedding_model

    pages = load_documents(file_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks: list[Document] = splitter.split_documents(pages)
    logger.info(
        f"Split into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})"
    )

    doc_id = uuid.uuid4().hex[:12]
    pipeline_id = uuid.uuid4().hex[:12]

    build_vector_store(chunks, pipeline_id, embedding_model)

    filename = Path(file_path).name
    ext = Path(file_path).suffix.lower()

    uploaded_doc = build_uploaded_document(doc_id, filename, ext, pages, len(chunks))
    pipeline_config = PipelineConfig(
        pipeline_id=pipeline_id,
        doc_id=doc_id,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=settings.default_top_k,
        embedding_model=embedding_model,
    )

    _document_registry[doc_id] = uploaded_doc
    _pipeline_registry[pipeline_id] = pipeline_config

    logger.info(f"Pipeline ready: pipeline_id={pipeline_id}, doc_id={doc_id}")
    return uploaded_doc, pipeline_config


def get_pipeline_config(pipeline_id: str) -> PipelineConfig:
    if pipeline_id not in _pipeline_registry:
        raise PipelineNotFoundError(f"Pipeline '{pipeline_id}' not found.")
    return _pipeline_registry[pipeline_id]


def register_pipeline(config: PipelineConfig) -> None:
    _pipeline_registry[config.pipeline_id] = config


def answer_question(pipeline_id: str, question: str) -> tuple[str, list[str]]:
    config = get_pipeline_config(pipeline_id)
    vector_store = load_vector_store(pipeline_id, config.embedding_model)
    retriever = vector_store.as_retriever(search_kwargs={"k": config.top_k})
    llm = _get_llm()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    result = qa_chain.invoke({"query": question})
    answer = result.get("result", "")
    sources = result.get("source_documents", [])
    contexts = [doc.page_content for doc in sources]
    return answer, contexts


def build_pipeline_for_config(
    doc_id: str,
    original_pipeline_id: str,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    embedding_model: str,
    raw_pages: list[Document],
) -> PipelineConfig:
    new_pipeline_id = uuid.uuid4().hex[:12]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_documents(raw_pages)
    build_vector_store(chunks, new_pipeline_id, embedding_model)

    config = PipelineConfig(
        pipeline_id=new_pipeline_id,
        doc_id=doc_id,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        embedding_model=embedding_model,
    )
    register_pipeline(config)
    logger.info(
        f"Built pipeline {new_pipeline_id} (chunk={chunk_size}, k={top_k}, model={embedding_model})"
    )
    return config


def get_raw_pages(file_path: str) -> list[Document]:
    return load_documents(file_path)
