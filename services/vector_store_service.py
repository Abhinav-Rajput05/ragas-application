"""
ChromaDB vector store service.
Manages per-pipeline collections with configurable embedding models.
"""

from __future__ import annotations

import os
import logging

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from core.config import get_settings
from core.exceptions import VectorStoreError
from utils.logger import logger

_CHROMA_SETTINGS = ChromaSettings(
    anonymized_telemetry=False,
    allow_reset=True,
)

_embedding_cache: dict[str, HuggingFaceEmbeddings] = {}


def get_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    """Return a cached HuggingFaceEmbeddings instance for the given model."""
    if model_name not in _embedding_cache:
        logger.info(f"Loading embedding model: {model_name}")
        _embedding_cache[model_name] = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info(f"Embedding model loaded: {model_name}")
    return _embedding_cache[model_name]


def _get_chroma_client() -> chromadb.PersistentClient:
    """Get a persistent ChromaDB client with telemetry disabled."""
    settings = get_settings()
    return chromadb.PersistentClient(
        path=settings.chroma_persist_dir,
        settings=_CHROMA_SETTINGS,
    )


def build_vector_store(
    chunks: list[Document],
    pipeline_id: str,
    embedding_model: str,
) -> Chroma:
    """
    Create or overwrite a ChromaDB collection for a pipeline.
    """
    settings = get_settings()
    try:
        embeddings = get_embeddings(embedding_model)
        collection_name = f"pipeline_{pipeline_id.replace('-', '_')}"
        client = _get_chroma_client()

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=settings.chroma_persist_dir,
            client=client,
        )
        logger.info(
            f"Indexed {len(chunks)} chunks into collection '{collection_name}' "
            f"using model '{embedding_model}'"
        )
        return vector_store

    except Exception as exc:
        raise VectorStoreError(f"Failed to build vector store: {exc}") from exc


def load_vector_store(
    pipeline_id: str,
    embedding_model: str,
) -> Chroma:
    """
    Load an existing ChromaDB collection for a pipeline.
    """
    settings = get_settings()
    try:
        embeddings = get_embeddings(embedding_model)
        collection_name = f"pipeline_{pipeline_id.replace('-', '_')}"
        client = _get_chroma_client()

        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=settings.chroma_persist_dir,
            client=client,
        )
        logger.info(f"Loaded vector store for pipeline '{pipeline_id}'")
        return vector_store

    except Exception as exc:
        raise VectorStoreError(f"Failed to load vector store: {exc}") from exc


def delete_collection(pipeline_id: str) -> None:
    """Delete a pipeline's ChromaDB collection."""
    try:
        client = _get_chroma_client()
        collection_name = f"pipeline_{pipeline_id.replace('-', '_')}"
        client.delete_collection(collection_name)
        logger.info(f"Deleted collection '{collection_name}'")
    except Exception as exc:
        logger.warning(f"Could not delete collection for '{pipeline_id}': {exc}")
