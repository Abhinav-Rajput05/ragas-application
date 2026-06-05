"""
Nexus API wrapper.
All LLM calls in the project go through this single module.
Uses the OpenAI-compatible interface pointed at Navigate Labs.
"""

from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI, OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from core.config import get_settings
from core.exceptions import LLMError
from utils.logger import logger


def _build_sync_client() -> OpenAI:
    settings = get_settings()
    return OpenAI(
        api_key=settings.nexus_api_key,
        base_url=settings.nexus_base_url,
    )


def _build_async_client() -> AsyncOpenAI:
    settings = get_settings()
    return AsyncOpenAI(
        api_key=settings.nexus_api_key,
        base_url=settings.nexus_base_url,
    )


_sync_client: OpenAI | None = None
_async_client: AsyncOpenAI | None = None


def get_sync_client() -> OpenAI:
    global _sync_client
    if _sync_client is None:
        _sync_client = _build_sync_client()
    return _sync_client


def get_async_client() -> AsyncOpenAI:
    global _async_client
    if _async_client is None:
        _async_client = _build_async_client()
    return _async_client


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def chat_sync(
    prompt: str,
    system: str = "You are a helpful AI assistant.",
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> str:
    """
    Synchronous chat completion via Nexus API.
    Retries up to 3 times with exponential backoff.
    """
    settings = get_settings()
    client = get_sync_client()
    try:
        response = client.chat.completions.create(
            model=settings.nexus_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = getattr(response.choices[0].message, "content", "") or ""
        logger.debug(f"LLM response received ({len(content)} chars)")
        return content
    except Exception as exc:
        logger.error(f"Nexus API call failed: {exc}")
        raise LLMError(f"Nexus API error: {exc}") from exc


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
async def chat_async(
    prompt: str,
    system: str = "You are a helpful AI assistant.",
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> str:
    """
    Async chat completion via Nexus API.
    Retries up to 3 times with exponential backoff.
    """
    settings = get_settings()
    client = get_async_client()
    try:
        response = await client.chat.completions.create(
            model=settings.nexus_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = getattr(response.choices[0].message, "content", "") or ""
        logger.debug(f"LLM async response received ({len(content)} chars)")
        return content
    except Exception as exc:
        logger.error(f"Nexus API async call failed: {exc}")
        raise LLMError(f"Nexus API error: {exc}") from exc


def chat_json_sync(
    prompt: str,
    system: str = "You are a helpful AI assistant. Always respond with valid JSON.",
    temperature: float = 0.1,
    max_tokens: int = 2048,
) -> Any:
    """
    Synchronous chat that expects and parses a JSON response.
    Strips markdown code fences if present.
    """
    raw = chat_sync(prompt, system=system, temperature=temperature, max_tokens=max_tokens)
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error(f"Failed to parse LLM JSON response: {exc}\nRaw: {raw[:300]}")
        raise LLMError(f"LLM returned invalid JSON: {exc}") from exc


async def chat_json_async(
    prompt: str,
    system: str = "You are a helpful AI assistant. Always respond with valid JSON.",
    temperature: float = 0.1,
    max_tokens: int = 2048,
) -> Any:
    """
    Async chat that expects and parses a JSON response.
    """
    raw = await chat_async(prompt, system=system, temperature=temperature, max_tokens=max_tokens)
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error(f"Failed to parse LLM JSON response: {exc}\nRaw: {raw[:300]}")
        raise LLMError(f"LLM returned invalid JSON: {exc}") from exc
