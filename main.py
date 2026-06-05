"""
RAG Doctor — FastAPI Application Entry Point
Run: uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import os
os.makedirs("./data/uploads", exist_ok = True)
os.makedirs("./data/chroma", exist_ok = True)
os.makedirs("./logs", exist_ok = True)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.router import api_router
from core.config import get_settings

settings = get_settings()

app = FastAPI(
title = "RAG Doctor API", 
description = "AI-powered RAG pipeline evaluation, diagnosis, and optimization platform.", 
version = "1.0.0", 
docs_url = "/docs", 
redoc_url = "/redoc", 
)

app.add_middleware(
CORSMiddleware, 
allow_origins = ["*"], 
allow_credentials = True, 
allow_methods = ["*"], 
allow_headers = ["*"], 
)

app.include_router(api_router, prefix = "/api/v1")


@app.get("/health")
async def health_check():
    return{"status": "ok", "service": "RAG Doctor"}
