"""
Main API router — aggregates all endpoint routers.
"""

from fastapi import APIRouter
from api.endpoints import upload, evaluate, diagnose, optimize, report

api_router = APIRouter()

api_router.include_router(upload.router)
api_router.include_router(evaluate.router)
api_router.include_router(diagnose.router)
api_router.include_router(optimize.router)
api_router.include_router(report.router)
