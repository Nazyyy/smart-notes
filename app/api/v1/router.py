# ### FILE: app/api/v1/router.py
"""
API Version 1 Master Route Aggregator.
"""

from fastapi import APIRouter
from app.api.v1.endpoints_documents import router as documents_router
from app.api.v1.endpoints_pages import router as pages_router
from app.api.v1.endpoints_recognition import router as recognition_router
from app.api.v1.endpoints_export import router as export_router

api_v1_router = APIRouter()

api_v1_router.include_router(documents_router)
api_v1_router.include_router(pages_router)
api_v1_router.include_router(recognition_router)
api_v1_router.include_router(export_router)
