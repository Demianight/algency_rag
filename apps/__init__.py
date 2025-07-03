from fastapi import APIRouter

from .documents import documents_router

core_router = APIRouter()

core_router.include_router(documents_router)
