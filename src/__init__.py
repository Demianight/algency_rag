from fastapi import APIRouter

from .chat import chat_router
from .documents import documents_router

core_router = APIRouter()

core_router.include_router(documents_router)
core_router.include_router(chat_router)
