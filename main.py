from contextlib import asynccontextmanager

from fastapi import FastAPI

from apps import core_router
from apps.documents.storage import display_qdrant_info, setup_qdrant_collection
from settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_qdrant_collection("documents")
    yield
    # Cleanup resources here if needed


app = FastAPI(lifespan=lifespan)

app.include_router(
    core_router,
    # prefix="/api/v1",
)


@app.get("/")
def health_check():
    return {"status": "alive"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
