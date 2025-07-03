from contextlib import asynccontextmanager

from fastapi import FastAPI

from apps import core_router
from settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize resources here if needed
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
