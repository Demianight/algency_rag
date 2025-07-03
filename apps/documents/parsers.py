from typing import Any

from fastapi import HTTPException
from llama_cloud_services import LlamaParse

from settings import settings

llama_parser = LlamaParse(
    api_key=settings.llama.api_key,
    verbose=True,
    language="ru",
)


async def parse_document_with_llama(tmp_path: str) -> Any:
    """Parse document using Llama parser."""
    try:
        docs = await llama_parser.aparse(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing document: {str(e)}")
    if isinstance(docs, list):
        raise HTTPException(
            status_code=500, detail="Expected a single JobResult, got a list."
        )
    return docs
