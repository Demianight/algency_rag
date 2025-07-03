from uuid import uuid4

from fastapi import APIRouter, UploadFile

from apps.llm.embeddings import create_embeddings

from .chunk_utils import save_temp_file, split_into_chunks
from .parsers import parse_document_with_llama

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/ingest")
async def ingest_file(file: UploadFile):
    document_id = uuid4().hex

    # Save file
    tmp_path = await save_temp_file(file)

    # Parse file
    docs = await parse_document_with_llama(tmp_path)

    # Split into chunks
    chunks = split_into_chunks(docs, document_id, file.filename or "unknown_name")

    await create_embeddings(chunks)

    return {"document_id": document_id, "status": "ingested"}
