from uuid import uuid4

from fastapi import APIRouter, UploadFile

from apps.documents.storage import push_to_qdrant

from .embeddings import embed_chunks
from .parsers import parse_document_with_llama
from .utils import save_temp_file, split_into_chunks

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

    # Create embeddings for chunks
    embeddings = await embed_chunks(chunks)

    # Push to Qdrant
    push_to_qdrant(
        chunks=chunks,
        embeddings=embeddings,
        collection_name="documents",
    )

    return {"document_id": document_id, "status": "ingested"}
