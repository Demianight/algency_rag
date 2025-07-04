from uuid import uuid4

from fastapi import APIRouter, Depends, UploadFile

from src.llm.dependencies import get_vectorstore
from src.llm.embeddings import create_embeddings

from .chunk_utils import save_temp_file, split_into_chunks
from .dependencies import get_llama_parser
from .parsers import parse_document_with_llama

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/ingest")
async def ingest_file(
    file: UploadFile,
    llama_parser=Depends(get_llama_parser),
    vectorstore=Depends(get_vectorstore),
):
    document_id = uuid4().hex

    # Save file
    tmp_path = await save_temp_file(file)

    # Parse file
    docs = await parse_document_with_llama(tmp_path, llama_parser)

    # Split into chunks
    chunks = split_into_chunks(
        docs, document_id, file.filename or "unknown_name"
    )

    await create_embeddings(chunks, vectorstore)

    return {"document_id": document_id, "status": "ingested"}
