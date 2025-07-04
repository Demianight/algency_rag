from uuid import uuid4

from fastapi import APIRouter, Depends, UploadFile

from src.llm.dependencies import get_vectorstore
from src.llm.embeddings import push_to_vectorstore
from src.llm.services import delete_document

from .chunk_utils import save_temp_file, split_into_chunks
from .dependencies import get_llama_parser
from .parsers import parse_document_with_llama

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/ingest")
async def ingest_file(
    file: UploadFile,
    tags: list[str] | None = None,
    llama_parser=Depends(get_llama_parser),
    vectorstore=Depends(get_vectorstore),
):
    # Save file
    tmp_path = await save_temp_file(file)

    # Parse file
    docs = await parse_document_with_llama(tmp_path, llama_parser)

    document_id = uuid4().hex
    # Split into chunks
    chunks = split_into_chunks(docs, document_id, file.filename or "unknown_name", tags)

    await push_to_vectorstore(chunks, vectorstore)

    return {"document_id": document_id, "status": "ingested"}


@router.delete("/{doc_id}")
async def document_delete(
    doc_id: str,
    vectorstore=Depends(get_vectorstore),
):
    """
    Delete a document and all its chunks from the vector store.
    """
    return delete_document(doc_id, vectorstore)
