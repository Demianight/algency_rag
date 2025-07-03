import tempfile
from uuid import uuid4

from fastapi import UploadFile
from llama_cloud_services.parse.types import JobResult
from llama_index.core.node_parser import TokenTextSplitter

from .schemas import Chunk

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


async def save_temp_file(file: UploadFile) -> str:
    """
    Save uploaded file to a temporary path.
    It's basically required by LLAMA parser.
    """
    file_bytes = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
        tmp.write(file_bytes)
        return tmp.name


def split_into_chunks(docs: JobResult, document_id: str, file_name: str) -> list[Chunk]:
    """Split parsed document into small text chunks."""
    result_chunks = []
    splitter = TokenTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )  # not really dependency inverted, but whatever

    for page in docs.pages:
        for item in page.items:
            text = item.value
            if not text:
                continue
            chunks = splitter.split_text(text)
            for chunk in chunks:
                result_chunks.append(
                    Chunk(
                        id=uuid4().hex,
                        document_id=document_id,
                        type=item.type,
                        text=chunk,
                        source=file_name or "unknown_source",
                    )
                )
    return result_chunks
