import tempfile

from fastapi import UploadFile
from llama_cloud_services.parse.types import JobResult
from llama_index.core.node_parser import TokenTextSplitter

from .schemas import Chunk

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


async def save_temp_file(file: UploadFile) -> str:
    """
    Save uploaded file to a temporary path.
    Required by Llama parser.
    """
    file_bytes = await file.read()
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=file.filename
    ) as tmp:
        tmp.write(file_bytes)
        return tmp.name


def split_into_chunks(
    docs: JobResult,
    document_id: str,
    file_name: str,
    tags: list[str] | None = None,
    splitter: TokenTextSplitter | None = None,
) -> list[Chunk]:
    """
    Split parsed document into smaller text chunks.
    """
    if splitter is None:
        splitter = TokenTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )

    result_chunks = []

    for page in docs.pages:
        for item in page.items:
            text = item.value
            if not text:
                continue
            chunks = splitter.split_text(text)
            for chunk in chunks:
                if not chunk.strip():
                    continue
                result_chunks.append(
                    Chunk(
                        document_id=document_id,
                        type=item.type,
                        text=chunk,
                        source=file_name or "unknown_source",
                        tags=tags or [],
                    )
                )
    return result_chunks
