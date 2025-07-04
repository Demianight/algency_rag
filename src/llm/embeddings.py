from typing import TYPE_CHECKING

from langchain.vectorstores import VectorStore

from src.settings import settings

if TYPE_CHECKING:
    from src.documents.schemas import Chunk


async def push_to_vectorstore(chunks: list["Chunk"], vectorstore: VectorStore) -> None:
    texts = [chunk.text for chunk in chunks]
    metadatas = [chunk.model_dump(exclude={"text"}) for chunk in chunks]

    ids = await vectorstore.aadd_texts(
        texts=texts,
        metadatas=metadatas,
    )

    if settings.debug:
        print(f"Added {len(texts)} chunks to vectorstore with IDs: {ids}")
