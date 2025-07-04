from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.documents.schemas import Chunk


async def create_embeddings(chunks: list["Chunk"], vectorstore) -> None:
    texts = [chunk.text for chunk in chunks]
    metadatas = [chunk.model_dump(exclude={"text"}) for chunk in chunks]

    await vectorstore.aadd_texts(
        texts=texts,
        metadatas=metadatas,
    )
