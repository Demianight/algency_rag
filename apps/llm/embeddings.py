from typing import TYPE_CHECKING

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from settings import settings

if TYPE_CHECKING:
    from apps.documents.schemas import Chunk

embeddings = OpenAIEmbeddings(
    model=settings.openai.embed_model,
    api_key=settings.openai.api_key,  # type: ignore[arg-type]
)
qdrant_client = QdrantClient(host=settings.qdrant.host, port=settings.qdrant.port)

vectorstore = QdrantVectorStore(
    client=qdrant_client,
    embedding=embeddings,
    collection_name=settings.qdrant.collection_name,
)


async def create_embeddings(chunks: list["Chunk"]):
    texts = [chunk.text for chunk in chunks]
    metadatas = [chunk.model_dump(exclude={"text"}) for chunk in chunks]

    await vectorstore.aadd_texts(
        texts=texts,
        metadatas=metadatas,
    )


def search_embeddings(query: str, top_k: int = 5):
    return vectorstore.similarity_search(query, k=top_k)
