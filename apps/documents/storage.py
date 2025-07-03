from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from settings import settings

from .schemas import Chunk

qdrant_client = QdrantClient(
    host=settings.qdrant.host, port=settings.qdrant.port
)  # Host and port loaded from settings


def push_to_qdrant(
    chunks: list[Chunk],
    embeddings: list[list[float]],
    collection_name: str,
):
    """Push chunks + embeddings into Qdrant."""
    points = []

    for chunk, embedding in zip(chunks, embeddings):
        points.append(
            PointStruct(
                id=chunk.id,
                vector=embedding,
                payload=chunk.model_dump(),  # stores all fields as metadata
            )
        )

    qdrant_client.upsert(collection_name=collection_name, points=points)


def display_qdrant_info(collection_name: str):
    collection_info = qdrant_client.get_collection(collection_name)
    print(collection_info)


def create_qdrant_collection(collection_name: str):
    """Create a new Qdrant collection with the specified name."""
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=1536,  # Size of OpenAI embeddings
            distance=Distance.COSINE,
        ),
    )
    print(f"Collection '{collection_name}' created successfully.")


def setup_qdrant_collection(collection_name: str):
    """Setup a Qdrant collection if it doesn't exist."""
    existing_collections = qdrant_client.get_collections().collections
    collection_names = [c.name for c in existing_collections]

    if collection_name not in collection_names:
        create_qdrant_collection(collection_name)
