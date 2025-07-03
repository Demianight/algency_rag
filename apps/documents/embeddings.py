from openai import AsyncOpenAI

from settings import settings

from .schemas import Chunk

client = AsyncOpenAI(api_key=settings.openai.api_key)


async def embed_chunks(chunks: list[Chunk]) -> list[list[float]]:
    """
    Creates embeddings for each chunk text using OpenAI embeddings API.
    Returns list of embeddings (each is a vector of floats).
    """
    # Collect all texts
    texts = [chunk.text for chunk in chunks]

    # Request embeddings
    response = await client.embeddings.create(
        model=settings.openai.model,
        input=texts,
    )

    # Extract vectors
    embeddings = [item.embedding for item in response.data]

    return embeddings
