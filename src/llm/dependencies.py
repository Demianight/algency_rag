from fastapi import Depends
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from pydantic import SecretStr
from qdrant_client import QdrantClient

from src.settings import settings


def get_llm():
    return ChatOpenAI(
        model=settings.openai.qa_model,
        api_key=SecretStr(settings.openai.api_key),
    )


def get_embeddings():
    return OpenAIEmbeddings(
        model=settings.openai.embed_model,
        api_key=SecretStr(settings.openai.api_key),
    )


def get_vectorstore(embeddings=Depends(get_embeddings)):
    client = QdrantClient(host=settings.qdrant.host, port=settings.qdrant.port)
    return QdrantVectorStore(
        client=client,
        embedding=embeddings,
        collection_name=settings.qdrant.collection_name,
    )
