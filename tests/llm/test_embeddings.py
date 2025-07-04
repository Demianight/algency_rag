# mypy: disable-error-code=import
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.llm.embeddings import push_to_vectorstore


@pytest.mark.asyncio
async def test_create_embeddings_calls_vectorstore_aadd_texts():
    # Mock Chunk objects
    chunk1 = MagicMock()
    chunk1.text = "text1"
    chunk1.model_dump.return_value = {"meta": 1}
    chunk2 = MagicMock()
    chunk2.text = "text2"
    chunk2.model_dump.return_value = {"meta": 2}
    chunks = [chunk1, chunk2]

    # Mock vectorstore with async aadd_texts
    vectorstore = MagicMock()
    vectorstore.aadd_texts = AsyncMock()

    await push_to_vectorstore(chunks, vectorstore)  # type: ignore

    vectorstore.aadd_texts.assert_awaited_once_with(
        texts=["text1", "text2"],
        metadatas=[{"meta": 1}, {"meta": 2}],
    )


def test_placeholder():
    pass
