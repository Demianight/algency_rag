# mypy: disable-error-code=import
import pytest
from fastapi import HTTPException

from src.documents.parsers import parse_document_with_llama


class DummyLlamaParser:
    def __init__(self, result=None, raise_exc=None):
        self.result = result
        self.raise_exc = raise_exc

    async def aparse(self, tmp_path):
        if self.raise_exc:
            raise self.raise_exc
        return self.result


@pytest.mark.asyncio
async def test_parse_document_with_llama_success():
    parser = DummyLlamaParser(result="jobresult")
    result = await parse_document_with_llama("/tmp/file", parser)  # type: ignore
    assert result == "jobresult"


@pytest.mark.asyncio
async def test_parse_document_with_llama_raises_on_exception():
    parser = DummyLlamaParser(raise_exc=ValueError("fail"))
    with pytest.raises(HTTPException) as exc:
        await parse_document_with_llama("/tmp/file", parser)  # type: ignore
    assert exc.value.status_code == 500
    assert "Error parsing document" in exc.value.detail


@pytest.mark.asyncio
async def test_parse_document_with_llama_raises_on_list_result():
    parser = DummyLlamaParser(result=[1, 2, 3])
    with pytest.raises(HTTPException) as exc:
        await parse_document_with_llama("/tmp/file", parser)  # type: ignore
    assert exc.value.status_code == 500
    assert "Expected a single JobResult" in exc.value.detail
