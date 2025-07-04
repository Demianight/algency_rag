# mypy: disable-error-code=import
from unittest.mock import MagicMock

from src.documents.chunk_utils import split_into_chunks
from src.documents.schemas import Chunk


def make_job_result(pages):
    job_result = MagicMock()
    job_result.pages = pages
    return job_result


def make_page(items):
    page = MagicMock()
    page.items = items
    return page


def make_item(value, type_="type1"):
    item = MagicMock()
    item.value = value
    item.type = type_
    return item


def test_split_into_chunks_normal():
    # Simulate splitter that splits on spaces
    class DummySplitter:
        def split_text(self, text):
            return text.split()

    docs = make_job_result(
        [
            make_page([make_item("foo bar"), make_item("baz")]),
            make_page([make_item("qux")]),
        ]
    )
    chunks = split_into_chunks(
        docs, "docid", "file.txt", splitter=DummySplitter()
    )  # type: ignore
    assert len(chunks) == 4
    assert all(isinstance(c, Chunk) for c in chunks)
    assert {c.text for c in chunks} == {"foo", "bar", "baz", "qux"}
    assert all(c.document_id == "docid" for c in chunks)
    assert all(c.source == "file.txt" for c in chunks)


def test_split_into_chunks_skips_empty_and_whitespace():
    class DummySplitter:
        def split_text(self, text):
            return [text]

    docs = make_job_result(
        [
            make_page([make_item(""), make_item("   "), make_item("real")]),
        ]
    )
    chunks = split_into_chunks(
        docs, "docid", "file.txt", splitter=DummySplitter()
    )  # type: ignore
    assert len(chunks) == 1
    assert chunks[0].text == "real"


def test_split_into_chunks_empty_docs():
    docs = make_job_result([])
    chunks = split_into_chunks(docs, "docid", "file.txt", splitter=MagicMock())
    assert chunks == []
