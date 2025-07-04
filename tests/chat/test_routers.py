import json

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain.schema import BaseRetriever, Document
from langchain_community.llms.fake import FakeListLLM

from src.chat.routers import router
from src.llm import dependencies as llm_dependencies

app = FastAPI()
app.include_router(router)


def make_llm_response():
    return {
        "result": "42",
        "source_documents": [
            {
                "metadata": {"_id": "id1", "document_id": "doc1"},
                "page_content": "answer text",
            },
            {
                "metadata": {"_id": "id2", "document_id": "doc2"},
                "page_content": "another text",
            },
        ],
    }


class DummyRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str):
        return [
            Document(
                page_content="answer text",
                metadata={"_id": "id1", "document_id": "doc1"},
            ),
            Document(
                page_content="another text",
                metadata={"_id": "id2", "document_id": "doc2"},
            ),
        ]


class DummyVectorstore:
    def as_retriever(self, search_kwargs=None):
        return DummyRetriever()


def dummy_llm():
    # Provide response as a stringified JSON
    return FakeListLLM(responses=[json.dumps(make_llm_response())])


def dummy_vectorstore():
    return DummyVectorstore()


app.dependency_overrides[llm_dependencies.get_llm] = dummy_llm
app.dependency_overrides[llm_dependencies.get_vectorstore] = dummy_vectorstore


def test_ask_endpoint_transforms_llm_response():
    """
    I cannot lie, this test - sucks.
    Thank god it even works.
    """
    client = TestClient(app)
    req = {"question": "What is the answer?", "top_k": 2}
    resp = client.post("/chat/ask", json=req)
    assert resp.status_code == 200
    data = resp.json()
    data = json.loads(
        data["result"]
    )  # Convert stringified JSON back to dict ONLY for testing
    assert data["result"] == "42"
    assert len(data["source_documents"]) == 2
    assert data["source_documents"][0]["metadata"]["_id"] == "id1"
    assert data["source_documents"][0]["metadata"]["document_id"] == "doc1"
    assert data["source_documents"][0]["page_content"] == "answer text"
    assert data["source_documents"][1]["metadata"]["_id"] == "id2"
    assert data["source_documents"][1]["metadata"]["document_id"] == "doc2"
    assert data["source_documents"][1]["page_content"] == "another text"
