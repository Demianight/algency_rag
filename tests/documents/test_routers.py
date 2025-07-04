# mypy: disable-error-code=import
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.documents.routers import router
from src.llm.services import delete_document

app = FastAPI()
app.include_router(router)
client = TestClient(app)


def test_delete_document_endpoint_success():
    """Test successful document deletion."""
    with patch(
        "src.documents.routers.get_vectorstore"
    ) as mock_get_vectorstore:
        mock_vectorstore = AsyncMock()
        mock_vectorstore.client = MagicMock()
        mock_get_vectorstore.return_value = mock_vectorstore

        response = client.delete("/documents/test_doc_id")

        assert response.status_code == 200
        assert response.json() == {
            "document_id": "test_doc_id",
            "status": "deleted",
        }


def test_delete_document_service_success():
    """Test successful document deletion at service level."""
    with patch("src.llm.services.settings") as mock_settings:
        mock_settings.qdrant.collection_name = "test_collection"

        mock_vectorstore = MagicMock()
        mock_client = MagicMock()
        mock_vectorstore.client = mock_client

        result = delete_document("test_doc_id", mock_vectorstore)

        assert result == {"document_id": "test_doc_id", "status": "deleted"}
        mock_client.delete.assert_called_once()
