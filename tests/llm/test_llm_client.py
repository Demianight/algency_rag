# mypy: disable-error-code=import
from unittest.mock import MagicMock, patch

from src.llm.dependencies import get_embeddings, get_llm, get_vectorstore


def test_get_llm_instantiates_chatopenai():
    with (
        patch("src.llm.dependencies.ChatOpenAI") as mock_chatopenai,
        patch("src.llm.dependencies.settings") as mock_settings,
        patch("src.llm.dependencies.SecretStr") as mock_secretstr,
    ):
        mock_settings.openai.qa_model = "model"
        mock_settings.openai.api_key = "key"
        get_llm()
        mock_chatopenai.assert_called_once_with(
            model="model",
            api_key=mock_secretstr.return_value,
        )
        mock_secretstr.assert_called_once_with("key")


def test_get_embeddings_instantiates_openaiembeddings():
    with (
        patch("src.llm.dependencies.OpenAIEmbeddings") as mock_embeddings,
        patch("src.llm.dependencies.settings") as mock_settings,
        patch("src.llm.dependencies.SecretStr") as mock_secretstr,
    ):
        mock_settings.openai.embed_model = "embed_model"
        mock_settings.openai.api_key = "key"
        get_embeddings()
        mock_embeddings.assert_called_once_with(
            model="embed_model",
            api_key=mock_secretstr.return_value,
        )
        mock_secretstr.assert_called_once_with("key")


def test_get_vectorstore_instantiates_qdrantvectorstore():
    with (
        patch("src.llm.dependencies.QdrantClient") as mock_client,
        patch("src.llm.dependencies.QdrantVectorStore") as mock_vectorstore,
        patch("src.llm.dependencies.settings") as mock_settings,
    ):
        mock_settings.qdrant.host = "host"
        mock_settings.qdrant.port = 1234
        mock_settings.qdrant.collection_name = "collection"
        embeddings = MagicMock()
        result = get_vectorstore(embeddings=embeddings)
        mock_client.assert_called_once_with(host="host", port=1234)
        mock_vectorstore.assert_called_once_with(
            client=mock_client.return_value,
            embedding=embeddings,
            collection_name="collection",
        )
        assert result == mock_vectorstore.return_value
