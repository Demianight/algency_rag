# mypy: disable-error-code=import
from unittest.mock import MagicMock, patch

from src.llm.services import ask_gpt


def test_ask_gpt_returns_expected_result():
    mock_retrieval_qa = MagicMock()
    mock_invoke = mock_retrieval_qa.invoke
    mock_invoke.return_value = {"result": "answer", "source_documents": [1, 2]}

    with patch("src.llm.services.RetrievalQA") as mock_chain:
        mock_chain.from_chain_type.return_value = mock_retrieval_qa
        result = ask_gpt(
            "question", llm="llm", vectorstore=MagicMock(), top_k=3
        )

    mock_chain.from_chain_type.assert_called_once_with(
        llm="llm",
        chain_type="stuff",
        retriever=mock_chain.from_chain_type.call_args[1]["retriever"],
        return_source_documents=True,
    )
    mock_invoke.assert_called_once_with({"query": "question"})
    assert result == {"result": "answer", "source_documents": [1, 2]}
