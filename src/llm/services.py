from fastapi import HTTPException
from langchain.chains import RetrievalQA
from langchain.vectorstores import VectorStore
from qdrant_client.http.models import FieldCondition, Filter, MatchValue
from qdrant_client.models import FilterSelector

from src.settings import settings


def ask_gpt(question: str, llm, vectorstore, top_k: int = 5) -> dict:
    """
    Ask a question to the LLM and return the answer along with source documents.
    """
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": top_k}),
        return_source_documents=True,
    ).invoke({"query": question})


def delete_document(doc_id: str, vectorstore: VectorStore) -> dict:
    """
    Delete a document and all its chunks from the vector store, based on metadata.
    """
    client = getattr(vectorstore, "client", None)
    if client is None:
        raise HTTPException(status_code=500, detail="Vectorstore has no Qdrant client")
    # Build metadata filter
    metadata_filter = Filter(
        must=[
            FieldCondition(key="metadata.document_id", match=MatchValue(value=doc_id))
        ]
    )
    try:
        client.delete(
            collection_name=settings.qdrant.collection_name,
            points_selector=FilterSelector(filter=metadata_filter),
        )
        return {"document_id": doc_id, "status": "deleted"}
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Document {doc_id} not found or could not be deleted: {e}",
        )
