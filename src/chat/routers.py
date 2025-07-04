from fastapi import APIRouter, Depends

from src.llm.dependencies import get_llm, get_vectorstore
from src.llm.services import ask_gpt
from src.settings import settings

from .schemas import AskRequest, AskResponse, SourceChunk

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post(
    "/ask",
    response_model=AskResponse,
)
async def ask_question(
    question: AskRequest,
    llm=Depends(get_llm),
    vectorstore=Depends(get_vectorstore),
):
    """
    Endpoint to ask a question.
    """
    response = ask_gpt(question.question, llm, vectorstore, top_k=question.top_k)

    if settings.debug:
        print(f"Response: {response}")

    return AskResponse(
        result=response["result"],
        source_documents=[
            SourceChunk(
                id=item.metadata["_id"],
                document_id=item.metadata["document_id"],
                text=item.page_content,
                tags=item.metadata.get("tags", []),
            )
            for item in response["source_documents"]
        ],
    )
