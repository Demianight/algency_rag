from fastapi import APIRouter

from apps.llm.services import ask_gpt

from .schemas import AskRequest, AskResponse, SourceChunk

router = APIRouter()


@router.post(
    "/ask",
    response_model=AskResponse,
)
async def ask_question(question: AskRequest):
    """
    Endpoint to ask a question.
    """
    response = ask_gpt(question.question, top_k=question.top_k)
    return AskResponse(
        result=response["result"],
        source_documents=[
            SourceChunk(
                id=item.metadata["id"],
                document_id=item.metadata["document_id"],
                text=item.page_content,
            )
            for item in response["source_documents"]
        ],
    )
