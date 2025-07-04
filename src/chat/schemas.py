from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


class SourceChunk(BaseModel):
    id: str
    document_id: str
    text: str


class AskResponse(BaseModel):
    result: str
    source_documents: list[SourceChunk]
