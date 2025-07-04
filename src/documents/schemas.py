from pydantic import BaseModel


class Chunk(BaseModel):
    document_id: str
    type: str
    text: str
    source: str
    tags: list[str] | None = None
