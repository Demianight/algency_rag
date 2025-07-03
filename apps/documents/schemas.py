from pydantic import BaseModel


class Chunk(BaseModel):
    id: str
    document_id: str
    type: str
    text: str
    source: str
