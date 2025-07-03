from fastapi import APIRouter, UploadFile

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/ingest")
async def ingest_document(document: UploadFile):
    contents = await document.read()
    return {"filename": document.filename, "size": len(contents)}
