# schemas.py
from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str
    pdf_id: str

class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[dict]] = None

class UploadResponse(BaseModel):
    message: str
    pdf_id: str
    filename: str