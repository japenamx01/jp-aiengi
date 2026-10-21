from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.llm_service import generate_estimation
from app.config import settings

router = APIRouter(prefix="/api/v1", tags=["estimations"])

class EstimationRequest(BaseModel):
    transcription: str

class EstimationResponse(BaseModel):
    estimation: str
    model: str
    provider: str

@router.post("/estimate", response_model=EstimationResponse)
def create_estimation(payload: EstimationRequest):
    try:
        result = generate_estimation(payload.transcription)
        return EstimationResponse(
            estimation=result,
            model=settings.LLM_MODEL,
            provider=settings.LLM_PROVIDER
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

