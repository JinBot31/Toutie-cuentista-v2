from fastapi import APIRouter
from backend.schemas.pictogram_schemas import PictogramGenerationRequest, PictogramGenerationResponse
from backend.services.pictogram_service import PictogramGenerationService

router = APIRouter(prefix="/pictogram", tags=["Pictogram Generation"])
pictogram_service = PictogramGenerationService()


@router.post("/generate", response_model=PictogramGenerationResponse)
async def generate_pictogram(request: PictogramGenerationRequest):
    pictogram_data = pictogram_service.generate_pictograms(text=request.text)
    return PictogramGenerationResponse(pictogram_data=pictogram_data)
