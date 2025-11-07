from fastapi import APIRouter
from backend.schemas.text_schemas import TextGenerationRequest, TextGenerationResponse
from backend.services.text_service import TextGenerationService

router = APIRouter(prefix="/text", tags=["Text Generation"])
text_service = TextGenerationService()


@router.post("/generate", response_model=TextGenerationResponse)
async def generate_text(request: TextGenerationRequest):
    generated_text = text_service.generate_story(
        prompt=request.prompt,
        max_tokens=request.max_tokens,
        tone=request.tone,
        complexity=request.complexity,
        sensory_friendly=request.sensory_friendly,
        story_type=request.story_type,
        protagonist_name=request.protagonist_name or "Protagonist"
    )
    
    return TextGenerationResponse(
        text=generated_text,
        metadata={
            "tone": request.tone,
            "complexity": request.complexity,
            "story_type": request.story_type
        }
    )