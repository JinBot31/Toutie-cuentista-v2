from fastapi import APIRouter
from backend.schemas.voice_schemas import VoiceGenerationRequest, VoiceGenerationResponse
from backend.services.voice_service import VoiceGenerationService

router = APIRouter(prefix="/voice", tags=["Voice Generation"])
voice_service = VoiceGenerationService()

@router.post("/generate", response_model=VoiceGenerationResponse)
async def generate_voice(request: VoiceGenerationRequest):
    result = voice_service.generate_audio(
        text=request.text,
        speaker_wav=request.speaker_wav or "",
        language=request.language,
        speed=request.voice_speed
    )
    return VoiceGenerationResponse(**result)