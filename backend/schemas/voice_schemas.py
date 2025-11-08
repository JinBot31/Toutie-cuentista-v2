"""
Pydantic schemas for voice generation and response.
Defines data structures for requests and responses of the audio generation system.
"""
from pydantic import BaseModel, Field
from typing import Optional

class VoiceGenerationRequest(BaseModel):
    """
    Request to generate audio from text and voice parameters.
    """
    text: str = Field(..., description="Texto a convertir en audio")
    language: str = Field(default="es", description="Idioma del audio")
    voice_speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Velocidad de la voz")
    speaker_wav: Optional[str] = Field(None, description="Ruta del archivo de referencia de voz (opcional, si no se especifica usa una aleatoria)")

class VoiceGenerationResponse(BaseModel):
    """
    Response containing the path of the generated audio, duration, and voice used.
    """
    audio_path: str = Field(..., description="Ruta del archivo de audio generado")
    duration: Optional[float] = Field(None, description="Duracion del audio en segundos")
    voice_used: str = Field(..., description="Voz utilizada para la generacion")