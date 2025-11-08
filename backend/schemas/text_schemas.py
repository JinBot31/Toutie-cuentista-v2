"""
Pydantic schemas for text generation and response.
Defines data structures for requests and responses of the story generation system.
"""
from pydantic import BaseModel, Field
from typing import Optional, Literal


class TextGenerationRequest(BaseModel):
    """
    Request to generate a story from a prompt and optional parameters.
    """
    prompt: str = Field(..., description="Prompt inicial para generar la historia")
    max_tokens: int = Field(default=120, ge=50, le=800, description="Cantidad maxima de tokens")
    
    tone: Literal["calmo", "energico", "neutral"] = Field(
        default="calmo",
        description="Tono de la narrativa"
    )
    
    complexity: Literal["simple", "intermedio", "avanzado"] = Field(
        default="simple",
        description="Nivel de complejidad del lenguaje"
    )
    
    sensory_friendly: bool = Field(
        default=True,
        description="Evitar descripciones con estimulos sensoriales intensos"
    )
    
    story_type: Literal["aventura", "cotidiana", "educativa", "fantasia"] = Field(
        default="cotidiana",
        description="Tipo de historia a generar"
    )
    
    protagonist_name: Optional[str] = Field(
        default="",
        description="Nombre del protagonista (opcional)"
    )


class TextGenerationResponse(BaseModel):
    """
    Response containing the generated text and associated metadata.
    """
    text: str = Field(..., description="Texto generado")
    metadata: dict = Field(
        default_factory=dict,
        description="Metadatos sobre la generacion (tokens usados, etc)"
    )