from pydantic import BaseModel, Field
from typing import Optional, Literal


class TextGenerationRequest(BaseModel):
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
    text: str = Field(..., description="Texto generado")
    metadata: dict = Field(
        default_factory=dict,
        description="Metadatos sobre la generacion (tokens usados, etc)"
    )