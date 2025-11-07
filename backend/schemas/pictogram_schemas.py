from pydantic import BaseModel, Field


class PictogramGenerationRequest(BaseModel):
    text: str = Field(..., description="Texto para generar pictogramas")


class PictogramGenerationResponse(BaseModel):
    pictogram_data: str = Field(..., description="Datos del pictograma generado")
