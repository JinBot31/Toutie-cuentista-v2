from pydantic import BaseModel, Field
from typing import List


class PictogramItem(BaseModel):
    id: int = Field(..., description="Identificador secuencial del pictograma")
    sentence: str = Field(..., description="Frase original de la historia")
    concept: str = Field(..., description="Concepto visual extraído")
    image: str = Field(..., description="Imagen en base64")


class PictogramData(BaseModel):
    paragraph: str = Field(..., description="Texto completo analizado")
    items: List[PictogramItem] = Field(..., description="Lista de pictogramas generados")


class PictogramGenerationRequest(BaseModel):
    text: str = Field(..., description="Texto para generar pictogramas")


class PictogramGenerationResponse(BaseModel):
    pictogram_data: PictogramData = Field(..., description="Datos de pictogramas generados")