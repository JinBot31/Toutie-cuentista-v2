from pydantic import BaseModel, Field
from typing import List, Optional


class PictogramGenerationRequest(BaseModel):
    text: str = Field(..., description="Texto para generar pictogramas")


class PictogramItem(BaseModel):
    id: int
    label: str
    image: str
    alt: Optional[str] = None


class PictogramData(BaseModel):
    paragraph: str
    items: List[PictogramItem] = []


class PictogramGenerationResponse(BaseModel):
    pictogram_data: PictogramData = Field(..., description="Datos del pictograma generado")
