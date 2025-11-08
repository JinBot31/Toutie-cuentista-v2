"""
Pydantic schemas for pictogram generation and response.
Defines data structures for requests and responses of the pictogram system.
"""
from pydantic import BaseModel, Field
from typing import List


class PictogramItem(BaseModel):
    """
    Represents a pictogram generated from a sentence.
    Includes identifier, original sentence, visual concept, and base64 image.
    """
    id: int = Field(..., description="Identificador secuencial del pictograma")
    sentence: str = Field(..., description="Frase original de la historia")
    concept: str = Field(..., description="Concepto visual extraído")
    image: str = Field(..., description="Imagen en base64")


class PictogramData(BaseModel):
    """
    Contains the complete analyzed text and the list of generated pictograms.
    """
    paragraph: str = Field(..., description="Texto completo analizado")
    items: List[PictogramItem] = Field(..., description="Lista de pictogramas generados")


class PictogramGenerationRequest(BaseModel):
    """
    Request to generate pictograms from text.
    """
    text: str = Field(..., description="Texto para generar pictogramas")


class PictogramGenerationResponse(BaseModel):
    """
    Response containing generated pictogram data.
    """
    pictogram_data: PictogramData = Field(..., description="Datos de pictogramas generados")