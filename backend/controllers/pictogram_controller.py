from fastapi import APIRouter
from backend.schemas.pictogram_schemas import (
    PictogramGenerationRequest,
    PictogramGenerationResponse,
    PictogramData,
)
from backend.schemas.text_schemas import TextGenerationRequest
from backend.services.pictogram_service import PictogramGenerationService
from backend.services.text_service import TextGenerationService

router = APIRouter(prefix="/pictogram", tags=["Pictogram Generation"])
pictogram_service = PictogramGenerationService()
text_service = TextGenerationService()


@router.post("/generate", response_model=PictogramGenerationResponse)
async def generate_pictogram(request: PictogramGenerationRequest):
    """Genera pictogramas a partir de un texto ya provisto por el cliente.

    Nota: el servicio puede devolver un dict (forma serializable). Para evitar
    errores de validación de Pydantic, convertimos explícitamente a
    `PictogramData` antes de construir la response model.
    """
    pictogram_data = pictogram_service.generate_pictograms(text=request.text)

    # Aceptar tanto dict como ya-instanciado PictogramData
    if isinstance(pictogram_data, dict):
        pictogram = PictogramData(**pictogram_data)
    else:
        # Si el servicio ya devolviera un PictogramData/Pydantic model
        pictogram = pictogram_data

    return PictogramGenerationResponse(pictogram_data=pictogram)


@router.post("/from_prompt")
async def generate_pictogram_from_prompt(request: TextGenerationRequest):
    """
    Flujo simple que toma un `prompt` (igual que el endpoint de texto),
    genera la historia usando el modelo de texto y luego genera pictogramas
    para la historia completa. Devuelve la historia y los pictogramas.
    """
    # 1) Generar la historia usando el servicio de texto
    story = text_service.generate_story(
        prompt=request.prompt,
        max_tokens=request.max_tokens,
        tone=request.tone,
        complexity=request.complexity,
        sensory_friendly=request.sensory_friendly,
        story_type=request.story_type,
        protagonist_name=request.protagonist_name or "Protagonist",
    )

    # 2) Generar pictogramas a partir del texto generado
    pictogram_data = pictogram_service.generate_pictograms(text=story)

    # Normalizar la forma de los pictogramas a PictogramData para consistencia
    if isinstance(pictogram_data, dict):
        pictogram = PictogramData(**pictogram_data)
    else:
        pictogram = pictogram_data

    # 3) Devolver ambos para que el cliente pueda mostrar la historia y las imágenes
    return {"story": story, "pictograms": pictogram.dict()}
