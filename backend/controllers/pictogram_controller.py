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
    """Generate pictograms from provided text.

    Note: The service may return a dict (serializable form). To avoid
    Pydantic validation errors, we explicitly convert to
    `PictogramData` before building the response model.

    Args:
        request (PictogramGenerationRequest): The request containing the text.

    Returns:
        PictogramGenerationResponse: The response with generated pictograms.
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
    """Generate pictograms from a prompt by first generating text.

    Simple flow that takes a `prompt` (same as text endpoint),
    generates the story using the text model and then generates pictograms
    for the complete story. Returns the story and the pictograms.

    Args:
        request (TextGenerationRequest): The request containing the prompt and parameters.

    Returns:
        dict: A dictionary containing the generated story and pictograms.
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
