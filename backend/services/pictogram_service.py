from typing import Dict, Any

from backend.models.pictogram_model import PictogramGenerationModel


class PictogramGenerationService:
    def __init__(self):
        self.model = PictogramGenerationModel()

    def generate_pictograms(self, text: str) -> Dict[str, Any]:
        """Genera pictogramas y devuelve un dict serializable con la forma:
        {"paragraph": str, "items": [ {id, label, image}, ... ]}
        """
        return self.model.generate(text)  # type: ignore
