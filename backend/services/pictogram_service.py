from backend.models.pictogram_model import PictogramGenerationModel


class PictogramGenerationService:
    def __init__(self):
        self.model = PictogramGenerationModel()
    
    def generate_pictograms(self, text: str) -> str:
        return self.model.generate(text)
