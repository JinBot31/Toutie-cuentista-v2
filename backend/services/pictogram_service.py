"""
Service for pictogram generation.
Uses the pictogram model to process text and return generated pictograms.
"""

from typing import Dict, Any

from backend.models.pictogram_model import PictogramGenerationModel


class PictogramGenerationService:
    """
    Service that uses the pictogram model to generate pictograms from text.
    """

    def __init__(self):
        self.model = PictogramGenerationModel()

    def generate_pictograms(self, text: str) -> Dict[str, Any]:
        """
        Generates pictograms from a given text.

        Args:
            text (str): Input text to analyze and generate pictograms.

        Returns:
            Dict[str, Any]: Dictionary with the generated pictograms.
        """
        return self.model.generate(text)  # type: ignore
