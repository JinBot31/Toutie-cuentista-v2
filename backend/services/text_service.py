"""
Service for generating personalized stories.
Builds prompts and uses the text model to generate narratives adapted to specific parameters.
"""
from backend.models.text_model import TextGenerationModel


class TextGenerationService:
    """
    Service that builds prompts and generates personalized stories using a text model.
    Allows adjusting tone, complexity, story type, and other parameters.
    """
    def __init__(self):
        self.model = TextGenerationModel()
        
        self.tone_instructions = {
            "calmo": "Usa un tono tranquilo y pausado. Evita conflictos o tensiones.",
            "energico": "Usa un tono dinamico y emocionante.",
            "neutral": "Usa un tono equilibrado."
        }
        
        self.complexity_instructions = {
            "simple": "Usa oraciones cortas (maximo 8 palabras) y vocabulario basico.",
            "intermedio": "Usa oraciones de longitud media con vocabulario accesible.",
            "avanzado": "Puedes usar estructuras complejas y vocabulario variado."
        }
    
    def generate_story(
        self,
        prompt: str,
        max_tokens: int = 120,
        tone: str = "calmo",
        complexity: str = "simple",
        sensory_friendly: bool = True,
        story_type: str = "cotidiana",
        protagonist_name: str = ""
    ) -> str:
        """
        Generates a personalized story from a prompt and optional parameters.

        Args:
            prompt (str): Initial prompt for the story.
            max_tokens (int): Maximum tokens to generate.
            tone (str): Tone of the narrative.
            complexity (str): Level of language complexity.
            sensory_friendly (bool): Whether to avoid intense sensory stimuli.
            story_type (str): Type of story.
            protagonist_name (str): Name of the protagonist.

        Returns:
            str: Generated story.
        """
        full_prompt = self._build_prompt(
            prompt, tone, complexity, sensory_friendly, story_type, protagonist_name
        )
        
        return self.model.generate(full_prompt, max_new_tokens=max_tokens)
    
    def _build_prompt(
        self,
        user_prompt: str,
        tone: str,
        complexity: str,
        sensory_friendly: bool,
        story_type: str,
        protagonist_name: str
    ) -> str:
        instructions = [
            f"Story type: {story_type}.",
            self.tone_instructions.get(tone, ""),
            self.complexity_instructions.get(complexity, ""),
        ]

        if sensory_friendly:
            instructions.append(
                "Avoid loud noises, bright lights, or unpleasant smells."
            )

        instructions.append("Use literal language and avoid complex metaphors.")

        if protagonist_name:
            instructions.append(f"The protagonist is called {protagonist_name}.")

        context = " ".join(instructions)

        return f"{context}\n\n{user_prompt}"