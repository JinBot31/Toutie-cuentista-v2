from backend.models.text_model import TextGenerationModel


class TextGenerationService:
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
            f"Tipo de historia: {story_type}.",
            self.tone_instructions.get(tone, ""),
            self.complexity_instructions.get(complexity, ""),
        ]
        
        if sensory_friendly:
            instructions.append(
                "Evita ruidos fuertes, luces intensas u olores desagradables."
            )
        
        instructions.append("Usa lenguaje literal y evita metaforas complejas.")
        
        if protagonist_name:
            instructions.append(f"El protagonista se llama {protagonist_name}.")
        
        context = " ".join(instructions)
        
        return f"{context}\n\n{user_prompt}"