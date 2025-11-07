from TTS.api import TTS
import os
import random
import torch
from pathlib import Path

class VoiceGenerationModel:
    def __init__(self, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"):
        self.model_name = model_name
        #self.tts = None
        self.voices_dir = Path(__file__).parent / "voices"
        self.available_voices = list(self.voices_dir.glob("*.wav"))
        
        self._patch_torch_load()
    
    def _patch_torch_load(self):
        original_load = torch.load
        def patched_load(*args, **kwargs):
            kwargs.setdefault('weights_only', False)
            return original_load(*args, **kwargs)
        torch.load = patched_load
    
    def load_model(self):
        if self.tts is None:
            self.tts = TTS(self.model_name, gpu=True)
    
    def get_random_voice(self) -> str:
        if not self.available_voices:
            raise ValueError("No hay archivos de voz disponibles en backend/models/voices/")
        return str(random.choice(self.available_voices))
    
    def generate(
        self, 
        text: str, 
        speaker_wav: str = "", 
        language: str = "es", 
        output_path: str = "",
        speed: float = 1.0
    ) -> str:
        if self.tts is None:
            self.load_model()
        
        if speaker_wav is None:
            speaker_wav = self.get_random_voice()
        
        if output_path is None:
            output_path = "resources/audio/output/generated_audio.wav"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        self.tts.tts_to_file(
            text=text,
            file_path=output_path,
            speaker_wav=speaker_wav,
            language=language,
            speed=speed
        )
        return output_path