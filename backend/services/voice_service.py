from backend.models.voice_model import VoiceGenerationModel
from datetime import datetime
import os


class VoiceGenerationService:
    def __init__(self):
        self.model = VoiceGenerationModel()
    
    def generate_audio(
        self, 
        text: str, 
        speaker_wav: str = "", 
        language: str = "es",
        speed: float = 1.0
    ) -> dict:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"resources/audio/output/story_{timestamp}.wav"
        
        voice_used = speaker_wav if speaker_wav else self.model.get_random_voice()
        
        audio_path = self.model.generate(
            text=text,
            speaker_wav=voice_used,
            language=language,
            output_path=output_path,
            speed=speed
        )
        
        duration = self._get_audio_duration(audio_path)
        
        return {
            "audio_path": audio_path,
            "duration": duration,
            "voice_used": os.path.basename(voice_used)
        }
    
    def _get_audio_duration(self, audio_path: str) -> float:
        try:
            import wave
            with wave.open(audio_path, 'r') as audio:
                frames = audio.getnframes()
                rate = audio.getframerate()
                duration = frames / float(rate)
                return round(duration, 2)
        except Exception:
            return 0.0