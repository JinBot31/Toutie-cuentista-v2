import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Cuentista para Autistas"
    app_version: str = "0.1.0"
    
    text_model_checkpoint: str = "Qwen/Qwen2.5-1.5B-Instruct"
    voice_model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    
    default_speaker_wav: str = os.path.join(os.getcwd(), "resources/audio/speaker.wav")
    audio_output_dir: str = os.path.join(os.getcwd(), "resources/audio/output")
    
    cors_origins: list = [
        "http://localhost",
        "http://localhost:8000",
        "http://localhost:9090",
        "http://localhost:5500",
        "http://127.0.0.1",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:9090",
        "http://127.0.0.1:5500",
    ]
    
    class Config:
        env_file = ".env"

settings = Settings()
