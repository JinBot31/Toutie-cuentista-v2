"""Configuración de la aplicación.

Este módulo define la clase `Settings` basada en `pydantic_settings.BaseSettings`.
Los valores pueden cargarse desde variables de entorno (por ejemplo, `APP_NAME`) o desde
un archivo `.env` (configurado en la clase interna `Config`).

La clase contiene valores por defecto pensados para desarrollo local; cambie las
variables de entorno o el `.env` para personalizar en producción.

Ejemplo de uso:
        from backend.config.settings import settings
        print(settings.app_name)

Convenciones:
 - Los nombres de las variables de entorno siguen la transformación por defecto de Pydantic
     (por ejemplo, `app_name` -> `APP_NAME`).
 - No se realizan operaciones costosas en el módulo: `settings = Settings()` se instancia al
     final para que otras partes de la aplicación la importen y usen.
"""
import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
        """Ajustes de la aplicación.

        Atributos (con sus valores por defecto):

        - app_name (str): Nombre legible de la aplicación. Por defecto: "Cuentista para Autistas".
        - app_version (str): Versión de la aplicación. Por defecto: "0.1.0".
        - text_model_checkpoint (str): Identificador/checkpoint del modelo de texto usado.
        - voice_model_name (str): Nombre del modelo de voz para TTS.
        - default_speaker_wav (str): Ruta al archivo .wav usado como altavoz por defecto.
        - audio_output_dir (str): Directorio donde se guardan los audios generados.
        - cors_origins (list): Lista de orígenes permitidos para CORS en desarrollo.

        Notas:
        - Cada atributo puede ser sobrescrito con la variable de entorno correspondiente siguiendo
            la convención de Pydantic (por ejemplo `APP_NAME` para `app_name`).
        - Esta clase hereda de `BaseSettings`, por lo que valida y carga las variables de entorno
            automáticamente.

        Ejemplo:

                # usando variables de entorno
                export APP_NAME="Mi App"
                export AUDIO_OUTPUT_DIR="/tmp/audio"
                from backend.config.settings import settings
                print(settings.app_name)  # -> "Mi App"
"""
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
        # URL base del servicio de pictogramas. Se puede configurar con la variable
        # de entorno PICTOGRAMS_API_BASE_URL o sobrescribir en un archivo .env.
        pictograms_api_base_url: str = "https://api.arasaac.org/api/"
        class Config:
                """Configuración adicional de Pydantic para `Settings`.
                - env_file: nombre del archivo desde el que se cargarán variables de entorno si existe.
                """
                env_file = ".env"

settings = Settings()
