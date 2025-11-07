from .text_controller import router as text_router
from .voice_controller import router as voice_router
from .pictogram_controller import router as pictogram_router

__all__ = [
    "text_router",
    "voice_router",
    "pictogram_router"
]
