import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.controllers import text_router, voice_router, pictogram_router
from backend.config import settings


app = FastAPI(
    title=settings.app_name,
    description="API para generacion de contenido adaptado a ninos con autismo usando 3 IAs locales",
    version=settings.app_version
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(text_router)
app.include_router(voice_router)
app.include_router(pictogram_router)


@app.get("/")
async def root():
    return {
        "message": "Cuentista para Autistas API",
        "version": settings.app_version,
        "endpoints": {
            "text": "/text/generate",
            "voice": "/voice/generate",
            "pictogram": "/pictogram/generate"
        }
    }


def main():
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )


if __name__ == "__main__":
    main()
