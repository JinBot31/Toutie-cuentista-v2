import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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
# Serve frontend files from the `frontend/view` folder (this repo uses `frontend/view/index.html`)
try:
    app.mount("/frontend/static", StaticFiles(directory="frontend/view"), name="static")
except Exception:
    # If the folder is not present, skip static mounting to keep the API running
    pass


@app.get("/")
def serve_index():
    # Serve the shipped index if present, otherwise return a simple JSON welcome
    import os
    index_path = "frontend/view/index.html"
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Cuentista para Autistas API", "version": settings.app_version}

@app.get("/info")
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
