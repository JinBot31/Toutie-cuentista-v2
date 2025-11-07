# Cuentista Interactivo para Niños con Autismo

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.11.9-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-GPL%20v3-orange)](LICENSE)
[![IA Local](https://img.shields.io/badge/IA%20Local-100%25-purple)]()

**Cuentista Touie** es una plataforma educativa de código abierto diseñada para apoyar el aprendizaje y la comunicación de niños con autismo mediante **tres inteligencias artificiales locales**:

- **Generación de texto adaptado** (Modelo: Qwen2.5-1.5B-Instruct)
- **Síntesis de voz natural** (Modelo: XTTS v2 con clonación de voz)
- **Representación visual mediante pictogramas**

Este sistema transforma historias personalizadas en narraciones adaptadas, acompañadas de audio y representaciones visuales, con el objetivo de mejorar la accesibilidad, comprensión y experiencia de aprendizaje.

---

## Filosofía del Proyecto

**Cuentista Touie** se fundamenta en los **tres pilares de CiberPaz**, un movimiento por la paz digital y el acceso equitativo a la tecnología:

### 1. Acceso Universal y Conocimiento Libre

- **100% código abierto** bajo licencia GPLv3
- **Sin barreras económicas**: totalmente gratuito para uso educativo, terapéutico y personal
- **Modelos de IA ejecutados localmente**: sin dependencias de servicios externos ni suscripciones
- **Sin recopilación de datos personales**: privacidad absoluta garantizada
- **Documentación completa y transparente**: cualquiera puede auditar, modificar y mejorar el código

### 2. Apoyo a Comunidades Vulnerables

- **Diseñado con y para la comunidad autista**: respetando la neurodiversidad como forma válida de cognición
- **Eliminación de barreras de comunicación**: adaptación sensorial, lingüística y cognitiva
- **Inclusión educativa real**: herramientas que se adaptan al usuario, no al revés
- **Empoderamiento mediante tecnología**: facilitando la autonomía y expresión personal
- **Accesibilidad prioritaria**: diseño universal que beneficia a todas las personas

### 3. Desarrollo Tecnológico Ético y Sostenible

- **Soberanía tecnológica**: herramientas que funcionan sin dependencias corporativas
- **Procesamiento local**: control total sobre los datos y el funcionamiento del sistema
- **Escalabilidad responsable**: optimizado para funcionar en hardware modesto
- **Sostenibilidad del proyecto**: comunidad activa y código mantenible
- **Conocimiento compartido**: contribuciones bienvenidas desde cualquier contexto geográfico o social

---

## Características Principales

### Generación de Historias Personalizadas

- Narrativas adaptadas con tres niveles de complejidad (simple, intermedio, avanzado)
- Tonos ajustables (calmo, energético, neutral)
- Configuración sensorial amigable para evitar estímulos intensos
- Tipos de historia: aventura, cotidiana, educativa, fantasía
- Soporte para personalización del protagonista

### Síntesis de Voz Natural

- Clonación de voz mediante muestras de referencia
- Selección aleatoria entre voces predefinidas
- Control de velocidad de reproducción (0.5x - 2.0x)
- Soporte multilingüe con énfasis en español

### Interfaz Web Intuitiva

- Diseño responsive adaptado a móviles y tablets
- Navegación simple con pictogramas
- Historial de historias generadas
- Vista previa en tiempo real

---

## Requisitos del Sistema

### Obligatorios

| Componente | Versión | Notas |
|------------|---------|-------|
| Python | 3.11.9 | **Versión exacta requerida** |
| pip | >= 23.0 | Para gestión de dependencias |
| Sistema Operativo | Linux / macOS / Windows | Probado en Fedora 40+ |

### Opcionales (para acelerar procesamiento)

| Componente | Requisitos | Beneficios |
|------------|-----------|-----------|
| GPU NVIDIA | 4GB+ VRAM | Acelera generación 5-10x |
| CUDA Toolkit | 11.8+ | Requerido para aceleración GPU |
| cuDNN | Compatible con CUDA | Optimización adicional |

> **NOTA**: Sin GPU, el sistema funciona completamente pero con tiempos de generación más largos (30-60s vs 5-10s).

---

## Instalación

### 1. Clonar el Repositorio

```bash
git clone https://github.com/tu-organizacion/CiberPaz-2025.git
cd CiberPaz-2025
```

### 2. Verificar Versión de Python

```bash
python --version  # Debe mostrar Python 3.11.9
```

> **ADVERTENCIA**: El proyecto **NO funcionará** con versiones de Python distintas a 3.11.9 debido a incompatibilidades con dependencias de transformers y TTS.

### 3. Instalar uv (Gestor de Paquetes Recomendado)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

O con pip:

```bash
pip install uv
```

### 4. Sincronizar Dependencias

```bash
uv sync
```

> **SUGERENCIA**: Si encuentras errores de dependencias, ejecuta `uv sync --reinstall` para forzar reinstalación limpia.

### 5. Verificar Instalación de Audio (Opcional)

Si experimentas problemas con la generación de voz:

```bash
uv pip install soundfile torchaudio torchcodec --force-reinstall
```

---

## Ejecución

### Iniciar el Backend (API)

```bash
uv run python main.py
```

O con uvicorn directamente:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

La API estará disponible en:
- **Interfaz principal**: http://localhost:8000
- **Documentación interactiva (Swagger)**: http://localhost:8000/docs
- **Documentación alternativa (ReDoc)**: http://localhost:8000/redoc

### Acceder a la Interfaz Web

Opción 1 - Servidor integrado:
```bash
# La interfaz ya está servida por FastAPI en http://localhost:8000
```

Opción 2 - Servidor independiente:
```bash
cd frontend/static
python -m http.server 9090
```

Luego abrir en navegador: http://localhost:9090

---

## Estructura del Proyecto

```
CiberPaz-2025/
├── main.py                      # Punto de entrada de la aplicación
├── pyproject.toml               # Configuración de dependencias
├── LICENSE                      # Licencia GPLv3
├── README.md                    # Este archivo
│
├── backend/
│   ├── config/
│   │   └── settings.py          # Configuración global
│   ├── controllers/             # Endpoints de la API
│   │   ├── text_controller.py
│   │   ├── voice_controller.py
│   │   └── pictogram_controller.py
│   ├── services/                # Lógica de negocio
│   │   ├── text_service.py
│   │   ├── voice_service.py
│   │   └── pictogram_service.py
│   ├── models/                  # Modelos de IA
│   │   ├── text_model.py
│   │   ├── voice_model.py
│   │   ├── pictogram_model.py
│   │   └── voices/              # Muestras de voz para clonación
│   └── schemas/                 # Validación de datos (Pydantic)
│       ├── text_schemas.py
│       ├── voice_schemas.py
│       └── pictogram_schemas.py
│
├── frontend/
│   └── static/
│       ├── index.html           # Interfaz principal
│       ├── endpoints.js         # Lógica de conexión con API
│       └── images/              # Recursos visuales
│
└── resources/
    └── audio/
        └── output/              # Audios generados (creados automáticamente)
```

---

## Uso de la API

### Generar Historia de Texto

**Endpoint**: `POST /text/generate`

```bash
curl -X POST "http://localhost:8000/text/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Escribe una historia sobre un niño llamado Giovanni que descubre un jardín mágico",
    "max_tokens": 300,
    "tone": "calmo",
    "complexity": "simple",
    "sensory_friendly": true,
    "story_type": "aventura",
    "protagonist_name": "Giovanni"
  }'
```

### Generar Audio

**Endpoint**: `POST /voice/generate`

```bash
curl -X POST "http://localhost:8000/voice/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Había una vez un niño llamado Giovanni...",
    "language": "es",
    "voice_speed": 1.0
  }'
```

> **SUGERENCIA**: La voz se selecciona aleatoriamente entre las disponibles en `backend/models/voices/`. Para usar una voz específica, añade el parámetro `"speaker_wav": "ruta/a/voz.wav"`.

### Generar Pictogramas

**Endpoint**: `POST /pictogram/generate`

```bash
curl -X POST "http://localhost:8000/pictogram/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "El niño juega en el parque"
  }'
```

> **NOTA**: El módulo de pictogramas está en desarrollo activo.

---

## Configuración Avanzada

### Variables de Entorno

Crear archivo `.env` en la raíz del proyecto:

```env
# Información de la aplicación
APP_NAME="Cuentista para Autistas"
APP_VERSION="0.1.0"

# Modelos de IA
TEXT_MODEL_CHECKPOINT="Qwen/Qwen2.5-1.5B-Instruct"
VOICE_MODEL_NAME="tts_models/multilingual/multi-dataset/xtts_v2"

# Rutas de recursos
AUDIO_OUTPUT_DIR="resources/audio/output"

# CORS (orígenes permitidos)
CORS_ORIGINS='["http://localhost", "http://localhost:9090", "http://localhost:8000"]'
```

### Agregar Nuevas Voces

1. Colocar archivos WAV de alta calidad (16kHz+, mono o estéreo) en `backend/models/voices/`
2. Reiniciar el servidor
3. Las voces estarán disponibles automáticamente para selección aleatoria

> **SUGERENCIA**: Para mejores resultados, usa muestras de audio de 3-10 segundos con voz clara y sin ruido de fondo.

---
