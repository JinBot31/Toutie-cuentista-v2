# 🌈 CiberPaz 2025 – Cuentista Interactivo para Niños con Autismo

<p align="center">
  <img src="https://img.shields.io/badge/FastAPI-0.115-green" />
  <img src="https://img.shields.io/badge/Python-3.11-blue" />
  <img src="https://img.shields.io/badge/IA%20Local-Sí-purple" />
</p>

El proyecto **CiberPaz 2025** es una aplicación diseñada para apoyar el aprendizaje y la comunicación de niños con autismo a través de **tres inteligencias artificiales locales**:

- 🧠 **Generación de texto** (Modelo: Qwen)
- 🗣️ **Síntesis de voz** (Modelo: XTTS v2)
- 🖼️ **Representación visual** mediante pictogramas

Este sistema transforma historias en narraciones adaptadas, acompañadas de audio y representaciones visuales, con el fin de mejorar la accesibilidad y comprensión.

---

## 📂 Estructura del Proyecto

CiberPaz-2025/

│ main.py

│ pyproject.toml

│ .env (opcional)

│

├── backend/

│ ├── config/settings.py

│ ├── controllers/

│ ├── services/

│ └── models/

│

├── frontend/

│ └── static/

│ ├── index.html

│ ├── script.js

│ └── style.css

│

└── resources/

└── audio/output/ # Aquí se guardan los audios generados

---

## ⚙️ Requisitos Previos

| Software / Requisito | Versión |
|----------------------|---------|
| Python               | **3.11 (Obligatorio)** |
| pip                  | Última versión |
| Torch + CUDA (Opcional) | Para acelerar procesamiento en GPU |
| GPU NVIDIA (Opcional) | Mejora tiempos de generación de texto y voz |

> **Sin GPU → Funciona igual, solo más lento.**

---

## 🚀 Instalación

```bash
### 1. Clonar el repositorio
git clone https://github.com/tu-org/CiberPaz-2025.git
cd CiberPaz-2025
### 2.Crear entorno virtual 
python3.11 -m venv venv

linux/mac 

source venv/bin/activate

windows 

venv/Scripts/activate 

### 3.Instalar dependencias 

pip install --upgrade pip
pip install -e .

En caso de error con el audio ejecutar:

pip install soundfile TTS torchaudio


### Ejecutar backend(API) 

python main.py

La api se ejecutará en: 

http://localhost:8000

### Ejecutar la interfaz web 

cd frontend/static
python -m http.server 9090

Luego abrir en navegador:

http://localhost:9090

### Variables de entorno (Opcionales)

Crear un archivo .env en la raíz y colocar 

APP_NAME="Cuentista para Autistas"
VOICE_MODEL="tts_models/multilingual/multi-dataset/xtts_v2"
