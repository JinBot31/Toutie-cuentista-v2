# backend/models/pictogram_model.py
import importlib
import os
import base64
from io import BytesIO
from typing import List, Dict, Any, Optional, cast
from PIL import Image, ImageDraw, ImageFont

_HAS_TORCH = True
_HAS_TRANSFORMERS = True
_HAS_DIFFUSERS = True
try:
    import torch
except Exception:
    torch = None
    _HAS_TORCH = False

try:
    transformers = importlib.import_module("transformers")
    AutoTokenizer = transformers.AutoTokenizer
    AutoModelForCausalLM = transformers.AutoModelForCausalLM
    set_seed = transformers.set_seed
except Exception:
    AutoTokenizer = None
    AutoModelForCausalLM = None
    set_seed = None
    _HAS_TRANSFORMERS = False

try:
    diffusers = importlib.import_module("diffusers")
    DiffusionPipeline = diffusers.DiffusionPipeline
except Exception:
    DiffusionPipeline = None
    _HAS_DIFFUSERS = False


class PictogramGenerationModel:
    def __init__(self):
        # use Any/Optional to keep static checkers happy when optional deps are missing
        self.text_model: Any = None
        self.tokenizer: Any = None
        self.image_pipe: Any = None
        self.device = "cpu"

        try:
            if torch is not None and torch.cuda.is_available():
                self.device = "cuda"
        except Exception:
            self.device = "cpu"
        self.TEXT_MODEL = os.environ.get("TEXT_MODEL_CHECKPOINT", "Qwen/Qwen2.5-1.5B-Instruct")
        self.IMAGE_MODEL = os.environ.get("IMAGE_MODEL_CHECKPOINT", "runwayml/stable-diffusion-v1-5")
        self.NUM_STEPS = int(os.environ.get("PCT_NUM_STEPS", 6))
        self.GUIDANCE_SCALE = float(os.environ.get("PCT_GUIDANCE_SCALE", 3.5))
        self.IMAGE_SIZE = (int(os.environ.get("PCT_IMAGE_W", 256)), int(os.environ.get("PCT_IMAGE_H", 256)))
        self.MAX_CONCEPTS = 5

        if set_seed is not None:
            try:
                set_seed(2024)
            except Exception:
                pass

    def load_model(self):
        """Carga ambos modelos si aún no están cargados. Si no están instaladas las dependencias,
        deja los atributos en None y el resto del código usará los fallbacks.
        """
        # Only attempt to load transformers objects if the optional dependency was imported
        if _HAS_TRANSFORMERS and AutoTokenizer is not None and AutoModelForCausalLM is not None and (
            self.text_model is None or self.tokenizer is None
        ):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.TEXT_MODEL, trust_remote_code=True)
                torch_dtype = None
                if torch is not None:
                    try:
                        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
                    except Exception:
                        torch_dtype = None

                load_kwargs = {}
                if torch_dtype is not None:
                    load_kwargs["torch_dtype"] = torch_dtype

                self.text_model = AutoModelForCausalLM.from_pretrained(
                    self.TEXT_MODEL, trust_remote_code=True, **load_kwargs
                )

                # Try to move model to device if CUDA available
                if self.device == "cuda" and torch is not None:
                    try:
                        self.text_model = self.text_model.to(self.device)
                    except Exception:
                        pass
            except Exception:
                print(f"Warning: could not load text model {self.TEXT_MODEL}. Using fallbacks.")
                try:
                    import traceback

                    traceback.print_exc()
                except Exception:
                    pass
                self.tokenizer = None
                self.text_model = None
        enable_heavy = os.environ.get("ENABLE_HEAVY_MODELS", "0") == "1"
        # Only use diffusers if it was successfully imported
        if enable_heavy and _HAS_DIFFUSERS and DiffusionPipeline is not None and self.image_pipe is None:
            try:
                import torch as _torch

                torch_dtype = _torch.float16 if self.device == "cuda" else _torch.float32
                self.image_pipe = DiffusionPipeline.from_pretrained(self.IMAGE_MODEL, torch_dtype=torch_dtype)
                if self.device == "cuda":
                    try:
                        self.image_pipe = self.image_pipe.to(self.device)
                    except Exception:
                        pass
            except Exception:
                self.image_pipe = None

    def _text_generate(self, prompt: str, max_new_tokens=80) -> str:
        # If text_model available use it; otherwise return a naive heuristic
        if self.text_model is not None and self.tokenizer is not None:
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                outputs = self.text_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True)
                return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            except Exception:
                pass

        if "LISTA:" in prompt:
            part = prompt.split("LISTA:", 1)[-1]
            lines = [l.strip() for l in part.splitlines() if l.strip()]
            if lines:
                return "\n".join(lines)

        candidates = [s.strip() for s in prompt.replace("\n", " ").split('.') if s.strip()]
        return "\n".join(candidates[: min(5, len(candidates))])

    def _to_base64_png(self, image: Image.Image) -> str:
        buf = BytesIO()
        image.save(buf, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    def generate(self, paragraph: str) -> Dict[str, Any]:
        """Genera pictogramas para el párrafo completo."""
        if self.text_model is None or self.image_pipe is None:
            self.load_model()

        prompt = f"Extrae hasta {self.MAX_CONCEPTS} ideas visuales del siguiente párrafo:\n{paragraph}\nLISTA:"
        raw = self._text_generate(prompt)
        raw_lines = [l.strip() for l in raw.split("\n") if l.strip()]
        labels_clean = []
        for l in raw_lines:
            lbl = l.strip("-0123456789. )\t\n")
            if not lbl:
                continue
            low = lbl.lower().strip().strip(":")
            if low in ("lista", "list", "lista:"):
                continue
            labels_clean.append(lbl)
        labels = labels_clean[: self.MAX_CONCEPTS]

        # 2. Generar pictogramas por concepto
        results = []
        for i, lbl in enumerate(labels):
            img_prompt = f"Flat simple pictogram of: {lbl}. Minimal strokes, clear silhouette, no text."
            img = None
            # Intentar generar con el modelo de difusión; si falla, usar placeholder simple
            if self.image_pipe is not None:
                try:
                    out = self.image_pipe(img_prompt, num_inference_steps=self.NUM_STEPS, guidance_scale=self.GUIDANCE_SCALE)
                    img = out.images[0].convert("RGBA")
                except Exception:
                    img = None

            if img is None:
                # Placeholder: imagen simple con fondo y el label
                w, h = self.IMAGE_SIZE
                img = Image.new("RGBA", (w, h), (255, 255, 255, 0))
                draw = ImageDraw.Draw(img)
                # Texto centrado
                try:
                    font = ImageFont.load_default()
                except Exception:
                    font = None
                text = (lbl or "?")[:40]
                # Compute text size robustly: prefer font.getsize, then draw.textbbox, then heuristic
                try:
                    if font is not None:
                        getsize = getattr(font, "getsize", None)
                        if callable(getsize):
                            coords = cast(tuple, getsize(text))
                            try:
                                text_w, text_h = int(coords[0]), int(coords[1])
                            except Exception:
                                text_w, text_h = (len(text) * 6, 12)
                        elif hasattr(draw, "textbbox"):
                            bbox = draw.textbbox((0, 0), text, font=font)
                            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                        else:
                            text_w, text_h = (len(text) * 6, 12)
                    elif hasattr(draw, "textbbox"):
                        bbox = draw.textbbox((0, 0), text, font=font)
                        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    else:
                        text_w, text_h = (len(text) * 6, 12)
                except Exception:
                    text_w, text_h = (len(text) * 6, 12)
                draw.rectangle([(0, 0), (w, h)], fill=(245, 245, 245, 255))
                draw.text(((w - text_w) / 2, (h - text_h) / 2), text, fill=(20, 20, 20), font=font)

            img_b64 = self._to_base64_png(img)
            results.append({"id": i + 1, "label": lbl, "image": img_b64})

        return {"paragraph": paragraph, "items": results}
