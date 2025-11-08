import os
import base64
import re
from io import BytesIO
from typing import List, Dict, Any
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from transformers import set_seed
from backend.models.text_model import TextGenerationModel


class PictogramGenerationModel:
    def __init__(self):
        self.text_model = TextGenerationModel()
        self.image_pipe = None
        self.device = "cpu"
        
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_gb = gpu_memory / (1024**3)
                
                if gpu_memory_gb > 6:
                    self.device = "cuda"
                else:
                    print(f"GPU memory ({gpu_memory_gb:.2f}GB) insufficient, using CPU")
            except Exception as e:
                print(f"GPU check failed: {e}, using CPU")
        
        self.IMAGE_MODEL = "stabilityai/sdxl-turbo"
        self.IMAGE_SIZE = (512, 512)
        self.NUM_STEPS = 2
        
        set_seed(2024)
    
    def load_model(self):
        if self.text_model.model is None:
            self.text_model.load_model()
        
        if self.image_pipe is None:
            try:
                if self.device == "cuda":
                    try:
                        torch.cuda.empty_cache()
                        self.image_pipe = StableDiffusionXLPipeline.from_pretrained(
                            self.IMAGE_MODEL,
                            torch_dtype=torch.float16,
                            variant="fp16"
                        )
                        self.image_pipe.to(self.device)
                        print("SDXL-Turbo loaded on CUDA")
                    except (torch.cuda.OutOfMemoryError, RuntimeError):
                        print("CUDA failed, using CPU")
                        self.device = "cpu"
                        torch.cuda.empty_cache()
                        self.image_pipe = StableDiffusionXLPipeline.from_pretrained(
                            self.IMAGE_MODEL,
                            torch_dtype=torch.float32
                        )
                else:
                    self.image_pipe = StableDiffusionXLPipeline.from_pretrained(
                        self.IMAGE_MODEL,
                        torch_dtype=torch.float32
                    )
                    
            except Exception as e:
                print(f"Failed to load model: {e}")
                self.image_pipe = None
    
    def _split_into_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 8]
    
    def _extract_key_words(self, sentence: str) -> List[str]:
        if self.text_model.model is None:
            self.load_model()
        
        if self.text_model.model is None:
            words = sentence.split()
            return words[:2] if len(words) >= 2 else words
        
        prompt = (
            f"Extrae SOLO el sustantivo principal (objeto, animal, persona) "
            f"y el verbo principal de esta frase. Responde con 2 palabras separadas por coma.\n"
            f"Frase: {sentence}\n"
            f"Respuesta:"
        )
        
        try:
            result = self.text_model.generate(prompt, max_new_tokens=10)
            words = [w.strip() for w in result.split(",") if w.strip()]
            
            words = [re.sub(r'[^\w\s]', '', w).strip() for w in words if w.strip()]
            
            return words[:2] if len(words) >= 2 else words if words else ["cosa"]
            
        except Exception as e:
            print(f"Error: {e}")
            words = sentence.split()
            return words[:2] if len(words) >= 2 else words
    
    def _generate_pictogram(self, key_words: List[str]) -> Image.Image:
        if not self.image_pipe:
            self.load_model()
        
        if not self.image_pipe:
            return self._create_simple_fallback(key_words)
        
        word = key_words[0] if key_words else "cosa"
        
        if len(key_words) > 1:
            prompt = (
                f"Louie style children drawing of {word} {key_words[1]}, "
                f"very simple, minimalist cartoon, basic shapes, "
                f"bright solid colors, thick black outlines, "
                f"educational pictogram for toddlers, flat design, "
                f"white background, no text, no letters"
            )
        else:
            prompt = (
                f"Louie style children drawing of {word}, "
                f"very simple, minimalist cartoon, basic shapes, "
                f"bright solid colors, thick black outlines, "
                f"educational pictogram for toddlers, flat design, "
                f"white background, no text, no letters"
            )
        
        negative_prompt = (
            "realistic, photo, complex, detailed, shadows, "
            "text, letters, words, blurry, dark"
        )
        
        try:
            with torch.inference_mode():
                image = self.image_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=self.NUM_STEPS,
                    guidance_scale=0.0,
                    height=self.IMAGE_SIZE[1],
                    width=self.IMAGE_SIZE[0]
                ).images[0]
            
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            return image
            
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return self._create_simple_fallback(key_words)
        except Exception as e:
            print(f"Generation error: {e}")
            return self._create_simple_fallback(key_words)
    
    def _create_simple_fallback(self, key_words: List[str]) -> Image.Image:
        img = Image.new("RGB", (256, 256), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        draw.ellipse([50, 50, 206, 206], fill=(100, 180, 255), outline=(40, 100, 200), width=4)
        
        if key_words:
            text = key_words[0][:12]
            bbox = draw.textbbox((0, 0), text, font=font_small)
            text_width = bbox[2] - bbox[0]
            x = (256 - text_width) // 2
            draw.text((x, 220), text, fill=(20, 20, 20), font=font_small)
        
        return img
    
    def _image_to_base64(self, image: Image.Image) -> str:
        image_small = image.resize((192, 192), Image.Resampling.LANCZOS)
        
        buffered = BytesIO()
        image_small.save(buffered, format="PNG", optimize=True)
        return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"
    
    def generate(self, text: str) -> Dict[str, Any]:
        if self.text_model.model is None or self.image_pipe is None:
            self.load_model()
        
        sentences = self._split_into_sentences(text)
        
        if len(sentences) > 8:
            step = max(1, len(sentences) // 8)
            sentences = [sentences[i] for i in range(0, len(sentences), step)][:8]
        
        pictograms = []
        
        for idx, sentence in enumerate(sentences):
            try:
                key_words = self._extract_key_words(sentence)
                
                image = self._generate_pictogram(key_words)
                image_b64 = self._image_to_base64(image)
                
                pictograms.append({
                    "id": idx + 1,
                    "sentence": sentence,
                    "keywords": key_words,
                    "image": image_b64
                })
                
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Sentence error {idx + 1}: {e}")
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                
                key_words = self._extract_key_words(sentence)
                fallback = self._create_simple_fallback(key_words)
                pictograms.append({
                    "id": idx + 1,
                    "sentence": sentence,
                    "keywords": key_words,
                    "image": self._image_to_base64(fallback)
                })
        
        return {
            "paragraph": text,
            "items": pictograms
        }