import os
import base64
import re
from io import BytesIO
from typing import List, Dict, Any, Optional
import requests
import unicodedata
import time
from urllib.parse import quote
from PIL import Image, ImageDraw, ImageFont
from backend.models.text_model import TextGenerationModel
from backend.config.settings import settings as app_settings

class PictogramGenerationModel:
    """
    Model for generating educational pictograms from text.
    Uses text models to extract keywords and image models to create pictograms.
    """
    def __init__(self):
        self.text_model = TextGenerationModel()
        # URL base del servicio de pictogramas (puede venir de settings o de la variable de entorno)
        self.base_url: str = getattr(app_settings, "pictograms_api_base_url", os.getenv("PICTOGRAMS_API_BASE_URL", "https://api.arasaac.org/api/"))
    
    def load_model(self):
        # Solo necesitamos cargar el modelo de texto (si corresponde). La parte de imágenes
        # se realiza mediante la API externa, por lo que no hay modelos locales para cargar aquí.
        if self.text_model.model is None:
            self.text_model.load_model()
    
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
            f"Extract ONLY the main noun (object, animal, person) "
            f"and the main verb from this sentence. Respond with 2 words separated by comma.\n"
            f"Sentence: {sentence}\n"
            f"Response:"
        )
        
        try:
            result = self.text_model.generate(prompt, max_new_tokens=10)

            # Extraer palabras robustamente de la respuesta del modelo. Muchos modelos
            # pueden responder con etiquetas o texto adicional; aquí buscamos secuencias
            # de letras (incluye acentos) y tomamos las dos primeras.
            tokens = re.findall(r"[^\W\d_]+", result, flags=re.UNICODE)

            # Normalizar: quitar tildes y espacios extra, bajar a minúsculas
            cleaned = []
            for t in tokens:
                t_norm = unicodedata.normalize('NFKD', t).encode('ascii', 'ignore').decode('ascii')
                t_norm = t_norm.strip().lower()
                if t_norm:
                    cleaned.append(t_norm)

            if cleaned:
                return cleaned[:2]

            # Fallback simple si el modelo no devolvió tokens útiles
            words = sentence.split()
            return words[:2] if len(words) >= 2 else words

        except Exception as e:
            print(f"Error extracting keywords: {e}")
            words = sentence.split()
            return words[:2] if len(words) >= 2 else words
    
    def _generate_pictogram(self, key_words: List[str]) -> Image.Image:
        # Construye texto de búsqueda a partir de las palabras clave y consulta la API externa
        if not key_words:
            return self._create_simple_fallback(key_words)
        # Crear la cadena de búsqueda con los tokens obtenidos (usar percent-encoding)
        search_text = quote(" ".join(key_words), safe='')

        try:
            results = self._search_pictograms("es", search_text)
            if results and len(results) > 0:
                # Preferir pictogramas sin violencia y esquemáticos si están disponibles.
                pict_id = None
                for r in results:
                    # Asegurarse de que r es dict
                    if not isinstance(r, dict):
                        continue
                    if r.get("violence"):
                        # ignorar resultados con marca de violencia
                        continue
                    if r.get("schematic"):
                        pict_id = r.get("_id")
                        break
                    if pict_id is None:
                        pict_id = r.get("_id")

                if pict_id is not None:
                    image = self._download_pictogram_image(pict_id)
                    if image:
                        return image

        except Exception as e:
            print(f"API pictogram generation error: {e}")

        # Si la API no devuelve nada o falla, usar fallback local
        return self._create_simple_fallback(key_words)

    def _search_pictograms(self, language: str, search_text: str) -> List[Dict[str, Any]]:
        """Consulta la API de pictogramas y devuelve la lista de resultados JSON.

        Args:
            language: código de idioma (por ejemplo 'es').
            search_text: texto de búsqueda URL-encoded.

        Returns:
            Lista de objetos (diccionarios) devueltos por la API o lista vacía en caso de error.
        """
        url = f"{self.base_url.rstrip('/')}/pictograms/{language}/search/{search_text}"
        attempts = 2
        for attempt in range(attempts):
            try:
                resp = requests.get(url, timeout=6)
                resp.raise_for_status()
                data = resp.json()

                # Aceptar varios formatos: lista directa o {results: [...] } o {data: [...]}
                if isinstance(data, list):
                    return data
                if isinstance(data, dict):
                    res = data.get('results')
                    if isinstance(res, list):
                        return res
                    res2 = data.get('data')
                    if isinstance(res2, list):
                        return res2

                print(f"_search_pictograms: unexpected response format: {type(data)}")
                return []

            except requests.RequestException as e:
                print(f"_search_pictograms request failed (attempt {attempt+1}): {e}")
                time.sleep(0.5)

        return []

    def _download_pictogram_image(self, id_pictogram: Any) -> Optional[Image.Image]:
        """Descarga la imagen PNG del pictograma por su id y la devuelve como PIL.Image.

        Args:
            id_pictogram: identificador del pictograma (tal como lo devuelve la API).

        Returns:
            PIL.Image si se descarga correctamente, o None si falla.
        """
        url = f"{self.base_url.rstrip('/')}/pictograms/{id_pictogram}"
        attempts = 2
        for attempt in range(attempts):
            try:
                resp = requests.get(url, timeout=8)
                resp.raise_for_status()

                ctype = resp.headers.get('content-type', '')
                if not ctype.startswith('image/'):
                    print(f"_download_pictogram_image: unexpected content-type {ctype}")
                    return None

                try:
                    img = Image.open(BytesIO(resp.content))
                    return img.convert("RGB")
                except OSError as e:
                    print(f"_download_pictogram_image: failed to parse image: {e}")
                    return None

            except requests.RequestException as e:
                print(f"_download_pictogram_image request failed (attempt {attempt+1}): {e}")
                time.sleep(0.5)

        return None
    
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
        if self.text_model.model is None:
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
                
                # Ajustar la estructura para cumplir con el schema PictogramItem:
                # id (int), sentence (str), concept (str), image (str)
                concept = " ".join(key_words) if key_words else ""
                # Añadir label/alt para compatibilidad con el frontend
                label = concept if concept else sentence
                pictograms.append({
                    "id": idx + 1,
                    "sentence": sentence,
                    "concept": concept,
                    "label": label,
                    "alt": label,
                    "image": image_b64
                })
                
                # no cache management necesario cuando se usan imágenes desde la API
                    
            except Exception as e:
                print(f"Sentence error {idx + 1}: {e}")
                # ignored: la API puede fallar por petición HTTP
                
                key_words = self._extract_key_words(sentence)
                fallback = self._create_simple_fallback(key_words)
                concept = " ".join(key_words) if key_words else ""
                label = concept if concept else sentence
                pictograms.append({
                    "id": idx + 1,
                    "sentence": sentence,
                    "concept": concept,
                    "label": label,
                    "alt": label,
                    "image": self._image_to_base64(fallback)
                })
        
        return {
            "paragraph": text,
            "items": pictograms
        }