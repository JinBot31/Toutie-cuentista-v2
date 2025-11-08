import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import os

class TextGenerationModel:
    """Model class for generating text using transformer-based language models."""
    def __init__(self, model_checkpoint: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.model_checkpoint = model_checkpoint
        self.tokenizer = None
        self.model = None
        self.device = "cpu"  # Default to CPU
        
        # Check if CUDA is available and has enough memory
        if torch.cuda.is_available():
            try:
                # Try to get GPU memory info
                torch.cuda.empty_cache()  # Clear cache first
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_gb = gpu_memory / (1024**3)
                
                # Only use CUDA if we have more than 4GB available
                if gpu_memory_gb > 4:
                    self.device = "cuda"
                else:
                    print(f"GPU memory ({gpu_memory_gb:.2f}GB) insufficient, using CPU")
            except Exception as e:
                print(f"GPU check failed: {e}, using CPU")
        
        set_seed(2024)
    
    def load_model(self):
        """Load the text generation model and tokenizer."""
        if self.model is None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_checkpoint,
                    trust_remote_code=True
                )
                
                # Load model with appropriate settings based on device
                if self.device == "cuda":
                    try:
                        # Try CUDA with float16
                        torch.cuda.empty_cache()
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_checkpoint,
                            torch_dtype=torch.float16,
                            trust_remote_code=True,
                            low_cpu_mem_usage=True
                        )
                        self.model = self.model.to("cuda")
                        print("Model loaded on CUDA with float16")
                    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                        print(f"CUDA loading failed: {e}")
                        print("Falling back to CPU...")
                        self.device = "cpu"
                        torch.cuda.empty_cache()
                        # Load on CPU instead
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_checkpoint,
                            torch_dtype=torch.float16,
                            trust_remote_code=True,
                            low_cpu_mem_usage=True
                        )
                        print("Model loaded on CPU with float16")
                else:
                    # Load directly on CPU
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_checkpoint,
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
                    print("Model loaded on CPU with float16")
                    
            except Exception as e:
                print(f"Failed to load model: {e}")
                # Keep model as None, will use fallback
                self.model = None
                self.tokenizer = None
    
    def _simple_fallback(self, prompt: str, max_new_tokens: int = 200) -> str:
        """Simple fallback when model cannot be loaded.

        Args:
            prompt (str): The original prompt for story generation.
            max_new_tokens (int): Maximum number of tokens (unused in fallback).

        Returns:
            str: A simple fallback story.
        """
        lines = [l.strip() for l in (prompt or "").splitlines() if l.strip()]
        lead = lines[0] if lines else "Había una vez"
        story = f"{lead}. Una pequeña aventura comenzó y el protagonista aprendió sobre la amistad y la curiosidad. Fin."
        return story
    
    def generate(self, prompt: str, max_new_tokens: int = 400) -> str:
        """Generate text based on the given prompt."""
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        # If model still couldn't be loaded, use fallback
        if self.model is None or self.tokenizer is None:
            print("Model not available, using fallback generation")
            return self._simple_fallback(prompt, max_new_tokens)
        
        try:
            messages = [
                {"role": "system", "content": "Eres un narrador de cuentos infantiles. Crea historias sencillas, positivas y completas en español."},
                {"role": "user", "content": prompt}
            ]
            
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Move inputs to the correct device
            inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            
            try:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            except torch.cuda.OutOfMemoryError:
                print("CUDA OOM during generation, clearing cache and retrying on CPU...")
                torch.cuda.empty_cache()
                self.device = "cpu"
                self.model = self.model.to("cpu")
                inputs = inputs.to("cpu")
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
            
            last_dot = response.rfind(".")
            if last_dot != -1:
                response = response[:last_dot+1]
            
            return response.strip()
            
        except Exception as e:
            print(f"Error during generation: {e}")
            return self._simple_fallback(prompt, max_new_tokens)