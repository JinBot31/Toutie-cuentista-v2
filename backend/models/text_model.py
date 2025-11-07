import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

class TextGenerationModel:
    def __init__(self, model_checkpoint: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.model_checkpoint = model_checkpoint
        self.tokenizer = None
        self.model = None
        set_seed(2024)
    
    def load_model(self):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_checkpoint,
                torch_dtype=torch.float16,
                device_map="cuda"
            )
    
    def generate(self, prompt: str, max_new_tokens: int = 400) -> str:
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        assert self.tokenizer is not None and self.model is not None
        
        messages = [
            {"role": "system", "content": "Eres un narrador de cuentos infantiles. Crea historias sencillas, positivas y completas en español."},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer([text], return_tensors="pt").to("cuda")
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