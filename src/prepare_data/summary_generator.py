import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

from omegaconf import OmegaConf

class SummaryGenerator:
    def __init__(self,
                 tokenizer: AutoTokenizer,
                 model: AutoModelForCausalLM,
                 config: OmegaConf):
        self.tokenizer = tokenizer
        self.model = model
        self.config = config
    
    def generate(self, article: str) -> dict | None:
        prompt_text = self.config.prompt.format(article=article)
        
        message = [
            { "role": "user", "content": prompt_text }
        ]
        prompt = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        if inputs.shape[-1] > self.model.config.max_position_embeddings:
            return None
        
        attention_mask = torch.ones(inputs.shape, device=self.model.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                num_return_sequences=self.config.num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id
            )

        output_texts = [
            self.tokenizer.decode(output[inputs.shape[-1]:], skip_special_tokens=True) for output in outputs
        ]
        output_texts = [text.strip() for text in output_texts if text.strip()]

        return {"article": article, "prompt": prompt_text, "summaries": output_texts}
    
    def generate_batch(self, articles: Dataset) -> Dataset:
        def f(example):
            generation = self.generate(example['article'])
            if generation is None:
                return {"summaries": [], "prompt": ""}
            return {k: v for k, v in generation.items() if k != 'article'}
        return articles.map(f, num_proc=1, desc="Generating summaries")
    