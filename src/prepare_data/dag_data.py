import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from abc import ABC, abstractmethod
from typing import Union
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rouge_score import rouge_scorer
from openai import OpenAI
import re

from datasets import Dataset
import pandas as pd
import json

from omegaconf import OmegaConf

from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).resolve()
MODEL_ROOT = Path(os.getenv("MODEL_ROOT")).resolve()
DATA_ROOT = Path(os.getenv("DATA_ROOT")).expanduser().resolve()
CONFIG_ROOT = Path(os.getenv("CONFIG_ROOT")).resolve()

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

        return {"article": article, "summaries": output_texts}
    
    def generate_batch(self, articles: Union[list[str], Dataset]) -> Dataset | list[dict]:
        if isinstance(articles, Dataset):
            def f(example):
                return {"summaries": self.generate(example['article'])["summaries"]}
            return articles.map(f, num_proc=1, desc="Generating summaries")
        else:
            outputs = []
            for article in tqdm(articles, desc="Generating summaries"):
                result = self.generate(article)
                if result is not None:
                    outputs.append(result)
            return outputs
    
class PreferenceScorer(ABC):
    @abstractmethod
    def compare(self, prompt: str, y1: str, y2: str, ref: str) -> int:
        pass
    
class ROUGEPreferenceScorer(PreferenceScorer):
    def __init__(self, rouge_type: str):
        self.rouge_type = rouge_type
        self.scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
                        
    def compare(self, prompt: str, y1: str, y2: str, ref: str) -> int:
        s1 = self.scorer.score(ref, y1)[self.rouge_type].fmeasure
        s2 = self.scorer.score(ref, y2)[self.rouge_type].fmeasure
                        
        return 0 if s1 > s2 else 1
    
class OpenAIPreferenceScorer(PreferenceScorer):
    def __init__(self, client: OpenAI, model_name: str, prompt_template: str):
        self.client = client
        self.model_name = model_name
        self.prompt_template = prompt_template
                      
    def compare(self, prompt: str, y1: str, y2: str, ref: str="") -> int | None:
        user_prompt = self.prompt_template.format(article=prompt, summary1=y1, summary2=y2)
        
        response = self.client.responses.create(
            model=self.model_name,
            input=user_prompt,
        )
        
        print(user_prompt)
        print(output_text)
        
        match = re.search(r'Preferred:\s*["\']?([12])["\']?', response.output_text)
        return match.group(1) == 2 if match else None
    
def get_preference_scorer(config: OmegaConf, openai_client=None) -> PreferenceScorer:
    if config.scorer.lower() == "rouge":
        return ROUGEPreferenceScorer(config.rouge.type)
    if config.scorer.lower() == "openai":
        return OpenAIPreferenceScorer(openai_client, config.openai.model, config.openai.prompt)
    
class PreferenceBuilder(ABC):
    def __init__(self, 
                 config: OmegaConf,
                 scorer: PreferenceScorer):
        self.config = config

    @abstractmethod
    def build(self, inputs):
        pass
    
def get_preference_builder(config: OmegaConf) -> PreferenceBuilder:
    pass
    
if __name__ == "__main__":
    config = OmegaConf.load(CONFIG_ROOT / "test.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = MODEL_ROOT / config.model_name
    dataset_path = DATA_ROOT / config.dataset_name

    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
    print("Model loaded successfully.")

    print(f"Loading dataset from {dataset_path}...")
    dataset = Dataset.from_pandas(pd.read_csv(dataset_path / "test.csv", nrows=config.nrow if "nrow" in config else None))
    print("Dataset loaded successfully.")
    
    generator = SummaryGenerator(tokenizer, model, config.generation)
    
    dataset = generator.generate_batch(dataset)
    
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    preference_scorer = get_preference_scorer(config.get_preference, openai_client=client)
    
    print(preference_scorer.compare(dataset[0]['article'], dataset[0]['summaries'][0], dataset[0]['summaries'][1]))
    