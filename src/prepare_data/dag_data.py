import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import openai

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

def get_summary_from_model(model: AutoModelForCausalLM,
                           tokenizer: AutoTokenizer,
                           article: str,
                           config: OmegaConf) -> list[str]:
    prompt_text = config.prompt.format(article=article)
    message = [
        { "role": "user", "content": prompt_text }
    ]
    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    attention_mask = torch.ones(inputs.shape, device=model.device)

    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_new_tokens=config.max_new_tokens,
        do_sample=True,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        num_return_sequences=config.num_return_sequences,
        pad_token_id=tokenizer.eos_token_id
    )
    
    output_texts = [
        tokenizer.decode(output[inputs.shape[-1]:], skip_special_tokens=True) for output in outputs
    ]
    output_texts = [text.strip() for text in output_texts if text.strip()]
    
    return output_texts

if __name__ == "__main__":
    config = OmegaConf.load(CONFIG_ROOT / "test.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = MODEL_ROOT / config.model_name
    dataset_path = DATA_ROOT / config.dataset_name

    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
    print("Model loaded successfully.")

    print(f"Loading dataset from {dataset_path}...")
    dataset = Dataset.from_pandas(pd.read_csv(dataset_path / "test.csv"))
    print("Dataset loaded successfully.")

    