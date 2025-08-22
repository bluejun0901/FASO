import os
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).resolve()
MODEL_ROOT = Path(os.getenv("MODEL_ROOT")).resolve()
DATA_ROOT = Path(os.getenv("DATA_ROOT")).resolve()
CONFIG_ROOT = Path(os.getenv("CONFIG_ROOT")).resolve()
SRC_ROOT = Path(os.getenv("SRC_ROOT")).resolve()

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
sys.path.append(str(SRC_ROOT))

from abc import ABC, abstractmethod
from typing import Union
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rouge_score import rouge_scorer
from openai import OpenAI
from openai.types import Batch
import re
import random
import time

from datasets import Dataset
import pandas as pd
import json
from utils.utility import *

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
    
    def generate_batch(self, articles: Union[list[str], Dataset]) -> Dataset | list[dict]:
        if isinstance(articles, Dataset):
            def f(example):
                generation = self.generate(example['article'])
                if generation is None:
                    return {"summaries": [], "prompt": ""}
                return {k: v for k, v in generation.items() if k != 'article'}
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
    def require_ref(self) -> bool:
        pass
    
    @abstractmethod
    def compare(self, prompt: str, y1: str, y2: str, ref: str) -> int | None:
        pass
    
    @abstractmethod
    def compare_batch(self, pairs: Union[list[dict], Dataset]) -> list[int]:
        pass
    
class ROUGEPreferenceScorer(PreferenceScorer):
    def __init__(self, config: OmegaConf):
        self.require_ref_flag = True
        self.rouge_type = config.type
        self.scorer = rouge_scorer.RougeScorer([self.rouge_type], use_stemmer=True)
        
    def require_ref(self):
        return self.require_ref_flag
                        
    def compare(self, prompt: str, y1: str, y2: str, ref: str) -> int | None:
        s1 = self.scorer.score(ref, y1)[self.rouge_type].fmeasure
        s2 = self.scorer.score(ref, y2)[self.rouge_type].fmeasure
                        
        return 0 if s1 > s2 else 1
    
    def compare_batch(self, pairs: Union[list[dict], Dataset]) -> list[int]:
        compared = []
        for pair in pairs:
            compared.append(self.compare(pair['prompt'], pair['y1'], pair['y2'], pair['ref']))
        return compared
    
class OpenAIPreferenceScorer(PreferenceScorer):
    def __init__(self, client: OpenAI, config: OmegaConf):
        self.require_ref_flag = False
        self.client = client
        self.model_name = config.model
        self.prompt_template = config.prompt
        self.pattern = config.preference_pattern
        
    def require_ref(self):
        return self.require_ref_flag
                      
    def compare(self, prompt: str, y1: str, y2: str, ref: str="") -> int | None:
        user_prompt = self.prompt_template.format(article=prompt, summary1=y1, summary2=y2)
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_prompt}
            ]
        )
        output_text = response.choices[0].message.content
        
        match = re.search(self.pattern, output_text)
        return int(match.group(1)) == 2 if match else None
    
    def compare_batch(self, pairs: Union[list[dict], Dataset]) -> list[int]:
        compared = []
        for pair in pairs:
            compared.append(self.compare(pair['prompt'], pair['y1'], pair['y2']))
        return compared
    
class BatchPreferenceScorer(PreferenceScorer):
    @abstractmethod
    def require_ref(self) -> bool:
        pass

    def compare(self, prompt: str, y1: str, y2: str, ref: str) -> int | None:
        raise Exception("BatchPreferenceScorer doesn't support 'compare'. Try calling compare_batch_* instead.")

    def compare_batch(self, pairs: Union[list[dict], Dataset]) -> list[int]:
        raise Exception("BatchPreferenceScorer doesn't support 'compare_batch'. Try calling compare_batch_* instead.")
    
    @abstractmethod
    def compare_batch_0(self, pairs: Union[list[dict], Dataset]) -> list[list[dict]]:
        pass

    @abstractmethod
    def compare_batch_1(self, path: list[str]) -> dict:
        pass

class OpenAIBatchPreferenceScorer(BatchPreferenceScorer):
    def __init__(self, client: OpenAI, config: OmegaConf):
        self.require_ref_flag = False
        self.client = client
        self.model_name = config.model
        self.prompt_template = config.prompt
        self.pattern = config.preference_pattern

        c = getattr(config, "batch", None)
        self.max_concurrent = getattr(c, "max_concurrent", 3) if c is not None else 3
        self.max_retries = getattr(c, "max_retries", 5) if c is not None else 5
        self.initial_backoff = getattr(c, "initial_backoff", 1.0) if c is not None else 1.0
        self.poll_interval = getattr(c, "poll_interval", 15.0) if c is not None else 15.0

        self.paths = []
        self.batch_files = []
        self.batchs = []
    
    def require_ref(self):
        return self.require_ref_flag
    
    def compare_batch_0(self, pairs: Union[list[dict], Dataset], request_size=30000) -> list[list[dict]]:
        requests = []
        for i, pair in enumerate(pairs):
            if i % request_size == 0:
                requests.append([])

            prompt = pair['prompt']
            summary1 = pair['y1']
            summary2 = pair['y2']

            user_prompt = self.prompt_template.format(article=prompt, summary1=summary1, summary2=summary2)
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_prompt}
            ]
            body = {
                "model": self.model_name,
                "messages": messages
            }
            requests[i // request_size].append({
                "custom_id": f"{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body
            })

        return requests
    
    def _retry(self, fn, *args, **kwargs):
        delay = self.initial_backoff
        for i in range(self.max_retries):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                if i == self.max_retries - 1:
                    raise e
                time.sleep(delay + random.random() * 0.25 * delay)
                delay *= 2

    def _submit_one(self, path: str) -> Batch:
        with open(path, "rb") as f:
            batch_file = self._retry(self.client.files.create, file=f, purpose="batch")
        batch = self._retry(
            self.client.batches.create,
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        self.batch_files.append(batch_file)
        self.batchs.append(batch)
        print(f"file {path} (file id: {batch_file.id}) submitted (batch id: {batch.id})")
        return batch

    def compare_batch_1(self, paths: list[str], max_concurrent: int | None = None) -> dict:
        pending = list(paths)
        in_flight: dict[str, Batch] = {}
        finished: list[Batch] = []

        count = {
            "completed": 0,
            "failed": 0,
            "expired": 0,
            "cancelled": 0,
            "canceled": 0,
        }

        if max_concurrent is None:
            max_concurrent = self.max_concurrent

        # Prime the window
        while pending and len(in_flight) < max_concurrent:
            b = self._submit_one(pending.pop(0))
            in_flight[b.id] = b

        # Monitor and refill
        while pending or in_flight:
            time.sleep(self.poll_interval)

            to_delete = []
            for batch_id in list(in_flight.keys()):
                b = self._retry(self.client.batches.retrieve, batch_id)
                if b.status in ("completed", "failed", "expired", "cancelled", "canceled"):
                    to_delete.append(batch_id)
                    finished.append(b)
                    if b.status in count:
                        count[b.status] += 1
                    else:
                        count[b.status] = 1
                    print(f"{batch_id} {b.status}")

            for bid in to_delete:
                in_flight.pop(bid, None)

            while pending and len(in_flight) < max_concurrent:
                b = self._submit_one(pending.pop(0))
                in_flight[b.id] = b

        # Optionally, update self.batchs snapshots with terminal objects
        by_id = {b.id: b for b in self.batchs}
        for b in finished:
            by_id[b.id] = b
        self.batchs = list(by_id.values())

        return count
    
def get_preference_scorer(config: OmegaConf, openai_client=None) -> PreferenceScorer:
    if config.scorer.lower() == "rouge":
        return ROUGEPreferenceScorer(config.rouge)
    if config.scorer.lower() == "openai":
        if config.openai.type != "batch":
            return OpenAIPreferenceScorer(openai_client, config.openai)
        else:
            return OpenAIBatchPreferenceScorer(openai_client, config.openai)
    
class PreferenceBuilder(ABC):
    @abstractmethod
    def generate_comparisons(self, dataset: Dataset) -> Dataset:
        pass
    
    @abstractmethod
    def build_with_comparisons(self, comparisons: list[int]) -> Dataset:
        pass
    
class PairwisePreferenceBuilder(PreferenceBuilder):
    def __init__(self, scorer):
        self.pairs = []
        self.scorer = scorer
        
    def generate_comparisons(self, dataset: Dataset) -> list[dict]:
        for example in dataset:
            prompt = example['prompt']
            ref = example['reference'] if self.scorer.require_ref() else ""
            summaries = example['summaries']
            
            for i, y1 in enumerate(summaries):
                for j, y2 in enumerate(summaries):
                    if i < j:
                        self.pairs.append({
                            'prompt': prompt,
                            'y1': y1,
                            'y2': y2,
                            'ref': ref
                        })
        
        return self.pairs
    
    def build_with_comparisons(self, comparisons: list[int]) -> Dataset:
        result = []
        
        for pref, pair in zip(comparisons, self.pairs):
            if pref is None:
                continue
            
            chosen, rejected = (pair['y1'], pair['y2']) if pref == 0 else (pair['y2'], pair['y1'])
            result.append({
                'prompt': pair['prompt'],
                'chosen': chosen,
                'rejected': rejected,
            })
            
        return Dataset.from_list(result)

def get_preference_builder(config: OmegaConf, scorer: PreferenceScorer) -> PreferenceBuilder:
    if config.builder.lower() == "pairwise":
        return PairwisePreferenceBuilder(scorer)
    
def is_preference_two_step(config: OmegaConf) -> bool:
    try:
        return config.scorer.lower() == "openai" and getattr(config.openai, "type", "") == "batch"
    except Exception:
        return False
    
if __name__ == "__main__":
    config = OmegaConf.load(CONFIG_ROOT / "test.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )

    model_path = MODEL_ROOT / config.model_name
    dataset_path = DATA_ROOT / config.dataset_name

    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
    print("Model loaded successfully.")

    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path / "test.csv", nrows=config.nrow if "nrow" in config else None)
    
    df = df.rename(columns=dict(config.col_renames))
    
    dataset = Dataset.from_pandas(df)
    print("Dataset loaded successfully.")
    
    generator = SummaryGenerator(tokenizer, model, config.generation)
    dataset = generator.generate_batch(dataset)
    print("Summaries generated successfully.")
    
    if is_preference_two_step(config.get_preference) == False:
        print("Labeling data...")
        preference_scorer = get_preference_scorer(config.get_preference, openai_client=client)
        preference_builder = get_preference_builder(config.get_preference, preference_scorer)
        
        comparisons = preference_builder.generate_comparisons(dataset)
        compared = preference_scorer.compare_batch(comparisons)
        result = preference_builder.build_with_comparisons(compared)
        print("Labeled successfully.")
        
        filename = get_filename("output", config.get_preference.builder, config.get_preference.scorer, suffix=".json")
        output_path = DATA_ROOT / config.output_dir / filename
        
        print(f"Saving result to {str(output_path)}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(output_path), "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        print("Saved successfully.")
    else:
        print("Creating jsonl file for request...")
        preference_scorer = get_preference_scorer(config.get_preference, openai_client=client)
        preference_builder = get_preference_builder(config.get_preference, preference_scorer)

        comparisons = preference_builder.generate_comparisons(dataset)
        requests = preference_scorer.compare_batch_0(comparisons)
        print("jsonl file created")

        base_filename = get_filename("request", config.get_preference.builder, config.get_preference.scorer, "*", suffix=".jsonl")
        print(f"Saving jsonl request to {str(DATA_ROOT / config.output_dir / base_filename)} ({len(requests)} files)...")

        paths = []
        for i, request in enumerate(requests):
            request_filename = get_filename(
                "request",
                config.get_preference.builder,
                config.get_preference.scorer,
                str(i).zfill(len(str(len(requests) - 1))),
                suffix=".jsonl"
            )
            request_output_path = DATA_ROOT / config.output_dir / request_filename

            paths.append(str(request_output_path))
            request_output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(str(request_output_path), "w", encoding="utf-8") as f:
                for item in request:
                    line = json.dumps(item, ensure_ascii=False)
                    f.write(line + "\n")
        print("Saved successfully")

        print("Processing Batch...")
        batch_result = preference_scorer.compare_batch_1(paths)
        print("Batches processed; summary:")
        for key, val in batch_result.items():
            print(f"  {key}: {val}")
