import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

from abc import ABC, abstractmethod

from omegaconf import OmegaConf
import string

class Generator(ABC):
    @abstractmethod
    def generate(self, example: dict) -> dict | None:
        pass

    @abstractmethod
    def generate_batch(self, dataset: Dataset) -> Dataset:
        pass

class ModelGenerator(Generator):
    def __init__(self,
                 tokenizer: AutoTokenizer,
                 model: AutoModelForCausalLM,
                 config: OmegaConf):
        self.tokenizer = tokenizer
        self.model = model
        self.config = config
        self.fields = [fname for _, fname, _, _ in string.Formatter().parse(self.config.prompt) if fname]

    @torch.inference_mode()
    def generate(self, example: dict) -> dict[str, str | list]:
        format_args: dict[str, str] = {}
        for field in self.fields:
            if field not in example.keys():
                raise KeyError(f"There is no {field} key in data")
            format_args[field] = str(example[field])
        prompt_text = self.config.prompt.format(**format_args)

        if self.tokenizer.chat_template is not None:
            message = [
                { "role": "user", "content": prompt_text }
            ]
            prompt = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        else:
            prompt = prompt_text

        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        if inputs.shape[-1] + self.config.max_new_tokens > self.model.config.max_position_embeddings:
            result: dict[str, str | list] = example.copy()
            result['prompt'] = prompt_text
            result['generated'] = ""
            return result

        attention_mask = torch.ones(inputs.shape, device=self.model.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **{str(k): v for k, v in self.config.items() if k != "prompt"}
            )

        result: dict[str, str | list] = example.copy()

        if self.tokenizer.chat_template is not None:
            output_texts = [
                self.tokenizer.decode(output[inputs.shape[-1]:], skip_special_tokens=True) for output in outputs
            ]
        else:
            output_texts = [
                self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs
            ]
        output_texts = [text.strip() for text in output_texts if text.strip()]

        result['prompt'] = prompt_text
        result['generated'] = output_texts
        return result

    @torch.inference_mode()
    def generate_batch(self, dataset: Dataset) -> Dataset:
        def f(example: dict):
            generation = self.generate(example)
            return {k: v for k, v in generation.items() if k not in example.keys()}
        return dataset.map(f, num_proc=1, desc="Generating summaries")

class ReferenceSummaryGenerator(Generator):
    def __init__(self,
                 config: OmegaConf):
        self.config = config
        if self.config.num_return_sequences != 1:
            raise ValueError("ReferenceSummaryGenerator only supports num_return_sequences=1")

    def generate(self, example: dict) -> dict | None:
        result: dict[str, str | list] = example.copy()
        result['prompt'] = ""
        result['generated'] = [result[self.config.ref_key]]
    
    def generate_batch(self, dataset: Dataset) -> Dataset:
        def f(example):
            return {"generated": [example[self.config.ref_key]], "prompt": ""}
        return dataset.map(f, num_proc=1, desc="Generating summaries")