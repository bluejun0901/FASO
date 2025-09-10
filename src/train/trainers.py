from abc import ABC, abstractmethod

import torch
from trl import AutoModelForCausalLMWithValueHead, DPOTrainer, DPOConfig

from datasets import Dataset
from omegaconf import OmegaConf
from typing import Any

from utils.utility import *

class mTrainer(ABC):
    @abstractmethod
    def preprocess(self, dataset: Dataset | list[dict[str, str]]) -> Dataset:
        pass

    @abstractmethod
    def train(self, output_dir: str, logging_dir: str) -> None:
        pass

class mDPOTrainer(mTrainer):
    def __init__(self,
                 tokenizer: Any,
                 model: AutoModelForCausalLMWithValueHead,
                 config: OmegaConf):
        self.tokenizer = tokenizer
        self.model = model
        self.config = config
        self.dataset = None

    def preprocess(self, dataset: Dataset | list[dict[str, str]]) -> Dataset:
        if dataset is list:
            self.dataset = Dataset.from_list(dataset)
        else:
            assert isinstance(dataset, Dataset)
            self.dataset = dataset
        return dataset
    
    def train(self, output_dir: str, logging_dir: str) -> None:
        if self.dataset is None:
            raise RuntimeError("You must call preprocess() before train().")

        train_config = OmegaConf.to_container(self.config, resolve=True)
        if not isinstance(train_config, dict):
            raise TypeError("train_config must be a dict")
        train_config = {str(k): v for k, v in train_config.items()}

        training_args = DPOConfig(
            output_dir=output_dir,
            logging_dir=logging_dir,
            **train_config
        )
        trainer = DPOTrainer(
            model=self.model,
            args=training_args, 
            processing_class=self.tokenizer, 
            train_dataset=self.dataset,
        )
        
        trainer.train()

def get_m_trainer(config: OmegaConf,
                  tokenizer: Any,
                  model: AutoModelForCausalLMWithValueHead) -> mTrainer:
    name = config.type.lower()
    if name == "dpo":
        return mDPOTrainer(tokenizer, model, config.dpo)
    raise Exception(f"Unknown preference builder: {config.type}")
