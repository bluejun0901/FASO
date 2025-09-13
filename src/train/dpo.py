import os
from pathlib import Path
from dotenv import load_dotenv
import argparse
load_dotenv()

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).resolve() # type: ignore
MODEL_ROOT = Path(os.getenv("MODEL_ROOT")).resolve() # type: ignore
DATA_ROOT = Path(os.getenv("DATA_ROOT")).resolve() # type: ignore
CONFIG_ROOT = Path(os.getenv("CONFIG_ROOT")).resolve() # type: ignore
LOG_ROOT = Path(os.getenv("LOG_ROOT")).resolve() # type: ignore

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

from datasets import Dataset
import json
from src.utils.utility import *
from src.train.trainers import *

from omegaconf import OmegaConf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_path", 
        type=str, 
        help="reletive path to configuration file"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="reletive path to train dataset"
    )
    args = parser.parse_args()

    config = OmegaConf.load(CONFIG_ROOT / args.config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = MODEL_ROOT / config.model_name
    dataset_path = DATA_ROOT / config.dataset_output_dir / args.dataset_path

    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(str(model_path), trust_remote_code=True).to(device)
    model.warnings_issued = {}
    print("Model loaded successfully.")

    print(f"Loading trian dataset from {str(dataset_path)}...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    dataset = Dataset.from_dict(dataset)
    print("Loaded successfully")

    print("Creating trainer...")
    trainer = get_m_trainer(config.trainer, tokenizer, model)
    print("Trainer created")

    output_dir = MODEL_ROOT / config.model_output_dir / get_filename(config.builder.type, config.scorer.type, config.trainer.type, suffix="")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = LOG_ROOT / config.log_dir / get_filename(config.builder.type, config.scorer.type, config.trainer.type, suffix="")
    log_dir.mkdir(parents=True, exist_ok=True)

    print("Preprocessing data...")
    trainer.preprocess(dataset)
    print("Data preprocessed")

    print("Training...")
    trainer.train(str(output_dir), str(log_dir))
    print("Finished")
