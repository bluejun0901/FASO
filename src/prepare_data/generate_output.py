import os
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).resolve() # type: ignore
MODEL_ROOT = Path(os.getenv("MODEL_ROOT")).resolve() # type: ignore
DATA_ROOT = Path(os.getenv("DATA_ROOT")).resolve() # type: ignore
CONFIG_ROOT = Path(os.getenv("CONFIG_ROOT")).resolve() # type: ignore
SRC_ROOT = Path(os.getenv("SRC_ROOT")).resolve() # type: ignore

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
sys.path.append(str(SRC_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import Dataset
import pandas as pd
import json
from utils.utility import *

from omegaconf import OmegaConf

from summary_generator import *
    
if __name__ == "__main__":
    config = OmegaConf.load(CONFIG_ROOT / input("input configuration path: "))
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    gen_filename = get_filename("generated", config.builder.type, config.scorer.type, suffix=".json")
    gen_output_path = DATA_ROOT / config.output_dir / gen_filename
    print(f"Saving generated summeries to {str(gen_output_path)}...")
    gen_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(gen_output_path), 'w', encoding="utf-8") as f:
        json.dump(dataset.to_dict(), f, ensure_ascii=False, indent=2)
    print("Saved successfully")