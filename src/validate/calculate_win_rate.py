import os
from pathlib import Path
from dotenv import load_dotenv
import argparse

load_dotenv()

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).resolve() # type: ignore
MODEL_ROOT = Path(os.getenv("MODEL_ROOT")).resolve() # type: ignore
DATA_ROOT = Path(os.getenv("DATA_ROOT")).resolve() # type: ignore
CONFIG_ROOT = Path(os.getenv("CONFIG_ROOT")).resolve() # type: ignore
SRC_ROOT = Path(os.getenv("SRC_ROOT")).resolve() # type: ignore

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import Dataset
import pandas as pd
from openai import OpenAI
from src.utils.utility import *

from omegaconf import OmegaConf
import json

from src.prepare_data.summary_generator import *
from src.prepare_data.preference_scorers import *
from src.validate.win_rate_calculator import WinRateCalculator

def get_summary_generator(model_path: str, generation_config: OmegaConf) -> SummaryGenerator:
    model_path = model_path.strip()
    if model_path.lower().endswith("reference"):
        return ReferenceSummaryGenerator(generation_config)
    else:
        print(f"Loading model from {MODEL_ROOT / model_path}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=True).to(device)
        print("Model loaded successfully.")
        return ModelSummaryGenerator(tokenizer, model, generation_config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_path", 
        type=str, 
        help="reletive path to configuration file"
    )
    parser.add_argument(
        "pairs_path",
        type=str,
        help="relative path to a file containing pairs of model paths"
    )
    parser.add_argument(
        "out_path",
        type=str,
        help="reletive path to output file"
    )
    args = parser.parse_args()

    config = OmegaConf.load(CONFIG_ROOT / args.config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )

    dataset_path = DATA_ROOT / config.dataset_name

    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path / "validation.csv", nrows=config.validation.nrow)
    df = df.rename(columns=dict(config.col_renames))
    dataset = Dataset.from_pandas(df)
    print(f"Successfully loaded")

    f = open(str(DATA_ROOT / args.pairs_path), "r")

    out_path = DATA_ROOT / args.out_path
    if out_path.exists():
        with open(str(out_path), "r") as check:
            try:
                existing = json.load(check)
            except:
                existing = []
    else:
        existing = []

    m = int(f.readline())

    win_rate_calculator = WinRateCalculator()
    win_rate_calculator.set_dataset(dataset)
    scorer = get_preference_scorer(config.validation.scorer, client)

    for i in range(m):
        ref_model_path, n = f.readline().split()
        n = int(n)

        ref_summary_generator = get_summary_generator(str(MODEL_ROOT / ref_model_path), config.validation.generation)
        win_rate_calculator.set_ref_generator(ref_summary_generator)
        win_rate_calculator.ref_generate()

        for j in range(n):
            target_model_path = f.readline()

            target_summary_generator = get_summary_generator(str(MODEL_ROOT / target_model_path), config.validation.generation)
            win_rate_calculator.set_target_generator(target_summary_generator)
            win_rate_calculator.target_generate()

            print(f"Calculating win rate of pair ({i}, {j})")
            win_rate, _ = win_rate_calculator.calculate_win_rate(dataset, scorer)
            print(f"Calculated successfully")
            
            res = {
                "ref_model": ref_model_path,
                "target_model": target_model_path,
                "win_rate": win_rate
            }
            for item in existing:
                if item["ref_model"] == ref_model_path and item["target_model"] == target_model_path:
                    item.update(res)
                    break
            else:
                existing.append(res)

    f.close()
    with open(str(out_path), "w") as f:
        json.dump(existing, f, indent=2)
