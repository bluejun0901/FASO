import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import argparse

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

from prepare_data.summary_generator import *
from prepare_data.preference_scorers import *
from validate.win_rate_calculator import WinRateCalculator

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
    out = open(str(DATA_ROOT / args.out_path), "w")

    m = int(f.readline())

    win_rate_calculator = WinRateCalculator()
    scorer = get_preference_scorer(config.validation.scorer, client)

    for i in range(m):
        ref_model_path, n = f.readline().split()
        n = int(n)

        print(f"Loading model from {MODEL_ROOT / ref_model_path}...")
        ref_tokenizer = AutoTokenizer.from_pretrained(MODEL_ROOT / ref_model_path, trust_remote_code=True)
        ref_model = AutoModelForCausalLM.from_pretrained(MODEL_ROOT / ref_model_path, trust_remote_code=True).to(device)
        print("Model loaded successfully.")

        ref_summary_generator = SummaryGenerator(ref_tokenizer, ref_model, config.validation.generation)
        win_rate_calculator.set_ref_generator(ref_summary_generator)

        for j in range(n):
            target_model_path = f.readline()

            print(f"Loading model from {MODEL_ROOT / target_model_path}...")
            target_tokenizer = AutoTokenizer.from_pretrained(MODEL_ROOT / target_model_path, trust_remote_code=True)
            target_model = AutoModelForCausalLM.from_pretrained(MODEL_ROOT / target_model_path, trust_remote_code=True).to(device)
            print("Model loaded successfully.")

            target_summary_generator = SummaryGenerator(target_tokenizer, target_model, config.validation.generation)
            win_rate_calculator.set_target_generator(target_summary_generator)

            print(f"Calculating win rate of pair ({i}, {j})")
            win_rate, _ = win_rate_calculator.calculate_win_rate(dataset, scorer)
            print(f"Calculated successfully")
            
            out.write(f"{win_rate:.05f} ")
        out.write("\n")

    f.close()
    out.close()