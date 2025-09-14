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
from typing import Any

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
        "out_path",
        type=str,
        help="relative path to JSON file to update in-place"
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

    out_path = DATA_ROOT / args.out_path
    # Load existing JSON list of pairs and win_rates
    if out_path.exists():
        with open(str(out_path), "r") as check:
            try:
                existing: list[dict] = json.load(check)
            except Exception:
                existing = []
    else:
        existing = []

    # Helper: write JSON back atomically after each update
    def write_json_atomic(path: Path, data: list[dict[str, Any]]) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(str(tmp_path), "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)

    # Prepare calculator and scorer
    win_rate_calculator = WinRateCalculator()
    win_rate_calculator.set_dataset(dataset)
    scorer = get_preference_scorer(config.validation.scorer, client)

    # Group targets by reference model for efficient generation
    from collections import defaultdict
    targets_by_ref: dict[str, list[str]] = defaultdict(list)

    # Map to index to update results back efficiently
    index_by_pair: dict[tuple[str, str], int] = {}

    for idx, item in enumerate(existing):
        ref_model_path = item.get("ref_model", "").strip()
        target_model_path = item.get("target_model", "").strip()
        win_rate_val = item.get("win_rate", None)

        # Consider win_rate empty if missing or None or empty string
        if ref_model_path and target_model_path:
            index_by_pair[(ref_model_path, target_model_path)] = idx
            if win_rate_val in (None, ""):
                targets_by_ref[ref_model_path].append(target_model_path)

    # print summary of what to process
    print(f"Total pairs in JSON: {len(existing)}")
    print(f"Pairs with empty win_rate to process: {sum(len(v) for v in targets_by_ref.values())}")
    print(f"Unique reference models to process: {len(targets_by_ref)}")
    for ref_model_path, targets in targets_by_ref.items():
        ref_path = MODEL_ROOT / ref_model_path
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference model path {ref_path} does not exist")
        print(f"  Reference model: {ref_model_path} -> {len(targets)} target models")
        for target_model_path in targets:
            target_path = MODEL_ROOT / target_model_path
            if not target_path.exists():
                raise FileNotFoundError(f"Target model path {target_path} does not exist")
            print(f"    Target model: {target_model_path}")

    if not targets_by_ref:
        print("No empty win_rate entries found. Nothing to do.")
    else:
        # Iterate each reference model once
        for i, (ref_model_path, targets) in enumerate(targets_by_ref.items()):
            print(f"Processing reference model {i+1}/{len(targets_by_ref)}: {ref_model_path}")
            ref_summary_generator = get_summary_generator(str(MODEL_ROOT / ref_model_path), config.validation.generation)
            win_rate_calculator.set_ref_generator(ref_summary_generator)
            win_rate_calculator.ref_generate()

            for j, target_model_path in enumerate(targets):
                target_summary_generator = get_summary_generator(str(MODEL_ROOT / target_model_path), config.validation.generation)
                win_rate_calculator.set_target_generator(target_summary_generator)
                win_rate_calculator.target_generate()

                print(f"Calculating win rate for pair ({ref_model_path}, {target_model_path})")
                win_rate, _ = win_rate_calculator.calculate_win_rate(dataset, scorer)
                print("Calculated successfully")

                # Update the corresponding item in-place
                idx = index_by_pair[(ref_model_path, target_model_path)]
                existing[idx]["win_rate"] = win_rate
                # Persist after each calculation
                write_json_atomic(out_path, existing)

    # Final write to ensure formatting and completeness (no-op if unchanged)
    write_json_atomic(out_path, existing)
