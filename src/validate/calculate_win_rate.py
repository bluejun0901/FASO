import os
from pathlib import Path
import shutil
from dotenv import load_dotenv
import argparse

load_dotenv()

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).resolve() # type: ignore
MODEL_ROOT = Path(os.getenv("MODEL_ROOT")).resolve() # type: ignore
DATA_ROOT = Path(os.getenv("DATA_ROOT")).resolve() # type: ignore
CONFIG_ROOT = Path(os.getenv("CONFIG_ROOT")).resolve() # type: ignore
SRC_ROOT = Path(os.getenv("SRC_ROOT")).resolve() # type: ignore

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
        print(f"Loading model from {model_path}...")
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
        help="relative path to configuration file"
    )
    parser.add_argument(
        "target_path",
        type=str,
        help="relative path to target model or directory of models"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="optional relative path for outputs; defaults to config.result_dir"
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

    ref_model_path = MODEL_ROOT / config.validation.ref_model_path
    models_path = MODEL_ROOT / args.target_path
    target_paths: list[Path] = []
    for path in models_path.rglob("*"):
        if path.is_dir() and path.name.startswith("checkpoint-"):
            if not (path / "config.json").exists():
                shutil.copy(MODEL_ROOT / "config.json", path / "config.json")
            if not (path / "generation_config.json").exists():
                shutil.copy(MODEL_ROOT / "generation_config.json", path / "generation_config.json")
            # keep as Path relative to MODEL_ROOT for consistent naming
            rel_path = path.relative_to(MODEL_ROOT)
            target_paths.append(rel_path)

    # Determine output directory
    result_dir = Path(args.out_dir) if args.out_dir is not None else Path(config.result_dir)
    out_path = DATA_ROOT / result_dir
    out_path.mkdir(parents=True, exist_ok=True)

    print("Looking for existing results...")
    results: dict[str, list] = {}
    for target_model_path in target_paths:
        name = str(target_model_path.parent.name)
        results[name] = []
        result_file = out_path / f"{name}_win_rate.json"
        if result_file.exists():
            with open(result_file, "r") as f:
                existing_results = json.load(f)
                # ensure list
                if isinstance(existing_results, list):
                    results[name].extend(existing_results)
                else:
                    results[name].append(existing_results)
    print("Existing results loaded.")

    def write_json_atomic(path: Path, data: list) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(str(tmp_path), "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)

    win_rate_calculator = WinRateCalculator()
    win_rate_calculator.set_dataset(dataset)
    scorer = get_preference_scorer(config.validation.scorer, client)

    print(f"Processing reference model: {ref_model_path}")
    ref_summary_generator = get_summary_generator(str(ref_model_path), config.validation.generation)
    win_rate_calculator.set_ref_generator(ref_summary_generator)
    win_rate_calculator.ref_generate()

    # Save reference generations
    ref_output_out_filename = out_path / "outputs" / f"{config.validation.ref_model_path.replace('/', '_')}.json"
    ref_output_out_filename.parent.mkdir(parents=True, exist_ok=True)
    # Convert Dataset to list[dict] for JSON writing
    if win_rate_calculator.ref_responses is not None:
        write_json_atomic(ref_output_out_filename, win_rate_calculator.ref_responses.to_list())
        print(f"Reference outputs saved to {ref_output_out_filename}")

    for j, target_model_path in enumerate(target_paths):
        steps = int(str(target_model_path).split("-")[-1]) if "checkpoint-" in str(target_model_path) else -1
        name = str(target_model_path.parent.name)
        if steps in [x.get('step') for x in results.get(name, [])]:
            print(f"Skipping target model {target_model_path}'s checkpoint {steps} as results already exist.")
            continue
        print(f"Processing target model {j+1}/{len(target_paths)}: {target_model_path}")
        target_summary_generator = get_summary_generator(str(MODEL_ROOT / target_model_path), config.validation.generation)
        win_rate_calculator.set_target_generator(target_summary_generator)
        win_rate_calculator.target_generate()

        # Save target generations
        target_output_out_filename = out_path / "outputs" / f"{str(target_model_path).replace('/', '_')}.json"
        target_output_out_filename.parent.mkdir(parents=True, exist_ok=True)
        if win_rate_calculator.target_responses is not None:
            target_records = win_rate_calculator.target_responses.to_list()
            write_json_atomic(target_output_out_filename, target_records)
            print(f"Target outputs saved to {target_output_out_filename}")

        print(f"Calculating win rate for {target_model_path}")
        win_rate, comp = win_rate_calculator.calculate_win_rate(dataset, scorer)
        print("Calculated successfully")

        result = {
            "ref_model": config.validation.ref_model_path,
            "target_model": name,
            "step": steps,
            "win_rate": win_rate,
            "n_compared": len([c for c in comp if c is not None]),
            "n_win": len([c for c in comp if c == 1]),
            "n_rows": len(comp),
            "comparisons": comp
        }

        result_out_filename = out_path / f"{name}_win_rate.json"
        result_out_filename.parent.mkdir(parents=True, exist_ok=True)
        results[name].append(result)
        write_json_atomic(result_out_filename, results[name])
