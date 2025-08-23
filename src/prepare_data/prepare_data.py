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

from abc import ABC, abstractmethod

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from openai import OpenAI

from datasets import Dataset
import pandas as pd
import json
from utils.utility import *

from omegaconf import OmegaConf

from summary_generator import *
from preference_scorers import *
from preference_builders import *
    
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

        print("Parsing Batch for preference...")
        compared = preference_scorer.compare_batch_2()
        print("Batch file Parsed")

        result = preference_builder.build_with_comparisons(compared)
        
        filename = get_filename("output", config.get_preference.builder, config.get_preference.scorer, suffix=".json")
        output_path = DATA_ROOT / config.output_dir / filename
        
        print(f"Saving result to {str(output_path)}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(output_path), "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        print("Saved successfully.")