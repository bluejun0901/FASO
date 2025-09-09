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

from openai import OpenAI

from datasets import Dataset
import json
from utils.utility import *

from omegaconf import OmegaConf

from preference_scorers import *

def generate_comparisons(dataset: Dataset, scorer: PreferenceScorer) -> list[dict]:
    pairs = []
    for k, example in enumerate(dataset):
        prompt = example['prompt'] # type: ignore
        ref = example['reference'] if scorer.require_ref() else "" # type: ignore
        summaries = example['summaries'] # type: ignore
        
        for i, y1 in enumerate(summaries):
            for j, y2 in enumerate(summaries):
                if i < j:
                    pairs.append({
                        'prompt': prompt,
                        'y1': y1,
                        'y2': y2,
                        'ref': ref,
                        'meta': f"{k}, {i}, {j}"
                    })
    
    return pairs
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_path", 
        type=str, 
        help="reletive path to configuration file"
    )
    parser.add_argument(
        "gen_filename", 
        type=str, 
        help="reletive path to generation filename"
    )
    parser.add_argument(
        "--cached",
        "-o",
        help="reletive path to partially processed pairs",
        default="None")
    args = parser.parse_args()

    config = OmegaConf.load(CONFIG_ROOT / args.config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )

    gen_filename = args.gen_filename
    gen_output_path = DATA_ROOT / config.dataset_output_dir / gen_filename
    print(f"Loading generated ouputs from {str(gen_output_path)}...")
    with open(gen_output_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    dataset = Dataset.from_dict(dataset)
    print("Loaded successfully")

    if args.cached == "None":
        filename = get_filename(
            "comparison",
            config.builder.type,
            config.scorer.type,
            suffix=".jsonl",
        )
        output_path = DATA_ROOT / config.dataset_output_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path = str(output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            pass
    else:
        output_path = str(DATA_ROOT / config.dataset_output_dir / args.cached)

    cache = {}
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = str(obj["id"])
            val = obj["result"]
            if val != 'None' and val is not None:
                cache[key] = int(val)

    if is_preference_two_step(config.scorer) == False:
        print("Labeling data...")
        preference_scorer = get_preference_scorer(config.scorer, openai_client=client)
        
        comparisons = generate_comparisons(dataset, preference_scorer)
        comparisons = [comparison for comparison in comparisons if comparison['meta'] not in cache]
        compared = preference_scorer.compare_batch(comparisons)
        print("Labeled successfully.")
        
        print(f"Saving result to {output_path}...")
        with open(output_path, "w", encoding="utf-8") as f:
            for pair, compare in zip(comparisons, compared):
                line = json.dumps({"id": pair['meta'], "result": compare}, ensure_ascii=False)
                f.write(line + "\n")
            for pair, compare in cache.items():
                line = json.dumps({"id": pair, "result": compare}, ensure_ascii=False)
                f.write(line + "\n")
        print("Saved successfully.")

    else:
        print("Creating jsonl file for request...")
        preference_scorer = get_preference_scorer(config.scorer, openai_client=client)

        comparisons = generate_comparisons(dataset, preference_scorer)
        comparisons = [comparison for comparison in comparisons if comparison['meta'] not in cache]
        requests = preference_scorer.compare_batch_0(comparisons)
        print("jsonl file created")

        base_path = get_filename(
            "request",
            config.builder.type,
            config.scorer.type,
            suffix="",
        )
        print(f"Saving jsonl request to {str(DATA_ROOT / config.dataset_output_dir / base_path)} ({len(requests)} files)...")

        paths = []
        for i, request in enumerate(requests):
            request_filename = get_filename(
                str(i).zfill(len(str(len(requests) - 1))),
                suffix=".jsonl"
            )
            request_output_path = DATA_ROOT / config.dataset_output_dir / base_path / request_filename

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
        
        filename = get_filename(
            "comparison",
            config.builder.type,
            config.scorer.type,
            suffix=".jsonl",
        )
        output_path = DATA_ROOT / config.dataset_output_dir / filename
        
        print(f"Saving result to {str(output_path)}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(output_path), "w", encoding="utf-8") as f:
            for pair, compare in zip(comparisons, compared):
                line = json.dumps({"id": pair['meta'], "result": compare}, ensure_ascii=False)
                f.write(line + "\n")
            for pair, compare in cache.items():
                line = json.dumps({"id": pair, "result": compare}, ensure_ascii=False)
                f.write(line + "\n")
        print("Saved successfully.")
