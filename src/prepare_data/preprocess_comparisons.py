import os
from pathlib import Path
from dotenv import load_dotenv
import argparse

import torch

from openai import OpenAI

from datasets import Dataset
import json
from src.utils.utility import get_filename

from omegaconf import OmegaConf

from src.prepare_data.preference_scorers import (
    PreferenceScorer,
    is_preference_two_step,
    get_preference_scorer,
)

load_dotenv()

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).resolve()  # type: ignore
MODEL_ROOT = Path(os.getenv("MODEL_ROOT")).resolve()  # type: ignore
DATA_ROOT = Path(os.getenv("DATA_ROOT")).resolve()  # type: ignore
CONFIG_ROOT = Path(os.getenv("CONFIG_ROOT")).resolve()  # type: ignore
SRC_ROOT = Path(os.getenv("SRC_ROOT")).resolve()  # type: ignore


def generate_comparisons(dataset: Dataset, scorer: PreferenceScorer) -> list[dict]:
    """Generate comparison pairs from a dataset for preference scoring.

    Args:
        dataset (Dataset): Dataset containing prompts, references, and generations.
        scorer (PreferenceScorer): Scorer used to determine if references are needed.

    Returns:
        list[dict]: List of comparison dictionaries with prompts and responses.
    """
    pairs = []
    for k, example in enumerate(dataset):
        prompt = example["prompt"]  # type: ignore
        ref = example["reference"] if scorer.require_ref() else ""  # type: ignore
        generated = example["generated"]  # type: ignore

        for i, y1 in enumerate(generated):
            for j, y2 in enumerate(generated):
                if i < j:
                    pairs.append(
                        {
                            "prompt": prompt,
                            "y1": y1,
                            "y2": y2,
                            "ref": ref,
                            "id": f"{k}, {i}, {j}",
                        }
                    )

    return pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_path", type=str, help="reletive path to configuration file"
    )
    parser.add_argument(
        "gen_filename", type=str, help="reletive path to generation filename"
    )
    parser.add_argument(
        "--cached",
        "-o",
        help="reletive path to partially processed pairs",
        default="None",
    )
    args = parser.parse_args()

    config = OmegaConf.load(CONFIG_ROOT / args.config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    gen_filename = args.gen_filename
    gen_output_path = DATA_ROOT / config.dataset_output_dir / gen_filename
    print(f"Loading generated ouputs from {str(gen_output_path)}...")
    with open(gen_output_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    dataset = Dataset.from_dict(dataset)
    print("Loaded successfully")

    if args.cached == "None":
        filename = f"{config.name}_comparison_{config.scorer.type}.jsonl"
        output_path = DATA_ROOT / config.dataset_output_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path = Path(str(output_path))
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
            if val != "None" and val is not None:
                cache[key] = int(val)

    if not is_preference_two_step(config.scorer):
        print("Labeling data...")
        preference_scorer = get_preference_scorer(config.scorer, openai_client=client)

        comparisons = generate_comparisons(dataset, preference_scorer)
        comparisons = [
            comparison for comparison in comparisons if comparison["id"] not in cache
        ]
        compared = preference_scorer.compare_batch(comparisons)
        print("Labeled successfully.")

        print(f"Saving result to {output_path}...")
        with open(output_path, "w", encoding="utf-8") as f:
            for pair, compare in zip(comparisons, compared):
                line = json.dumps(
                    {"id": pair["id"], "result": compare}, ensure_ascii=False
                )
                f.write(line + "\n")
            for pair, compare in cache.items():
                line = json.dumps({"id": pair, "result": compare}, ensure_ascii=False)
                f.write(line + "\n")
        print("Saved successfully.")

    else:
        print("Creating jsonl file for request...")
        preference_scorer = get_preference_scorer(config.scorer, openai_client=client)

        comparisons = generate_comparisons(dataset, preference_scorer)
        comparisons = [
            comparison for comparison in comparisons if comparison["id"] not in cache
        ]
        requests = preference_scorer.compare_batch_0(comparisons)
        print("jsonl file created")

        base_path = get_filename(
            "request",
            config.scorer.type,
            suffix="",
        )
        print(
            f"Saving jsonl request to {str(DATA_ROOT / config.dataset_output_dir / base_path)} ({len(requests)} files)..."
        )

        paths = []
        for i, request in enumerate(requests):
            request_filename = get_filename(
                str(i).zfill(len(str(len(requests) - 1))), suffix=".jsonl"
            )
            request_output_path = (
                DATA_ROOT / config.dataset_output_dir / base_path / request_filename
            )

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

        print(f"Saving result to {str(output_path)}...")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(output_path), "w", encoding="utf-8") as f:
            for pair, compare in zip(comparisons, compared):
                line = json.dumps(
                    {"id": pair["id"], "result": compare}, ensure_ascii=False
                )
                f.write(line + "\n")
            for pair, compare in cache.items():
                line = json.dumps({"id": pair, "result": compare}, ensure_ascii=False)
                f.write(line + "\n")
        print("Saved successfully.")
