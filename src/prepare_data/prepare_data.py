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

from openai import OpenAI

from datasets import Dataset
import json
from utils.utility import *

from omegaconf import OmegaConf

from preference_scorers import *
from preference_builders import *
    
if __name__ == "__main__":
    config = OmegaConf.load(CONFIG_ROOT / input("input configuration path: "))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )

    model_path = MODEL_ROOT / config.model_name
    dataset_path = DATA_ROOT / config.dataset_name

    gen_filename = input("input generated filename: ")
    gen_output_path = DATA_ROOT / config.output_dir / gen_filename
    print(f"Loading generated ouputs from {str(gen_output_path)}...")
    with open(gen_output_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    dataset = Dataset.from_dict(dataset)
    print("Loaded successfully")
    
    comparison_file = DATA_ROOT / config.output_dir / input("input comparison filename: ")

    preference_scorer = CachedPreferenceScorer(str(comparison_file))
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