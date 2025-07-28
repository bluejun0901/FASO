import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import torch
from transformers import AutoTokenizer, TrainerCallback
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, Trainer
from huggingface_hub import login

from datasets import Dataset
import pandas as pd

import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).resolve()
MODEL_ROOT = Path(os.getenv("MODEL_ROOT")).resolve()
DATA_ROOT = Path(os.getenv("DATA_ROOT")).expanduser().resolve()

device = "cuda" if torch.cuda.is_available() else "cpu"

def process_one_data(data: dict, 
                     tokenizer: AutoTokenizer) -> dict:
    prompt = f"Summarize the following text in a TL;DR style in **one sentence**\n\n{data['article']}\n"
    response = data["highlights"]
    full_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
        tokenize=False,
    )

    tokenized = tokenizer(
        full_text,
        truncation=True,
        padding="max_length",
        max_length=512,
    )

    prompt_only = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
    )
    prompt_len = len(tokenizer(prompt_only, truncation=True, max_length=512)["input_ids"])

    labels = tokenized["input_ids"].copy()
    labels[:prompt_len] = [-100] * prompt_len
    tokenized["labels"] = labels

    return tokenized

def preprocess_dataset(dataset: Dataset,
                       tokenizer: AutoTokenizer) -> Dataset:
    """
    Preprocess the dataset by applying the process_one_date function to each example.
    """
    return dataset.map(
        process_one_data,
        batched=False,
        remove_columns=dataset.column_names,
        desc="Processing dataset",
        fn_kwargs={"tokenizer": tokenizer},
    )

if __name__ == "__main__":
    login(token=os.getenv("HF_TOKEN"))

    # load model and tokenizer 
    print("loading model")
    model_path = MODEL_ROOT / 'sft' / 'TinyLlama' / 'TinyLlama-1.1B-Chat-v1.0'

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(str(model_path)).to(device)
    model.warnings_issued = {}
    print("model loaded")

    print("loading dataet")
    dataset_path = DATA_ROOT / '.kaggle' / 'cnn_dailymail' / 'train.csv'
    df = pd.read_csv(dataset_path)  # Load subset of dataset
    dataset = Dataset.from_pandas(df)
    print("dataset loaded")

    print("preprocessing dataset")
    dataset = preprocess_dataset(dataset, tokenizer)
    print("dataset preprocessed")

    output_dir = MODEL_ROOT / 'finetuned' / 'TinyLlama' / 'TinyLlama-1.1B-Chat-v1.0'

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        logging_steps=50,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=5,
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )

    print("starting training")
    trainer.train()
    print("training finished")