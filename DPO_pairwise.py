# TODO: eval data는 별도로 준비해서 넣어야 함(too long time for evaluation)

import torch
from transformers import AutoTokenizer, TrainerCallback
from trl import AutoModelForCausalLMWithValueHead, DPOTrainer, DPOConfig
from huggingface_hub import login

from openai import OpenAI
from difflib import SequenceMatcher

from datasets import Dataset
import pandas as pd

from torch.utils.tensorboard import SummaryWriter

import os
import json

device = "cuda" if torch.cuda.is_available() else "cpu"

def init():
    # use one gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"

    # openai client login
    global openai_client 
    openai_client = OpenAI(
        api_key="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"
    )

    # huggingface login
    login(token="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")

# get one summery from the model
def get_summery_from_model(model: AutoModelForCausalLMWithValueHead,
                           tokenizer: AutoTokenizer,
                           content: str) -> str:
    message = [
        {
            "role": "user",
            "content": f"Summarize the following text in a TL;DR style \
                in **one sentence**\n\n{content}\n"
        }
    ]
    prompt = tokenizer.apply_chat_template(message, tokenize=False, \
                                           add_generation_prompt=True)

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids)
    
    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        top_k=50,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )  
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # <|assistant|> 태그 이후 텍스트만 추출
    assistant_tag = "<|assistant|>"
    if assistant_tag in output_text:
        output_text = output_text.split(assistant_tag, 1)[1].strip()

    return output_text

# raw data preparation
def prepare_raw_data(model: AutoModelForCausalLMWithValueHead,
                     tokenizer: AutoTokenizer, 
                     dataset: Dataset) -> list:
    print("Preparing raw data...")
    
    raw_data = []

    model.eval()
    with torch.no_grad():
        for data in dataset:
            prompt = data['content']
            answer = data['answer']
            response_a = get_summery_from_model(model, tokenizer, prompt)
            response_b = get_summery_from_model(model, tokenizer, prompt)
            raw_data.append({
                "prompt": prompt,
                "response_a": response_a,
                "response_b": response_b,
                "answer": answer
            })
            print(f"Processed: {len(raw_data)}/{len(dataset)}", end="\r")

    print("\nRaw data preparation complete.")
    return raw_data

# simulate human preference using GPT-4o-mini
def get_preference_from_gpt(prompt: str, 
                            response_a: str, 
                            response_b: str) -> str:
    user_prompt = f"""
        [PROMPT]
        {prompt}

        [RESPONSE A]
        {response_a}

        [RESPONSE B]
        {response_b}

        Only output a single letter: A or B. 
        Do not explain your answer. 
        Do not include anything else.
        Which response is better?
    """

    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    content = completion.choices[0].message.content.strip().upper()
    return "A" if "A" in content else "B"

# simulate human preference by similarity
def get_preference_by_similarity(response_a: str, 
                                 response_b: str,
                                 answer: str) -> str:
    def similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    sim_a = similarity(response_a, answer)
    sim_b = similarity(response_b, answer)

    return "A" if sim_a >= sim_b else "B"
    

# prepare pairwise compared data
def prepare_pairwise_data(raw_data: list) -> list:
    print("Preparing pairwise compared data...")
    
    pairwise_data = []
    for item in raw_data:
        prompt = item['prompt']
        response_a = item['response_a']
        response_b = item['response_b']
        
        # determine which response is closer to the ground truth answer
        answer = item["answer"]
        better = get_preference_by_similarity(response_a, response_b, answer)  # TODO: use GPT-4o-mini for better preference simulation
        if better == "A":
            chosen, rejected = response_a, response_b
        else:
            chosen, rejected = response_b, response_a
        
        pairwise_data.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        })
        print(f"Processed: {len(pairwise_data)}/{len(raw_data)}", end="\r")

    print("\nPairwise data preparation complete.")
    return pairwise_data

# compute win rate for evaluation
def compute_win_rate(model: AutoModelForCausalLMWithValueHead,
                     tokenizer: AutoTokenizer,
                     pairwise_data: list) -> float:
    model.eval()
    wins = 0
    total = len(pairwise_data)
    
    with torch.no_grad():
        for item in pairwise_data:
            prompt = item["prompt"]
            chosen = item["chosen"]
            rejected = item["rejected"]
            
            def get_logprob(text):
                inputs = tokenizer(prompt + text, return_tensors="pt").to(device)
                outputs = model(**inputs, labels=inputs["input_ids"])
                # outputs.loss is the negative log-likelihood per token
                return -outputs.loss.item()
            
            lp_chosen = get_logprob(chosen)
            lp_rejected = get_logprob(rejected)
            
            if lp_chosen > lp_rejected:
                wins += 1
        
    return wins / total

if __name__ == "__main__":
    init()

    # load model and tokenizer
    print("loading model")
    model_path = "models/sft/TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path).to(device)
    model.warnings_issued = {}
    print("model loaded")

    # load reference model
    print("loading reference model")
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path).to("cpu")
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    print("reference model loaded")

    # load dataset
    print("loading dataet")
    df = pd.read_csv("~/.kaggle/cnn_dailymail/train.csv", nrows=3000) # TODO: change here
    df = df[["article", "highlights"]].rename(columns={
        "article": "content",
        "highlights": "answer"
    })
    dataset = Dataset.from_pandas(df)
    print("dataset loaded")

    # prepare raw data
    raw_data = prepare_raw_data(model, tokenizer, dataset)
    with open("cache/raw_data.json", "w", encoding="utf-8") as f:
        json.dump(raw_data, f, ensure_ascii=False, indent=4)

    # prepare pairwise compared data
    dpo_data = prepare_pairwise_data(raw_data)

    # prepare dataset for DPO
    train_dataset = Dataset.from_list(dpo_data)

    # set DPO training
    training_args = DPOConfig(
        output_dir="models/rlhf/DPO_pairwise/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        logging_steps=5,
        logging_dir="logs/DPO_pairwise/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        fp16=True,
        num_train_epochs=5.0,
        save_steps=50,
        save_strategy="steps",
        save_total_limit=10
    )
    trainer = DPOTrainer(
        model=model,
        args=training_args, 
        processing_class=tokenizer, 
        train_dataset=train_dataset,
    )

    # set logging callback
    class WinRateCallback(TrainerCallback):
        def __init__(self, model, tokenizer, eval_data, writer):
            self.model = model
            self.tokenizer = tokenizer
            self.eval_data = eval_data
            self.writer = writer
        
        def on_log(self, args, state, control, logs=None, **kwargs):
            win_rate = compute_win_rate(self.model, self.tokenizer, self.eval_data)
            print(f"\nStep {state.global_step}: Win rate = {win_rate:.4f}\n")
            self.writer.add_scalar("eval/win_rate", win_rate, state.global_step)
            self.writer.flush()

    writer = SummaryWriter(log_dir="logs/DPO_pairwise/TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    trainer.add_callback(WinRateCallback(model, tokenizer, dpo_data, writer))

    # run
    trainer.train()