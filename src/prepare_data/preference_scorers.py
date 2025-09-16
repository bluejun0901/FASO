from abc import ABC, abstractmethod
from typing import Union, Any, Callable, TypeVar

from datasets import Dataset

from rouge_score import rouge_scorer
from openai import OpenAI
from openai.types import Batch
import re
import random
import time

import json
from tqdm import tqdm

from omegaconf import OmegaConf
T = TypeVar("T")

class PreferenceScorer(ABC):
    @abstractmethod
    def require_ref(self) -> bool:
        pass
    
    @abstractmethod
    def compare(self, prompt: str, y1: str, y2: str, ref: str, meta: str | None=None) -> int | None:
        pass
    
    @abstractmethod
    def compare_batch(self, pairs: Union[list[dict], Dataset]) -> list[int | None]:
        pass
    
class ROUGEPreferenceScorer(PreferenceScorer):
    def __init__(self, config: OmegaConf):
        self.require_ref_flag = True
        self.rouge_type = config.type
        self.scorer = rouge_scorer.RougeScorer([self.rouge_type], use_stemmer=True)
        
    def require_ref(self):
        return self.require_ref_flag
                        
    def compare(self, prompt: str, y1: str, y2: str, ref: str, meta: str | None=None) -> int | None:
        s1 = self.scorer.score(ref, y1)[self.rouge_type].fmeasure
        s2 = self.scorer.score(ref, y2)[self.rouge_type].fmeasure
                        
        return 0 if s1 > s2 else 1
    
    def compare_batch(self, pairs: Union[list[dict], Dataset]) -> list[int | None]:
        compared = []
        for pair in pairs:
            compared.append(self.compare(pair['prompt'], pair['y1'], pair['y2'], pair['ref'])) # type: ignore
        return compared
    
class OpenAIPreferenceScorer(PreferenceScorer):
    def __init__(self, client: OpenAI, config: OmegaConf):
        self.require_ref_flag = False
        self.client = client
        self.model_name = config.model
        self.prompt_template = config.prompt
        self.pattern: str = config.preference_pattern
        
    def require_ref(self):
        return self.require_ref_flag
                      
    def compare(self, prompt: str, y1: str, y2: str, ref: str="") -> int | None:
        # Randomize which response is shown first to the judge
        swapped = random.choice([False, True])
        first = y2 if swapped else y1
        second = y1 if swapped else y2
        user_prompt = self.prompt_template.format(article=prompt, summary1=first, summary2=second)

        response = self.client.responses.create(
            model=self.model_name,
            input=user_prompt,
            reasoning={
                "effort": "minimal"
            },
            max_output_tokens=128
        )
        output_text = response.output[1].content[0].text #type: ignore
        assert output_text is not None

        match = re.search(self.pattern, output_text)
        if not match:
            return None

        judged_idx = 0 if match.group(1) == "1" else 1

        return (1 - judged_idx) if swapped else judged_idx

    def compare_batch(self, pairs: Union[list[dict], Dataset]) -> list[int | None]:
        compared = []
        for pair in tqdm(pairs, desc="Comparing"):
            compared.append(self.compare(pair['prompt'], pair['y1'], pair['y2'])) #type: ignore
        return compared
    
class CachedPreferenceScorer(PreferenceScorer):
    def __init__(self, comparison_file: str):
        self.require_ref_flag = False
        self.comparison_file_path = comparison_file
        self._cache: dict[str, int] = {}
        self._load_file()

    def _normalize_id(self, s: str) -> str:
        parts = [p.strip() for p in s.split(",")]
        return ", ".join(parts)

    def _load_file(self) -> None:
        self._cache.clear()
        with open(self.comparison_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                key = self._normalize_id(str(obj["id"]))
                val = int(obj["result"])
                self._cache[key] = val

    def require_ref(self):
        return self.require_ref_flag

    def compare(self, prompt: str, y1: str, y2: str, ref: str, meta: str | None = None) -> int | None:
        if meta is None:
            raise Exception("'meta' field must be included when calling CachedPreferenceScorer")
        key = self._normalize_id(str(meta))
        if key not in self._cache:
            raise KeyError(f"Comparison id '{key}' not found in {self.comparison_file_path}")
        return self._cache[key]

    def compare_batch(self, pairs: Union[list[dict], Dataset]) -> list[int | None]:
        compared: list[int | None] = []
        for pair in pairs:
            compared.append(self.compare(pair['prompt'], pair['y1'], pair['y2'], pair['ref'], pair['meta']))  # type: ignore
        return compared
    
class BatchPreferenceScorer(PreferenceScorer):
    @abstractmethod
    def require_ref(self) -> bool:
        pass

    def compare(self, prompt: str, y1: str, y2: str, ref: str) -> int | None:
        raise Exception("BatchPreferenceScorer doesn't support 'compare'. Try calling compare_batch_* instead.")

    def compare_batch(self, pairs: Union[list[dict], Dataset]) -> list[int]:
        raise Exception("BatchPreferenceScorer doesn't support 'compare_batch'. Try calling compare_batch_* instead.")
    
    @abstractmethod
    def compare_batch_0(self, pairs: Union[list[dict], Dataset]) -> list[list[dict]]:
        pass

    @abstractmethod
    def compare_batch_1(self, path: list[str]) -> dict:
        pass

    @abstractmethod
    def compare_batch_2(self) -> list[int | None]:
        pass

class OpenAIBatchPreferenceScorer(BatchPreferenceScorer):
    def __init__(self, client: OpenAI, config: OmegaConf):
        self.require_ref_flag = False
        self.client = client
        self.model_name = config.model
        self.prompt_template = config.prompt
        self.pattern = config.preference_pattern
        c = getattr(config, "batch", None)
        self.max_concurrent = getattr(c, "max_concurrent", 3) if c is not None else 3
        self.max_retries = getattr(c, "max_retries", 5) if c is not None else 5
        self.initial_backoff = getattr(c, "initial_backoff", 1.0) if c is not None else 1.0
        self.poll_interval = getattr(c, "poll_interval", 15.0) if c is not None else 15.0
        self.max_request_size = getattr(c, "max_request_per_batch", 30000) if c is not None else 30000

        self.paths = []
        self.batch_files = []
        self.batchs = []
        self.total = 0

        self.pairs = []

        self.swapped: list[bool] = []
    
    def require_ref(self):
        return self.require_ref_flag

    def compare_batch_0(self, pairs: Union[list[dict], Dataset]) -> list[list[dict]]:
        requests = []
        self.total = len(pairs)
        for i, pair in enumerate(pairs):
            self.pairs.append(pair)
            if i % self.max_request_size == 0:
                requests.append([])

            prompt = pair['prompt'] # type: ignore
            y1 = pair['y1'] # type: ignore
            y2 = pair['y2'] # type: ignore

            # Randomize presentation order per pair and record mapping
            swapped = random.choice([False, True])
            self.swapped.append(swapped)
            summary1 = y2 if swapped else y1
            summary2 = y1 if swapped else y2

            user_prompt = self.prompt_template.format(article=prompt, summary1=summary1, summary2=summary2)
            body = {
                "model": self.model_name,
                "input": user_prompt,
                "reasoning": {
                    "effort": "minimal"
                },
                "max_output_tokens": 128
            }
            requests[i // self.max_request_size].append({
                "custom_id": f"{i} {swapped}",
                "method": "POST",
                "url": "/v1/responses",
                "body": body
            })

        return requests
    
    def _retry(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T: # type: ignore
        delay = self.initial_backoff
        for i in range(self.max_retries):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                if i == self.max_retries - 1:
                    raise e
                time.sleep(delay + random.random() * 0.25 * delay)
                delay *= 2

    def _submit_one(self, path: str) -> tuple[int, Batch]:
        with open(path, "rb") as f:
            batch_file = self._retry(self.client.files.create, file=f, purpose="batch")
        batch = self._retry(
            self.client.batches.create,
            input_file_id=batch_file.id,
            endpoint="/v1/responses",
            completion_window="24h",
        )
        self.batch_files.append(batch_file)
        self.batchs.append(batch)
        tqdm.write(f"file {path} (file id: {batch_file.id}) submitted (batch id: {batch.id})")
        return (1, batch)

    def compare_batch_1(self, paths: list[str], max_concurrent: int | None = None) -> dict:
        self.paths = list(paths)
        pending = list(paths)
        random.shuffle(pending)
        in_flight: dict[str, Batch] = {}
        batch_to_path: dict[str, str] = {}
        attempts: dict[str, int] = {}
        finished: list[Batch] = []
        can_add_batch = True

        count = {
            "completed": 0,
            "failed": 0,
            "expired": 0,
            "canceled": 0,
        }

        if max_concurrent is None:
            max_concurrent = self.max_concurrent
        assert max_concurrent is not None

        pbar_total = self.total if self.total is not None else 0
        pbar = tqdm(total=pbar_total, desc="Batch progress", unit="req")

        while can_add_batch and pending and len(in_flight) < max_concurrent:
            submit = pending.pop(0)
            attempts[submit] = attempts.get(submit, 0) + 1
            b = self._submit_one(submit)[1]
            while b.status in ("validating",):
                time.sleep(self.poll_interval)
                b = self._retry(self.client.batches.retrieve, b.id)
            if b.status in ("failed", "expired", "canceled"):
                # Requeue if attempts remain, otherwise count as final failure state
                if attempts[submit] < self.max_retries:
                    pending.append(submit)
                if in_flight:
                    can_add_batch = False
            else:
                in_flight[b.id] = b
                batch_to_path[b.id] = submit

        last_completed = 0
        try:
            while pending or in_flight:
                time.sleep(self.poll_interval)

                to_delete = []
                for batch_id in list(in_flight.keys()):
                    b = self._retry(self.client.batches.retrieve, batch_id)
                    in_flight[batch_id] = b

                    if b.status in ("completed", "failed", "expired", "canceled"):
                        to_delete.append(batch_id)
                        finished.append(b)
                        if b.status == "completed":
                            can_add_batch = True
                        else:
                            path = batch_to_path.get(batch_id)
                            if path is not None:
                                if attempts.get(path, 0) < self.max_retries:
                                    attempts[path] = attempts.get(path, 0) + 1
                                    pending.append(path)
                            can_add_batch = True
                        if b.status in count:
                            count[b.status] += 1
                        else:
                            count[b.status] = 1
                        tqdm.write(f"{batch_id} {b.status}")

                for bid in to_delete:
                    in_flight.pop(bid, None)

                agg_counts = {"total": 0, "completed": 0, "failed": 0}
                agg_counts["total"] = self.total
                for _b in list(in_flight.values()) + finished:
                    rc = _b.request_counts
                    if rc:
                        agg_counts["completed"] += rc.completed
                        agg_counts["failed"] += rc.failed

                if agg_counts["total"] > 0:
                    delta = agg_counts["completed"] - last_completed
                    if delta > 0:
                        pbar.update(delta)
                        last_completed += delta
                    pbar.set_postfix({
                        "failed": agg_counts["failed"],
                        "running": len(in_flight),
                        "finished": len(finished),
                    })

                while can_add_batch and pending and len(in_flight) < max_concurrent:
                    submit = pending.pop(0)
                    attempts[submit] = attempts.get(submit, 0) + 1
                    b = self._submit_one(submit)[1]
                    while b.status in ("validating",):
                        time.sleep(self.poll_interval)
                        b = self._retry(self.client.batches.retrieve, b.id)
                    if b.status in ("failed", "expired", "canceled"):
                        if attempts[submit] < self.max_retries:
                            pending.append(submit)
                        if in_flight:
                            can_add_batch = False
                    else:
                        in_flight[b.id] = b
                        batch_to_path[b.id] = submit

        finally:
            pbar.close()

        by_id = {b.id: b for b in self.batchs}
        for b in finished:
            by_id[b.id] = b
        self.batchs = list(by_id.values())

        return count

    def _parse_output_line(self, line: str) -> tuple[int, int | None]:
        obj = json.loads(line)
        idx = int(obj.get("custom_id").split()[0])  # came from compare_batch_0
        body = obj.get("response", {}).get("body", {})
        output = body.get("output", [])[1].get("content", "")[0]
        if not output:
            return idx, None
        output_text = output.get("text", "")
        match = re.search(self.pattern, output_text)
        if not match:
            return idx, None
        
        judged_idx = 0 if match.group(1) == "1" else 1
        swapped = self.swapped[idx] if 0 <= idx < len(self.swapped) else False
        orig_idx = (1 - judged_idx) if swapped else judged_idx
        return idx, orig_idx
    
    def compare_batch_2(self) -> list[int | None]:
        result: list[int | None] = [None for _ in range(len(self.pairs))]

        for b in self.batchs:
            if getattr(b, "status", None) == "completed" and getattr(b, "output_file_id", None):
                content_resp = self._retry(self.client.files.content, b.output_file_id)
                data = getattr(content_resp, "text", None)
                if data is None:
                    raw = content_resp.read()
                    if isinstance(raw, bytes):
                        data = raw.decode("utf-8", errors="ignore")
                    else:
                        data = str(raw)
                for raw_line in data.splitlines():
                    if not raw_line.strip():
                        continue
                    idx, pref = self._parse_output_line(raw_line)
                    if 0 <= idx < len(result):
                        result[idx] = pref

        return result

def get_preference_scorer(config: OmegaConf, openai_client: OpenAI | None) -> PreferenceScorer:
    if config.type.lower() == "rouge":
        return ROUGEPreferenceScorer(config.rouge)
    if config.type.lower() == "openai":
        assert openai_client is not None
        if config.openai.type != "batch":
            return OpenAIPreferenceScorer(openai_client, config.openai)
        else:
            return OpenAIBatchPreferenceScorer(openai_client, config.openai)
    raise Exception("Unknown scorer")

def is_preference_two_step(config: OmegaConf) -> bool:
    return config.type.lower() == "openai" and getattr(config.openai, "type", "") == "batch"
