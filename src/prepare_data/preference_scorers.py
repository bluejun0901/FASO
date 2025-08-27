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
        user_prompt = self.prompt_template.format(article=prompt, summary1=y1, summary2=y2)

        response = self.client.responses.create(
            model=self.model_name,
            input=user_prompt
        )
        output_text = response.choices[0].message.content
        assert output_text is not None

        match = re.search(self.pattern, output_text)
        if not match:
            return None
        # '1' means y1 chosen -> 0, '2' means y2 chosen -> 1
        return 0 if match.group(1) == "1" else 1

    def compare_batch(self, pairs: Union[list[dict], Dataset]) -> list[int | None]:
        compared = []
        for pair in pairs:
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

        self.paths = []
        self.batch_files = []
        self.batchs = []
        self.total = 0

        self.pairs = []

    
    def require_ref(self):
        return self.require_ref_flag
    
    def compare_batch_0(self, pairs: Union[list[dict], Dataset], request_size=30000) -> list[list[dict]]:
        requests = []
        self.total = len(pairs)
        for i, pair in enumerate(pairs):
            self.pairs.append(pair)
            if i % request_size == 0:
                requests.append([])

            prompt = pair['prompt'] # type: ignore
            summary1 = pair['y1'] # type: ignore
            summary2 = pair['y2'] # type: ignore

            user_prompt = self.prompt_template.format(article=prompt, summary1=summary1, summary2=summary2)
            body = {
                "model": self.model_name,
                "input": user_prompt,
                "reasoning": {
                    "effort": "minimal"
                }
            }
            requests[i // request_size].append({
                "custom_id": f"{i}",
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

    def _submit_one(self, path: str) -> Batch:
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
        print(f"file {path} (file id: {batch_file.id}) submitted (batch id: {batch.id})")
        return batch

    def compare_batch_1(self, paths: list[str], max_concurrent: int | None = None) -> dict:
        pending = list(paths)
        in_flight: dict[str, Batch] = {}
        finished: list[Batch] = []

        count = {
            "completed": 0,
            "failed": 0,
            "expired": 0,
            "canceled": 0,
        }

        if max_concurrent is None:
            max_concurrent = self.max_concurrent
        assert max_concurrent is not None

        while pending and len(in_flight) < max_concurrent:
            b = self._submit_one(pending.pop(0))
            in_flight[b.id] = b

        while pending or in_flight:
            time.sleep(self.poll_interval)

            to_delete = []
            for batch_id in list(in_flight.keys()):
                b = self._retry(self.client.batches.retrieve, batch_id)

                if b.status in ("completed", "failed", "expired", "canceled"):
                    to_delete.append(batch_id)
                    finished.append(b)
                    if b.status in count:
                        count[b.status] += 1
                    else:
                        count[b.status] = 1
                    print(f"{batch_id} {b.status}")

            for bid in to_delete:
                in_flight.pop(bid, None)

            agg_counts = {"total": 0, "completed": 0, "failed": 0}
            agg_counts["total"] = self.total
            for _b in list(in_flight.values()) + finished:
                rc = getattr(_b, "request_counts", None)
                if rc:
                    agg_counts["completed"] += rc.completed
                    agg_counts["failed"] += rc.failed

            if agg_counts["total"] >= 0:
                gprog = int(50 * agg_counts["completed"] / agg_counts["total"])
                gbar = "[" + "#" * gprog + "-" * (50 - gprog) + "]"
                print(f"ALL progress: {gbar} {agg_counts['completed']}/{agg_counts['total']} ", f"(failed: {agg_counts['failed']})")

            while pending and len(in_flight) < max_concurrent:
                b = self._submit_one(pending.pop(0))
                in_flight[b.id] = b

        by_id = {b.id: b for b in self.batchs}
        for b in finished:
            by_id[b.id] = b
        self.batchs = list(by_id.values())

        return count

    def _parse_output_line(self, line: str) -> tuple[int, int | None]:
        obj = json.loads(line)
        idx = int(obj["custom_id"])  # came from compare_batch_0
        body = obj.get("response", {}).get("body", {})
        choices = body.get("choices", [])
        if not choices:
            return idx, None
        msg = choices[0].get("message", {})
        output_text = msg.get("content", "")
        match = re.search(self.pattern, output_text)
        if not match:
            return idx, None
        # Map '1' -> 0 (y1), '2' -> 1 (y2)
        return idx, (0 if match.group(1) == "1" else 1)
    
    def compare_batch_2(self) -> list[int | None]:
        result: list[int | None] = [None for _ in range(len(self.pairs))]

        for b in self.batchs:
            if getattr(b, "status", None) == "completed" and getattr(b, "output_file_id", None):
                content_resp = self._retry(self.client.files.content, b.output_file_id)
                # Support both text attribute and binary stream
                data = getattr(content_resp, "text", None)
                if data is None:
                    # Assume file-like stream with .read()
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