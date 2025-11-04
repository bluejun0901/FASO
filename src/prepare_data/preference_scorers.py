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
    """Abstract interface for scoring preferences between model outputs."""

    @abstractmethod
    def require_ref(self) -> bool:
        """Indicate whether the scorer requires reference text.

        Returns:
            bool: True if reference text is needed, otherwise False.
        """
        pass

    @abstractmethod
    def compare(
        self, prompt: str, y1: str, y2: str, ref: str, id: str | None = None
    ) -> int | None:
        """Score a single pair of responses.

        Args:
            prompt (str): Prompt text associated with the responses.
            y1 (str): First candidate response.
            y2 (str): Second candidate response.
            ref (str): Reference text if required by the scorer.
            id (str | None): Optional identifier for the comparison.

        Returns:
            int | None: Preferred response index or None if undecided.
        """
        pass

    @abstractmethod
    def compare_batch(self, pairs: Union[list[dict], Dataset]) -> list[int | None]:
        """Score a batch of response pairs.

        Args:
            pairs (Union[list[dict], Dataset]): Iterable of comparison dictionaries.

        Returns:
            list[int | None]: Preference results for each comparison.
        """
        pass


class ROUGEPreferenceScorer(PreferenceScorer):
    """Preference scorer that compares responses using ROUGE metrics."""

    def __init__(self, config: OmegaConf):
        """Initialize the ROUGE scorer with configuration options.

        Args:
            config (OmegaConf): Configuration specifying the ROUGE variant.
        """
        self.require_ref_flag = True
        self.rouge_type = config.type
        self.scorer = rouge_scorer.RougeScorer([self.rouge_type], use_stemmer=True)

    def require_ref(self):
        """Return whether the scorer requires reference text."""
        return self.require_ref_flag

    def compare(
        self, prompt: str, y1: str, y2: str, ref: str, id: str | None = None
    ) -> int | None:
        """Compare two responses using ROUGE and return the preferred index.

        Args:
            prompt (str): Prompt text associated with the responses.
            y1 (str): First candidate response.
            y2 (str): Second candidate response.
            ref (str): Reference summary against which ROUGE is computed.
            id (str | None): Optional identifier for the comparison.

        Returns:
            int | None: Index of the preferred response (0 or 1).
        """
        s1 = self.scorer.score(ref, y1)[self.rouge_type].fmeasure
        s2 = self.scorer.score(ref, y2)[self.rouge_type].fmeasure

        return 0 if s1 > s2 else 1

    def compare_batch(self, pairs: Union[list[dict], Dataset]) -> list[int | None]:
        """Compare a batch of response pairs using ROUGE scores.

        Args:
            pairs (Union[list[dict], Dataset]): Iterable of comparison dictionaries.

        Returns:
            list[int | None]: List of preferred indices for each pair.
        """
        compared = []
        for pair in pairs:
            compared.append(
                self.compare(pair["prompt"], pair["y1"], pair["y2"], pair["ref"])  # type: ignore
            )  # type: ignore
        return compared


class OpenAIPreferenceScorer(PreferenceScorer):
    """Preference scorer that queries the OpenAI Responses API."""

    def __init__(self, client: OpenAI, config: OmegaConf):
        """Initialize the OpenAI scorer with client and configuration.

        Args:
            client (OpenAI): OpenAI client instance used for API calls.
            config (OmegaConf): Configuration specifying prompts and model details.
        """
        self.require_ref_flag = False
        self.client = client
        self.model_name = config.model
        self.prompt_template = config.prompt
        self.prompt_parse = config.prompt_parse
        self.pattern: str = config.preference_pattern

    def require_ref(self):
        """Return whether the scorer requires reference text."""
        return self.require_ref_flag

    def compare(
        self, prompt: str, y1: str, y2: str, ref: str = "", id: str | None = None
    ) -> int | None:
        """Compare two responses by querying the OpenAI model.

        Args:
            prompt (str): Prompt text presented to the judge.
            y1 (str): First candidate response.
            y2 (str): Second candidate response.
            ref (str): Unused placeholder to match interface signature.
            id (str | None): Optional identifier for the comparison.

        Returns:
            int | None: Index of the preferred response or None if undecided.

        Raises:
            ValueError: If the prompt cannot be parsed into the expected format.
        """
        # Randomize which response is shown first to the judge
        swapped = random.choice([False, True])
        first = y2 if swapped else y1
        second = y1 if swapped else y2
        parsed_prompt = re.match(self.prompt_parse, prompt)
        if not parsed_prompt:
            raise ValueError(
                f"Prompt does not match the expected format: {self.prompt_parse}"
            )
        user_prompt = self.prompt_template.format(
            prompt=parsed_prompt.group(1), generated1=first, generated2=second
        )

        response = self.client.responses.create(
            model=self.model_name,
            input=user_prompt,
            reasoning={"effort": "minimal"},
            max_output_tokens=128,
        )
        output_text = response.output[1].content[0].text  # type: ignore
        assert output_text is not None

        match = re.search(self.pattern, output_text)
        if not match:
            return None

        judged_idx = 0 if match.group(1) == "1" else 1

        return (1 - judged_idx) if swapped else judged_idx

    def compare_batch(self, pairs: Union[list[dict], Dataset]) -> list[int | None]:
        """Compare a batch of responses using sequential OpenAI requests.

        Args:
            pairs (Union[list[dict], Dataset]): Iterable of comparison dictionaries.

        Returns:
            list[int | None]: Preference results for each comparison.
        """
        compared = []
        for pair in tqdm(pairs, desc="Comparing"):
            compared.append(self.compare(pair["prompt"], pair["y1"], pair["y2"]))  # type: ignore
        return compared


class CachedPreferenceScorer(PreferenceScorer):
    """Preference scorer that reads precomputed results from a cache file."""

    def __init__(self, comparison_file: str):
        """Initialize the scorer with a path to cached comparisons.

        Args:
            comparison_file (str): Path to the file containing cached results.
        """
        self.require_ref_flag = False
        self.comparison_file_path = comparison_file
        self._cache: dict[str, int] = {}
        self._load_file()

    def _normalize_id(self, s: str) -> str:
        """Normalize comparison identifiers for consistent cache lookups.

        Args:
            s (str): Raw identifier string.

        Returns:
            str: Normalized identifier.
        """
        parts = [p.strip() for p in s.split(",")]
        return ", ".join(parts)

    def _load_file(self) -> None:
        """Load cached comparison results from disk."""
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
        """Return whether the scorer requires reference text."""
        return self.require_ref_flag

    def compare(
        self, prompt: str, y1: str, y2: str, ref: str, id: str | None = None
    ) -> int | None:
        """Retrieve a cached comparison result.

        Args:
            prompt (str): Prompt text (unused but kept for interface).
            y1 (str): First candidate response.
            y2 (str): Second candidate response.
            ref (str): Reference text (unused).
            id (str | None): Identifier corresponding to the cached result.

        Returns:
            int | None: Cached preference label for the comparison.

        Raises:
            Exception: If the identifier is missing.
            KeyError: If the identifier is not found in the cache.
        """
        if id is None:
            raise Exception(
                "'id' field must be included when calling CachedPreferenceScorer"
            )
        key = self._normalize_id(str(id))
        if key not in self._cache:
            raise KeyError(
                f"Comparison id '{key}' not found in {self.comparison_file_path}"
            )
        return self._cache[key]

    def compare_batch(self, pairs: Union[list[dict], Dataset]) -> list[int | None]:
        """Retrieve cached comparison results for a batch.

        Args:
            pairs (Union[list[dict], Dataset]): Iterable of comparison dictionaries.

        Returns:
            list[int | None]: Cached preference labels for each comparison.
        """
        compared: list[int | None] = []
        for pair in pairs:
            compared.append(
                self.compare(
                    pair["prompt"],  # type: ignore
                    pair["y1"],  # type: ignore
                    pair["y2"],  # type: ignore
                    pair["ref"],  # type: ignore
                    pair["id"],  # type: ignore
                )
            )  # type: ignore
        return compared


class BatchPreferenceScorer(PreferenceScorer):
    """Base class for multi-step batch preference scorers."""

    @abstractmethod
    def require_ref(self) -> bool:
        """Indicate whether the scorer requires reference text.

        Returns:
            bool: True if reference text is required.
        """
        pass

    def compare(
        self, prompt: str, y1: str, y2: str, ref: str, id: str | None = None
    ) -> int | None:
        """Disallow single comparison calls for batch scorers.

        Raises:
            Exception: Always raised to direct callers to batch methods.
        """
        raise Exception(
            "BatchPreferenceScorer doesn't support 'compare'. Try calling compare_batch_* instead."
        )

    def compare_batch(self, pairs: Union[list[dict], Dataset]) -> list[int | None]:
        """Disallow single-call batch comparisons for multi-step scorers.

        Raises:
            Exception: Always raised to direct callers to staged methods.
        """
        raise Exception(
            "BatchPreferenceScorer doesn't support 'compare_batch'. Try calling compare_batch_* instead."
        )

    @abstractmethod
    def compare_batch_0(self, pairs: Union[list[dict], Dataset]) -> list[list[dict]]:
        """Prepare API request payloads from comparison pairs."""
        pass

    @abstractmethod
    def compare_batch_1(
        self, paths: list[str], max_concurrent: int | None = None
    ) -> dict:
        """Submit prepared payloads and monitor batch jobs."""
        pass

    @abstractmethod
    def compare_batch_2(self) -> list[int | None]:
        """Parse batch results into preference labels."""
        pass


class OpenAIBatchPreferenceScorer(BatchPreferenceScorer):
    """Batch preference scorer that interacts with the OpenAI batch API."""

    def __init__(self, client: OpenAI, config: OmegaConf):
        """Initialize the batch scorer with API client and configuration.

        Args:
            client (OpenAI): OpenAI client used to submit batch jobs.
            config (OmegaConf): Configuration describing prompts and batching.
        """
        self.require_ref_flag = False
        self.client = client
        self.model_name = config.model
        self.prompt_template = config.prompt
        self.prompt_parse = config.prompt_parse
        self.pattern = config.preference_pattern
        c = getattr(config, "batch", None)
        self.max_concurrent = getattr(c, "max_concurrent", 3) if c is not None else 3
        self.max_retries = getattr(c, "max_retries", 5) if c is not None else 5
        self.initial_backoff = (
            getattr(c, "initial_backoff", 1.0) if c is not None else 1.0
        )
        self.poll_interval = (
            getattr(c, "poll_interval", 15.0) if c is not None else 15.0
        )
        self.max_request_size = (
            getattr(c, "max_request_per_batch", 30000) if c is not None else 30000
        )

        self.paths = []
        self.batch_files = []
        self.batchs = []
        self.total = 0

        self.pairs = []

        self.swapped: list[bool] = []

    def require_ref(self):
        """Return whether the scorer requires reference text."""
        return self.require_ref_flag

    def compare_batch_0(self, pairs: Union[list[dict], Dataset]) -> list[list[dict]]:
        """Chunk comparison pairs into request payloads for batch submission.

        Args:
            pairs (Union[list[dict], Dataset]): Iterable of comparison dictionaries.

        Returns:
            list[list[dict]]: Nested lists representing batched request bodies.

        Raises:
            ValueError: If a prompt does not match the expected parse pattern.
        """
        requests = []
        self.total = len(pairs)
        for i, pair in enumerate(pairs):
            self.pairs.append(pair)
            if i % self.max_request_size == 0:
                requests.append([])

            prompt = re.match(self.prompt_parse, pair["prompt"])  # type: ignore
            if not prompt:
                raise ValueError(
                    f"Prompt does not match the expected format: {self.prompt_parse}"
                )
            y1 = pair["y1"]  # type: ignore
            y2 = pair["y2"]  # type: ignore

            # Randomize presentation order per pair and record mapping
            swapped = random.choice([False, True])
            self.swapped.append(swapped)
            summary1 = y2 if swapped else y1
            summary2 = y1 if swapped else y2

            user_prompt = self.prompt_template.format(
                prompt=prompt.group(1), generated1=summary1, generated2=summary2
            )
            body = {
                "model": self.model_name,
                "input": user_prompt,
                "reasoning": {"effort": "minimal"},
                "max_output_tokens": 128,
            }
            requests[i // self.max_request_size].append(
                {
                    "custom_id": f"{i} {swapped}",
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": body,
                }
            )

        return requests

    def _retry(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:  # type: ignore
        """Retry helper with exponential backoff for API calls.

        Args:
            fn (Callable[..., T]): Function to execute.
            *args: Positional arguments forwarded to ``fn``.
            **kwargs: Keyword arguments forwarded to ``fn``.

        Returns:
            T: Result returned by the invoked function.

        Raises:
            Exception: Propagates the last exception if retries are exhausted.
        """
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
        """Submit a single batch request file to the OpenAI API.

        Args:
            path (str): Path to the JSONL request file.

        Returns:
            tuple[int, Batch]: Pair of submission count increment and batch metadata.
        """
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
        tqdm.write(
            f"file {path} (file id: {batch_file.id}) submitted (batch id: {batch.id})"
        )
        return (1, batch)

    def compare_batch_1(
        self, paths: list[str], max_concurrent: int | None = None
    ) -> dict:
        """Submit requests and monitor job progress until completion.

        Args:
            paths (list[str]): Paths to JSONL request files.
            max_concurrent (int | None): Maximum simultaneous batch submissions.

        Returns:
            dict: Summary counts of batch job states.
        """
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
                    pbar.set_postfix(
                        {
                            "failed": agg_counts["failed"],
                            "running": len(in_flight),
                            "finished": len(finished),
                        }
                    )

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
        """Parse a single line from the batch output file.

        Args:
            line (str): Raw JSONL line from the batch output.

        Returns:
            tuple[int, int | None]: Pair of index and judged preference.
        """
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
        swapped = True if obj.get("custom_id").split()[1] == "True" else False
        orig_idx = (1 - judged_idx) if swapped else judged_idx
        return idx, orig_idx

    def compare_batch_2(self) -> list[int | None]:
        """Parse completed batch outputs into preference labels.

        Returns:
            list[int | None]: Preference results aligned with original pairs.
        """
        result: list[int | None] = [None for _ in range(len(self.pairs))]

        for b in self.batchs:
            if getattr(b, "status", None) == "completed" and getattr(
                b, "output_file_id", None
            ):
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


def get_preference_scorer(
    config: OmegaConf, openai_client: OpenAI | None
) -> PreferenceScorer:
    """Instantiate a preference scorer based on configuration.

    Args:
        config (OmegaConf): Configuration describing the scorer type.
        openai_client (OpenAI | None): OpenAI client for API-based scorers.

    Returns:
        PreferenceScorer: Configured preference scorer instance.

    Raises:
        Exception: If the scorer type is not recognized or client is missing.
    """
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
    """Return whether the configured scorer requires multi-step batching.

    Args:
        config (OmegaConf): Scorer configuration.

    Returns:
        bool: True if the scorer performs a two-step batch process.
    """
    return (
        config.type.lower() == "openai"
        and getattr(config.openai, "type", "") == "batch"
    )
