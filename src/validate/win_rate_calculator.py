from datasets import Dataset

from src.utils.utility import *
from src.prepare_data.preference_scorers import *
from src.prepare_data.summary_generator import *

class WinRateCalculator:
    def __init__(self, 
                 ref_generator: SummaryGenerator | None=None,
                 target_generator: SummaryGenerator | None=None):
        self.ref_generator = ref_generator
        self.target_generator = target_generator
        self.dataset: Dataset | None = None
        self.ref_responses: Dataset | None = None
        self.target_responses: Dataset | None = None

    def set_ref_generator(self, ref_generator: SummaryGenerator) -> None:
        self.ref_generator = ref_generator

    def set_target_generator(self, target_genetartor: SummaryGenerator) -> None:
        self.target_generator = target_genetartor

    def set_dataset(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def ref_generate(self) -> None:
        if self.ref_generator is None:
            raise RuntimeError("No reference generator assigned")
        if self.dataset is None:
            raise RuntimeError("No dataset assigned")
        self.ref_responses = self.ref_generator.generate_batch(self.dataset)

    def target_generate(self) -> None:
        if self.target_generator is None:
            raise RuntimeError("No target generator assigned")
        if self.dataset is None:
            raise RuntimeError("No dataset assigned")
        self.target_responses = self.target_generator.generate_batch(self.dataset)

    # win rate for targetgenerator
    def calculate_win_rate(self, prompts: Dataset, scorer: PreferenceScorer) -> tuple[float, list[int | None]]:
        if self.ref_responses is None or self.target_responses is None:
            raise RuntimeError("No responses generated")
        if len(self.ref_responses) != len(self.target_responses):
            raise RuntimeError("Response size mismatch")
        
        n = len(prompts)
        ref_responses = self.ref_responses
        target_responses = self.target_responses

        pairs: list[dict] = []
        for i in range(n):
            if len(ref_responses[i]['summaries']) > 0 and len(target_responses[i]['summaries']) > 0:
                pairs.append({
                    'prompt': prompts[i]['article'],
                    'y1': ref_responses[i]['summaries'][0],
                    'y2': target_responses[i]['summaries'][0],
                    'ref': prompts[i]['reference'] if scorer.require_ref() else "",
                    'meta': f"{i}, {i}, {i}"
                })

        res = scorer.compare_batch(pairs)

        total = 0
        win = 0
        for r in res:
            if r is not None:
                total += 1
            if r == 1:
                win += 1

        return (win / total, res)
                