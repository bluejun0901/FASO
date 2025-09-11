from datasets import Dataset

from utils.utility import *
from prepare_data.preference_scorers import *
from prepare_data.summary_generator import SummaryGenerator

class WinRateCalculator:
    def __init__(self, 
                 ref_generator: SummaryGenerator | None=None,
                 target_generator: SummaryGenerator | None=None):
        self.ref_generator = ref_generator
        self.target_generator = target_generator

    def set_ref_generator(self, ref_generator: SummaryGenerator) -> None:
        self.ref_generator = ref_generator

    def set_target_generator(self, target_genetartor: SummaryGenerator) -> None:
        self.target_generator = target_genetartor

    # win rate for targetgenerator
    def calculate_win_rate(self, prompts: Dataset, scorer: PreferenceScorer) -> tuple[float, list[int | None]]:
        if self.ref_generator is None or self.target_generator is None:
            raise RuntimeError("No generator assigned")
        
        n = len(prompts)
        ref_responses = self.ref_generator.generate_batch(prompts)
        target_responses = self.target_generator.generate_batch(prompts)

        pairs: list[dict] = []
        for i in range(n):
            if len(ref_responses[i]['summaries']) > 0 and len(target_responses[i]['summaries']) > 0:
                pairs.append({
                    'prompt': prompts[i],
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
                