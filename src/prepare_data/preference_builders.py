from abc import ABC, abstractmethod

from datasets import Dataset
from omegaconf import OmegaConf

from preference_scorers import *

class PreferenceBuilder(ABC):
    @abstractmethod
    def generate_comparisons(self, dataset: Dataset) -> Dataset:
        pass
    
    @abstractmethod
    def build_with_comparisons(self, comparisons: list[int | None]) -> Dataset:
        pass
    
class PairwisePreferenceBuilder(PreferenceBuilder):
    def __init__(self, scorer):
        self.pairs = []
        self.scorer = scorer
        
    def generate_comparisons(self, dataset: Dataset) -> list[dict]:
        for example in dataset:
            prompt = example['prompt'] # type: ignore
            ref = example['reference'] if self.scorer.require_ref() else "" # type: ignore
            summaries = example['summaries'] # type: ignore
            
            for i, y1 in enumerate(summaries):
                for j, y2 in enumerate(summaries):
                    if i < j:
                        self.pairs.append({
                            'prompt': prompt,
                            'y1': y1,
                            'y2': y2,
                            'ref': ref
                        })
        
        return self.pairs
    
    def build_with_comparisons(self, comparisons: list[int | None]) -> Dataset:
        result = []
        for pref, pair in zip(comparisons, self.pairs):
            if pref is None:
                continue
            chosen, rejected = (pair['y1'], pair['y2']) if pref == 0 else (pair['y2'], pair['y1'])
            result.append({
                'prompt': pair['prompt'],
                'chosen': chosen,
                'rejected': rejected,
            })
        return Dataset.from_list(result)

def get_preference_builder(config: OmegaConf, scorer: PreferenceScorer) -> PreferenceBuilder:
    if config.builder.lower() == "pairwise":
        return PairwisePreferenceBuilder(scorer)
    raise Exception("Unknown preference builder")