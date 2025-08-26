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
    
# 완전 그래프, 사이클 O
class CyclicPreferenceBuilder(PreferenceBuilder):
    def __init__(self, scorer):
        self.pairs = []
        self.scorer = scorer
        
    def generate_comparisons(self, dataset: Dataset) -> list[dict]:
        for k, example in enumerate(dataset):
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
                            'ref': ref,
                            'meta': f"{k}, {i}, {j}"
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

# 순위 DPO [선행연구]
class RankPreferenceBuilder(PreferenceBuilder):
    """
    Builds pairwise data per Ranking-DPO prior work.
    """
    def __init__(self, scorer):
        self.scorer = scorer
        self.pairs: list[dict] = []

    def generate_comparisons(self, dataset: Dataset) -> list[dict]:
        """
        TODO: Implement ranking-based pair creation for DPO baseline.
        """
        raise NotImplementedError("RankDPOPreferenceBuilder.generate_comparisons is not implemented yet.")

    def build_with_comparisons(self, comparisons: list[int | None]) -> Dataset:
        """
        TODO: Map comparison results to DPO (prompt, chosen, rejected).
        """
        raise NotImplementedError("RankDPOPreferenceBuilder.build_with_comparisons is not implemented yet.")


# 사이클 X, 추론 X, DPO
class AcyclicNoReasonPreferenceBuilder(PreferenceBuilder):
    """
    Builds a DAG of preferences (cycle-free). Reasoning OFF.
    """
    def __init__(self, scorer):
        self.scorer = scorer
        self.example_count = 0
        self.pairs: list[dict] = []

    def generate_comparisons(self, dataset: Dataset) -> list[dict]:
        self.example_count = len(dataset)
        for k, example in enumerate(dataset):
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
                            'ref': ref,
                            'meta': f"{k}, {i}, {j}"
                        })
        
        return self.pairs

    def build_with_comparisons(self, comparisons: list[int | None]) -> Dataset:
        self.edges = [[] for i in range(self.example_count)]
        raise NotImplementedError("AcyclicNoReasonPreferenceBuilder.build_with_comparisons is not implemented yet.")


# 사이클 X, 추론 O, DPO
class AcyclicReasonPreferenceBuilder(PreferenceBuilder):
    """
    Builds a DAG of preferences (cycle-free). Reasoning ON.
    """
    def __init__(self, scorer):
        self.scorer = scorer
        self.pairs: list[dict] = []

    def generate_comparisons(self, dataset: Dataset) -> list[dict]:
        """
        TODO: Same as acyclic no-reason, but plan to attach reasoning-specific fields in 'meta' if needed.
        """
        raise NotImplementedError("AcyclicReasonPreferenceBuilder.generate_comparisons is not implemented yet.")

    def build_with_comparisons(self, comparisons: list[int | None]) -> Dataset:
        """
        TODO: Convert to (prompt, chosen, rejected) and optionally carry reasoning annotations.
        """
        raise NotImplementedError("AcyclicReasonPreferenceBuilder.build_with_comparisons is not implemented yet.")


# 사이클 O, DPO (확률 수식 변형)
class CyclicModifiedProbPreferenceBuilder(PreferenceBuilder):
    """
    Same topology as Cyclic, but intended for a modified probability formula during training.
    This builder may need to tag 'meta' or store extra fields for the trainer to pick up.
    """
    def __init__(self, scorer):
        self.scorer = scorer
        self.pairs: list[dict] = []

    def generate_comparisons(self, dataset: Dataset) -> list[dict]:
        """
        TODO: You can begin with the same full K-combination-of-2 as CyclicPreferenceBuilder,
        and inject a 'variant' flag in meta (e.g., 'k, i, j|modprob') if needed.
        """
        raise NotImplementedError("CyclicModifiedProbPreferenceBuilder.generate_comparisons is not implemented yet.")

    def build_with_comparisons(self, comparisons: list[int | None]) -> Dataset:
        """
        TODO: Standard (prompt, chosen, rejected) conversion; trainer will handle modified prob formula.
        """
        raise NotImplementedError("CyclicModifiedProbPreferenceBuilder.build_with_comparisons is not implemented yet.")

def get_preference_builder(config: OmegaConf, scorer: PreferenceScorer) -> PreferenceBuilder:
    name = config.builder.lower()
    if name == "cyclic":
        return CyclicPreferenceBuilder(scorer)
    if name == "rank":
        return RankPreferenceBuilder(scorer)
    if name == "acyclic_no_reason":
        return AcyclicNoReasonPreferenceBuilder(scorer)
    if name == "acyclic_reason":
        return AcyclicReasonPreferenceBuilder(scorer)
    if name == "cyclic_modified_prob":
        return CyclicModifiedProbPreferenceBuilder(scorer)
    raise Exception(f"Unknown preference builder: {config.builder}")