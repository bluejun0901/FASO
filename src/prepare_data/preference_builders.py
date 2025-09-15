from abc import ABC, abstractmethod
from typing import Callable

from datasets import Dataset
from omegaconf import OmegaConf

from src.prepare_data.preference_scorers import *

from src.utils.utility import *

class PreferenceBuilder(ABC):
    @abstractmethod
    def generate_comparisons(self, dataset: Dataset) -> Dataset:
        pass
    
    @abstractmethod
    def build_with_comparisons(self, comparisons: list[int | None]) -> Dataset:
        pass

def get_cycle_removal_algorithm(name: str) -> Callable:
    name = name.lower()
    if name == "kahn":
        return remove_cycles_kahn
    if name == "dfs":
        return remove_cycles_dfs
    if name == "permutation":
        return remove_cycles_permutation
    if name == "deterministic":
        return remove_cycles_expodential
    raise Exception(f"Unknown cycle removal algorithm: {name}")

# 완전 그래프, 사이클 O
class CyclicPreferenceBuilder(PreferenceBuilder):
    def __init__(self, scorer):
        self.pairs = []
        self.scorer = scorer
        
    def generate_comparisons(self, dataset: Dataset) -> list[dict]:
        self.pairs = []
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

# 사이클 X, 추론 X, DPO
class AcyclicNoReasonPreferenceBuilder(PreferenceBuilder):
    """
    Builds a DAG of preferences (cycle-free). Reasoning OFF.
    """
    def __init__(self, config: OmegaConf, scorer: PreferenceScorer):
        self.scorer = scorer
        self.config = config
        self.example_count = 0
        self.pairs: list[dict] = []

    def generate_comparisons(self, dataset: Dataset) -> list[dict]:
        self.pairs = []
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
        self.groups = [[] for _ in range(self.example_count)]
        for pref, pair in zip(comparisons, self.pairs):
            if pref is None:
                continue
            k = int(pair['meta'].split(",")[0])
            self.groups[k].append((pair, pref))

        result = []
        for group in tqdm(self.groups, desc="Building preferences"):
            if len(group) == 0:
                continue
            prompt = group[0][0]['prompt']
            max_idx = 0

            for pair, pref in group:
                _, i, j = map(int, pair['meta'].split(", "))
                max_idx = max(max_idx, i, j)

            summaries = [""] * (max_idx + 1)
            graph = [[] for _ in range(max_idx + 1)]

            for pair, pref in group:
                _, i, j = map(int, pair['meta'].split(", "))
                y1, y2 = pair['y1'], pair['y2']
                max_idx = max(max_idx, i, j)
                summaries[i] = y1
                summaries[j] = y2
                if pref == 0:
                    graph[i].append(j)
                else:
                    graph[j].append(i)

            algo = getattr(self.config, "cycle_removal", "kahn").lower()
            kept_edges = get_cycle_removal_algorithm(algo)(graph)

            for i, neis in enumerate(kept_edges):
                for nei in neis:
                    chosen, rejected = (summaries[i], summaries[nei])
                    result.append({
                        'prompt': prompt,
                        'chosen': chosen,
                        'rejected': rejected,
                    })

        return Dataset.from_list(result)

# 사이클 X, 추론 O, DPO
class AcyclicReasonPreferenceBuilder(PreferenceBuilder):
    """
    Builds a DAG of preferences (cycle-free). Reasoning ON.
    """
    def __init__(self, 
                 scorer: PreferenceScorer, 
                 config: OmegaConf):
        self.scorer = scorer
        self.config = config
        self.pairs: list[dict] = []

    def generate_comparisons(self, dataset: Dataset) -> list[dict]:
        self.pairs = []
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
        self.groups = [[] for _ in range(self.example_count)]
        for pref, pair in zip(comparisons, self.pairs):
            if pref is None:
                continue
            k = int(pair['meta'].split(",")[0])
            self.groups[k].append((pair, pref))

        result = []
        for group in tqdm(self.groups, desc="Building preferences"):
            if len(group) == 0: 
                continue
            prompt = group[0][0]['prompt']
            max_idx = 0

            for pair, pref in group:
                _, i, j = map(int, pair['meta'].split(", "))
                max_idx = max(max_idx, i, j)

            summaries = [""] * (max_idx + 1)
            graph = [[] for _ in range(max_idx + 1)]

            for pair, pref in group:
                _, i, j = map(int, pair['meta'].split(", "))
                y1, y2 = pair['y1'], pair['y2']
                max_idx = max(max_idx, i, j)
                summaries[i] = y1
                summaries[j] = y2
                if pref == 0:
                    graph[i].append(j)
                else:
                    graph[j].append(i)

            # feedback arc 제거, 추론 (cycle removal)
            algo = getattr(self.config, "cycle_removal", "kahn").lower()
            kept_edges = get_cycle_removal_algorithm(algo)(graph)

            kept_edges = add_transitive_edges(kept_edges)

            for i, neis in enumerate(kept_edges):
                for nei in neis:
                    chosen, rejected = (summaries[i], summaries[nei])
                    result.append({
                        'prompt': prompt,
                        'chosen': chosen,
                        'rejected': rejected,
                    })

        return Dataset.from_list(result)

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
    name = config.type.lower()
    if name == "cyclic":
        return CyclicPreferenceBuilder(scorer)
    if name == "acyclic_no_reason":
        return AcyclicNoReasonPreferenceBuilder(config.acyclic_no_reason, scorer)
    if name == "acyclic_reason":
        return AcyclicReasonPreferenceBuilder(scorer, config.acyclic_reason)
    if name == "cyclic_modified_prob":
        return CyclicModifiedProbPreferenceBuilder(scorer)
    raise Exception(f"Unknown preference builder: {config.type}")
