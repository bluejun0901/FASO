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
    def __init__(self, config: OmegaConf, scorer: PreferenceScorer):
        self.scorer = scorer
        self.config = config
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
        self.groups = [[] for _ in range(self.example_count)]
        for pref, pair in zip(comparisons, self.pairs):
            if pref is None:
                continue
            k = int(pair['meta'].split(",")[0])
            self.groups[k].append((pair, pref))

        result = []
        for group in self.groups:
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

            # feedback arc 제거 (cycle removal)
            algo = getattr(self.config, "cycle_removal", "kahn").lower()

            def remove_cycles_kahn(adj_list: list[list[int]]) -> set[tuple[int, int]]:
                n = len(adj_list)
                # Build indegree and adjacency as sets for O(1) removals
                adj = [set(nei) for nei in adj_list]
                indeg = [0] * n
                for u in range(n):
                    for v in adj[u]:
                        indeg[v] += 1

                from collections import deque
                q = deque([i for i in range(n) if indeg[i] == 0])

                original_edges = set((u, v) for u in range(n) for v in adj[u])
                removed_for_breaking: set[tuple[int, int]] = set()

                remaining_nodes = set(range(n))

                while remaining_nodes:
                    while q:
                        u = q.popleft()
                        if u not in remaining_nodes:
                            continue
                        remaining_nodes.discard(u)
                        for v in list(adj[u]):
                            # normal processing of kept edge
                            adj[u].discard(v)
                            indeg[v] -= 1
                            if indeg[v] == 0:
                                q.append(v)

                    if not remaining_nodes:
                        break

                    # Cycle detected: pick a node to break cycles.
                    # Heuristic: node with maximum outdegree - indegree
                    best_u = None
                    best_score = None
                    for u in list(remaining_nodes):
                        score = (len(adj[u]) - indeg[u])
                        if best_score is None or score > best_score:
                            best_score = score
                            best_u = u
                    assert best_u is not None
                    # Remove one outgoing edge from best_u (pick arbitrary)
                    if adj[best_u]:
                        v = next(iter(adj[best_u]))
                        adj[best_u].discard(v)
                        indeg[v] -= 1
                        removed_for_breaking.add((best_u, v))
                        if indeg[v] == 0:
                            q.append(v)
                    else:
                        # No outgoing edge (shouldn't happen often); remove the node
                        remaining_nodes.discard(best_u)

                kept = original_edges - removed_for_breaking
                return kept

            def remove_cycles_dfs(adj_list: list[list[int]]) -> set[tuple[int, int]]:
                n = len(adj_list)
                color = [0] * n  # 0=unvisited,1=visiting,2=done
                removed: set[tuple[int, int]] = set()

                def dfs(u: int):
                    color[u] = 1
                    for v in adj_list[u]:
                        if color[v] == 0:
                            dfs(v)
                        elif color[v] == 1:
                            # back edge u->v forms a cycle; remove it
                            removed.add((u, v))
                        else:
                            # color[v] == 2 -> forward/cross edge, ok
                            pass
                    color[u] = 2

                for i in range(n):
                    if color[i] == 0:
                        dfs(i)

                original_edges = set((u, v) for u in range(n) for v in adj_list[u])
                kept = original_edges - removed
                return kept

            if algo == "dfs":
                kept_edges = remove_cycles_dfs(graph)
            else:
                kept_edges = remove_cycles_kahn(graph)

            # Convert kept edges to (prompt, chosen, rejected)
            for (i, j) in kept_edges:
                yi = summaries[i]
                yj = summaries[j]
                if not yi or not yj:
                    continue
                result.append({
                    'prompt': prompt,
                    'chosen': yi,
                    'rejected': yj,
                })

        return Dataset.from_list(result)

# 사이클 X, 추론 O, DPO
class AcyclicReasonPreferenceBuilder(PreferenceBuilder):
    """
    Builds a DAG of preferences (cycle-free). Reasoning ON.
    """
    def __init__(self, scorer):
        self.scorer = scorer
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
        self.groups = [[] for _ in range(self.example_count)]
        for pref, pair in zip(comparisons, self.pairs):
            if pref is None:
                continue
            k = int(pair['meta'].split(",")[0])
            self.groups[k].append((pair, pref))

        result = []
        for group in self.groups:
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
            raise NotImplementedError("you have to implement this")

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
    if name == "rank":
        return RankPreferenceBuilder(scorer)
    if name == "acyclic_no_reason":
        return AcyclicNoReasonPreferenceBuilder(config, scorer)
    if name == "acyclic_reason":
        return AcyclicReasonPreferenceBuilder(scorer)
    if name == "cyclic_modified_prob":
        return CyclicModifiedProbPreferenceBuilder(scorer)
    raise Exception(f"Unknown preference builder: {config.type}")
