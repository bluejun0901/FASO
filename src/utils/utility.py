from datetime import datetime
from collections import deque

def get_filename(*prefixes, suffix=".json"):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    prefix_str = "_".join(prefixes)
    return f"{prefix_str}_{timestamp}{suffix}"

def remove_cycles_kahn(adj_list: list[list[int]]) -> list[list[int]]:
    n = len(adj_list)

    adj = [set(nei) for nei in adj_list]
    indeg = [0] * n
    for u in range(n):
        for v in adj[u]:
            indeg[v] += 1

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

                adj[u].discard(v)
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)

        if not remaining_nodes:
            break

        best_u = None
        best_score = None
        for u in list(remaining_nodes):
            score = (len(adj[u]) - indeg[u])
            if best_score is None or score > best_score:
                best_score = score
                best_u = u
        assert best_u is not None

        if adj[best_u]:
            v = next(iter(adj[best_u]))
            adj[best_u].discard(v)
            indeg[v] -= 1
            removed_for_breaking.add((best_u, v))
            if indeg[v] == 0:
                q.append(v)
        else:
            remaining_nodes.discard(best_u)

    kept = original_edges - removed_for_breaking

    out_adj: list[list[int]] = [[] for _ in range(n)]
    buckets = [[] for _ in range(n)]
    for (u, v) in kept:
        buckets[u].append(v)
    for u in range(n):
        out_adj[u] = buckets[u]

    return out_adj

def remove_cycles_dfs(adj_list: list[list[int]]) -> list[list[int]]:
    n = len(adj_list)
    color = [0] * n
    removed: set[tuple[int, int]] = set()

    def dfs(u: int):
        color[u] = 1
        for v in adj_list[u]:
            if color[v] == 0:
                dfs(v)
            elif color[v] == 1:
                removed.add((u, v))
            else:
                pass
        color[u] = 2

    for i in range(n):
        if color[i] == 0:
            dfs(i)

    original_edges = set((u, v) for u in range(n) for v in adj_list[u])
    kept = original_edges - removed

    out_adj: list[list[int]] = [[] for _ in range(n)]
    buckets = [[] for _ in range(n)]
    for (u, v) in kept:
        buckets[u].append(v)
    for u in range(n):
        out_adj[u] = buckets[u]

    return out_adj

# 완전탐색으로 하기
# 그

def add_transitive_edges(adj_list: list[list[int]]) -> list[list[int]]:
    n = len(adj_list)

    indeg = [0] * n
    for u in range(n):
        for v in adj_list[u]:
            indeg[v] += 1

    q = deque([u for u in range(n) if indeg[u] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in adj_list[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    assert len(topo) == len(adj_list)

    reach = [set() for _ in range(n)]

    for u in reversed(topo):
        for v in adj_list[u]:
            reach[u].add(v)
            reach[u].update(reach[v])

    new_adj = [[] for _ in range(n)]
    for u in range(n):
        new_adj[u] = sorted(reach[u])

    return new_adj
