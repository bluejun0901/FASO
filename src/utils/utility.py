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

# naive O(n! * n) implementation
def remove_cycles_permutation(adj_list: list[list[int]]) -> list[list[int]]:
    from itertools import permutations

    n = len(adj_list)
    if n == 0:
        return []

    edges: list[tuple[int, int]] = []
    for u in range(n):
        for v in adj_list[u]:
            if 0 <= v < n and u != v:
                edges.append((u, v))
    if not edges:
        return [[] for _ in range(n)]
    edges = list(set(edges))

    m = len(edges)

    best_perm = None
    best_count = -1

    inv_pos = [0] * n

    for perm in permutations(range(n)):
        for i, node in enumerate(perm):
            inv_pos[node] = i

        cnt = 0
        for u, v in edges:
            if inv_pos[u] < inv_pos[v]:
                cnt += 1

        if cnt > best_count:
            best_count = cnt
            best_perm = perm
            if best_count == m:
                break

    assert best_perm is not None

    for i, node in enumerate(best_perm):
        inv_pos[node] = i  # reuse array

    out_adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        if inv_pos[u] < inv_pos[v]:
            out_adj[u].append(v)

    return out_adj

def remove_cycles_expodential(adj_list: list[list[int]]) -> list[list[int]]:
    n = len(adj_list)
    if n == 0:
        return []

    # Build, for each v, a bitmask of incoming-edge sources: InMask[v][u]=1 iff u->v exists.
    in_mask = [0] * n
    for u in range(n):
        for v in adj_list[u]:
            if 0 <= v < n and v != u:
                in_mask[v] |= (1 << u)

    full = (1 << n) - 1
    # dp[mask] = max number of forward edges achievable using exactly the vertices in 'mask'
    dp = [-10**18] * (1 << n)
    best_last = [-1] * (1 << n)  # which vertex is placed last for the optimal dp[mask]
    dp[0] = 0

    # Iterate all non-empty masks
    for mask in range(1, full + 1):
        # iterate v as the last vertex in 'mask'
        m = mask
        while m:
            v_bit = m & -m
            v = (v_bit.bit_length() - 1)
            prev = mask ^ v_bit
            # if v is last, edges counted are all from 'prev' into v
            gain = (in_mask[v] & prev).bit_count()
            val = dp[prev] + gain
            if val > dp[mask]:
                dp[mask] = val
                best_last[mask] = v
            m ^= v_bit

    # Reconstruct an ordering (from first to last)
    order_rev = []
    cur = full
    while cur:
        v = best_last[cur]
        if v == -1:
            # fallback (shouldn't happen), pick any set bit
            v = (cur & -cur).bit_length() - 1
        order_rev.append(v)
        cur ^= (1 << v)
    order = order_rev[::-1]  # now from first to last

    # Keep only edges that go forward under this order
    pos = [-1] * n
    for i, v in enumerate(order):
        pos[v] = i

    dag = [[] for _ in range(n)]
    for u in range(n):
        pu = pos[u]
        if pu == -1:
            continue
        for v in adj_list[u]:
            if 0 <= v < n and pu < pos[v]:
                dag[u].append(v)

    return dag

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
