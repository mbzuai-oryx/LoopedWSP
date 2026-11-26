"""Maze generation + ground-truth loop-by-loop states for the looped BFS block.

State layout per cell (35 numbers), mirroring the LoopedWSP maze transformer:
  0 bias(1) 1 r 2 c 3 r^2+c^2 4 wall 5 start 6 goal            (what & where, 7)
  7 v0 8 v1 9 v2  10 pD 11 pL 12 pR 13 pU  14 path             (search state, 8)
  15..34 scratch: for dir in D,L,R,U: nb v0,v1,v2,path,points-to-me (20)
"""
import numpy as np
from collections import deque

DIRS = [(1, 0), (0, -1), (0, 1), (-1, 0)]  # down, left, right, up (tie-break order)
OPP = [3, 2, 1, 0]                          # index of the opposite direction
D = 35
S_V, S_P, S_PATH = 7, 10, 14                # offsets
SEARCH = slice(7, 15)
SCRATCH = slice(15, 35)


def gen_maze(n, rng, extra=0.08, min_dist=None):
    """n odd grid size (border = walls). DFS perfect maze, then knock out extra walls
    so several equally short paths exist (like Maze-Hard)."""
    m = (n - 1) // 2
    wall = np.ones((n, n), dtype=np.int8)
    seen = np.zeros((m, m), bool)
    stack = [(rng.integers(m), rng.integers(m))]
    seen[stack[0]] = True
    wall[2 * stack[0][0] + 1, 2 * stack[0][1] + 1] = 0
    while stack:
        a, b = stack[-1]
        nbrs = [(a + da, b + db) for da, db in DIRS
                if 0 <= a + da < m and 0 <= b + db < m and not seen[a + da, b + db]]
        if not nbrs:
            stack.pop(); continue
        na, nb = nbrs[rng.integers(len(nbrs))]
        seen[na, nb] = True
        wall[2 * na + 1, 2 * nb + 1] = 0
        wall[a + na + 1, b + nb + 1] = 0
        stack.append((na, nb))
    # extra openings -> cycles / multiple shortest paths
    for r in range(1, n - 1):
        for c in range(1, n - 1):
            if wall[r, c] and (r + c) % 2 == 1 and rng.random() < extra:
                wall[r, c] = 0
    opens = np.argwhere(wall == 0)
    if min_dist is None:
        min_dist = n  # demand reasonably long paths
    for _ in range(200):
        g = tuple(opens[rng.integers(len(opens))])
        dist = bfs(wall, g)
        cand = np.argwhere(dist >= min_dist)
        if len(cand):
            s = tuple(cand[rng.integers(len(cand))])
            return wall, s, g, dist
        min_dist = max(2, min_dist - 2)
    raise RuntimeError


def bfs(wall, g):
    n = wall.shape[0]
    dist = np.full(wall.shape, -1, int)
    dist[g] = 0
    q = deque([g])
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < n and 0 <= cc < n and not wall[rr, cc] and dist[rr, cc] < 0:
                dist[rr, cc] = dist[r, c] + 1
                q.append((rr, cc))
    return dist


def solve(wall, s, g, dist):
    """pointer dir per cell (-1 none), path index from S (-1 off path)."""
    n = wall.shape[0]
    ptr = np.full(wall.shape, -1, int)
    for r in range(n):
        for c in range(n):
            if dist[r, c] >= 1:
                for k, (dr, dc) in enumerate(DIRS):
                    if dist[r + dr, c + dc] == dist[r, c] - 1:
                        ptr[r, c] = k; break
    pidx = np.full(wall.shape, -1, int)
    cur, i = s, 0
    while True:
        pidx[cur] = i
        if cur == g: break
        dr, dc = DIRS[ptr[cur]]
        cur = (cur[0] + dr, cur[1] + dc); i += 1
    return ptr, pidx


def embed(wall, s, g):
    n = wall.shape[0]
    x = np.zeros((n, n, D), np.float32)
    r, c = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    x[..., 0] = 1; x[..., 1] = r; x[..., 2] = c; x[..., 3] = r * r + c * c
    x[..., 4] = wall
    x[s + (5,)] = 1; x[g + (6,)] = 1
    x[g + (S_V + 0,)] = 1     # goal: distance 0
    x[s + (S_PATH,)] = 1      # start: on path
    return x.reshape(n * n, D)


def trajectory(dist, ptr, pidx, T):
    """targets[t] = 8 search numbers per cell after t loops, t = 0..T."""
    n = dist.shape[0]
    dS = pidx.max()
    out = np.zeros((T + 1, n, n, 8), np.float32)
    for t in range(T + 1):
        vis = (dist >= 0) & (dist <= t)
        for k in range(3):
            out[t, ..., k] = vis & (dist % 3 == k)
        pv = vis & (dist >= 1)
        for k in range(4):
            out[t, ..., 3 + k] = pv & (ptr == k)
        out[t, ..., 7] = (pidx == 0) | ((pidx >= 1) & (t >= dS + pidx))
    return out.reshape(T + 1, n * n, 8)


def make_dataset(num, n, seed, extra=0.08):
    rng = np.random.default_rng(seed)
    items = []
    for _ in range(num):
        wall, s, g, dist = gen_maze(n, rng, extra)
        ptr, pidx = solve(wall, s, g, dist)
        items.append(dict(wall=wall, s=s, g=g, dist=dist, ptr=ptr, pidx=pidx,
                          dS=int(dist[s]), x0=embed(wall, s, g)))
    return items


def trajectory_w(it, T):
    """Search state (dims 7..14) of the hand-written LoopedWSP block after t loops, t=0..T.
    Wave at loop t = dist<=t; pointers and path lag the wave by one loop."""
    a = trajectory(it["dist"], it["ptr"], it["pidx"], T)
    out = a.copy()
    out[1:, :, 3:] = a[:-1, :, 3:]
    out[0, :, 3:7] = 0
    out[0, :, 7] = 0; out[0, :, 7][np.ravel_multi_index(it["s"], it["dist"].shape)] = 1
    return out
