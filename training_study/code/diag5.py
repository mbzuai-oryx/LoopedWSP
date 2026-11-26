"""Uniform attention (scores = 0) but the WRITTEN value/output/MLP weights downstream.
Per head: mean -dL/ds_ij by the key's offset. Does each head's gradient point at its own neighbour?"""
import json, math
import numpy as np, torch, torch.nn.functional as F
from maze import make_dataset, trajectory_w
from wsp import load_written, maze_tokens
torch.set_num_threads(2)
S = slice(7, 15)
items = make_dataset(64, 15, seed=7); rng = np.random.default_rng(0)
m = load_written(None); b = m.blocks[0]
acc = {}
for it in items:
    T = 2 * it["dS"] + 2
    tok = torch.tensor(maze_tokens(it["wall"], it["s"], it["g"]))[None]
    traj = torch.tensor(trajectory_w(it, T), dtype=torch.float64)
    t = int(rng.integers(1, it["dS"]))
    h0 = m.embed(tok); x = torch.cat([h0[..., :7], traj[t][None], h0[..., 15:]], -1)
    N = x.shape[1]
    s = torch.zeros(1, 4, N, N, dtype=torch.float64, requires_grad=True)
    v = b.Wv(x).view(1, N, 4, 8).transpose(1, 2)
    o = (torch.softmax(s, -1) @ v).transpose(1, 2).reshape(1, N, 32)
    h = x + b.Wo(o); h = h + b.W2(F.relu(b.W1(h)))
    F.mse_loss(h[..., S], traj[t + 1][None]).backward()
    g = -s.grad[0]
    rc = tok[0, :, :2]; dr = rc[None, :, 0] - rc[:, None, 0]; dc = rc[None, :, 1] - rc[:, None, 1]
    chg = (traj[t + 1] != traj[t]).any(-1)
    if not chg.any(): continue
    for hd in range(4):
        G = g[hd][chg]
        for o_ in [(1, 0), (0, -1), (0, 1), (-1, 0), (0, 0), (2, 0), (1, 1)]:
            msk = (dr[chg] == o_[0]) & (dc[chg] == o_[1])
            if msk.any(): acc.setdefault((hd, o_), []).append(float(G[msk].mean()))
        msk = (dr[chg].abs() + dc[chg].abs()) > 2
        acc.setdefault((hd, "far"), []).append(float(G[msk].mean()))
names = {(1, 0): "D", (0, -1): "L", (0, 1): "R", (-1, 0): "U", (0, 0): "self", (2, 0): "2D", (1, 1): "DR", "far": "far"}
out = {}
for hd in range(4):
    row = {names[o_]: float(np.mean(acc[(hd, o_)])) for o_ in names}
    out[f"head{hd}"] = row
    print(f"head {hd}:", "  ".join(f"{k} {v*1e3:+.3f}" for k, v in row.items()), "(x1e-3)")
json.dump(out, open("results/diag5.json", "w"), indent=1)
