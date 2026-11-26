"""Where is the written solution fragile, how sharp must attention be, can it recover from one error?"""
import json, math
import numpy as np, torch
from maze import make_dataset, trajectory_w
from wsp import load_written, maze_tokens

torch.set_num_threads(2)
S = slice(7, 15)
items = make_dataset(40, 15, seed=99)
T = 2 * max(i["dS"] for i in items) + 4
tok = torch.tensor(np.stack([maze_tokens(i["wall"], i["s"], i["g"]) for i in items]))
traj = torch.tensor(np.stack([trajectory_w(i, T) for i in items]), dtype=torch.float64).transpose(0, 1)
ch = tok[..., 2].clone(); ch[(traj[T, ..., 7] > 0.5) & (ch == 1)] = 4
out = {}


def solve(M, h=None, start=0):
    with torch.no_grad():
        h = M.embed(tok) if h is None else h
        for _ in range(start, T): h = M.block(h)
        return float((M.readout(h).argmax(-1) == ch).all(-1).double().mean())


groups = {"position embedding (E_row, E_col)": ["E_row", "E_col"], "char embedding": ["E_char"],
          "attention Wq": ["blocks.0.Wq"], "attention Wk, Wv, Wo": ["blocks.0.Wk", "blocks.0.Wv", "blocks.0.Wo"],
          "MLP": ["blocks.0.W1", "blocks.0.W2"], "readout": ["out."]}
res = {}
for g, prefixes in groups.items():
    res[g] = {}
    for eps in [1e-3, 1e-2, 1e-1]:
        r = []
        for seed in range(3):
            torch.manual_seed(seed); M = load_written(None)
            with torch.no_grad():
                for n, p in M.named_parameters():
                    if any(n.startswith(px) for px in prefixes): p.mul_(1 + eps * torch.randn_like(p))
            r.append(solve(M))
        res[g][eps] = float(np.mean(r))
    print(g, res[g], flush=True)
out["basin_by_group"] = res

# attention temperature: scale Wq by c -> logit gap 100c
temp = {}
for c in [1, 0.3, 0.1, 0.07, 0.05, 0.03, 0.02, 0.01]:
    M = load_written(None)
    with torch.no_grad(): M.blocks[0].Wq.weight.mul_(c)
    temp[c] = solve(M); print("Wq scale", c, "gap", 100 * c, "solve", temp[c], flush=True)
out["attn_gap"] = {str(100 * c): v for c, v in temp.items()}

# one discrete error: at loop t0, flip one bit of the search state of one visited open cell
M = load_written(None)
flip_res = []
torch.manual_seed(0)
with torch.no_grad():
    h = M.embed(tok)
    t0 = 6
    for _ in range(t0): h = M.block(h)
    for kind in ["residue", "pointer", "path"]:
        ok = 0
        for b in range(len(items)):
            hh = h.clone()
            vis = (traj[t0, b, :, :3].sum(-1) > 0).nonzero().flatten()
            cell = vis[torch.randint(len(vis), (1,))].item()
            if kind == "residue":
                k = int(traj[t0, b, cell, :3].argmax()); hh[b, cell, 7 + k] = 0; hh[b, cell, 7 + (k + 1) % 3] = 1
            elif kind == "pointer":
                hh[b, cell, 10:14] = 0; hh[b, cell, 10 + np.random.randint(4)] = 1
            else:
                hh[b, cell, 14] = 1 - hh[b, cell, 14]
            for _ in range(t0, T): hh = M.block(hh)
            ok += bool((M.readout(hh).argmax(-1)[b] == ch[b]).all())
        flip_res.append((kind, ok / len(items)))
        print("one flipped", kind, "bit at loop", t0, "-> solved", ok / len(items), flush=True)
out["one_flip"] = flip_res

# continuous noise on the state at loop t0 (all cells)
noise = {}
with torch.no_grad():
    for sg in [0.01, 0.05, 0.1, 0.2, 0.3]:
        torch.manual_seed(1)
        hh = h.clone(); hh[..., S] += sg * torch.randn_like(hh[..., S])
        noise[sg] = solve(M, hh, start=t0)
        print("state noise", sg, "solve", noise[sg], flush=True)
out["state_noise"] = noise
json.dump(out, open("results/diag2.json", "w"), indent=1)
