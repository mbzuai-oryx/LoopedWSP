"""Can the written loop recover from ONE wrong bit, or from small continuous noise on its state?"""
import json
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
M = load_written(None)
out = {}
def run_from(h, t0):
    with torch.no_grad():
        for _ in range(t0, T): h = M.block(h)
        return (M.readout(h).argmax(-1) == ch).all(-1)
with torch.no_grad():
    h = M.embed(tok); t0 = 6
    for _ in range(t0): h = M.block(h)
    rng = np.random.default_rng(0)
    for kind in ["residue", "pointer", "path_on_open_offpath_cell"]:
        hh = h.clone()
        for b in range(len(items)):
            if kind == "path_on_open_offpath_cell":
                cand = ((tok[b, :, 2] == 1) & (traj[T, b, :, 7] < .5)).nonzero().flatten()
            else:
                cand = ((traj[t0, b, :, :3].sum(-1) > 0) & (tok[b, :, 2] != 3)).nonzero().flatten()
            cell = cand[rng.integers(len(cand))].item()
            if kind == "residue":
                k = int(traj[t0, b, cell, :3].argmax()); hh[b, cell, 7:10] = 0; hh[b, cell, 7 + (k + 1) % 3] = 1
            elif kind == "pointer":
                cur = int(traj[t0, b, cell, 3:7].argmax()) if traj[t0, b, cell, 3:7].sum() > 0 else -1
                new = (cur + 1 + rng.integers(3)) % 4
                hh[b, cell, 10:14] = 0; hh[b, cell, 10 + new] = 1
            else:
                hh[b, cell, 14] = 1
        ok = run_from(hh, t0)
        out[kind] = float(ok.double().mean()); print("one wrong", kind, "bit at loop", t0, "-> solved", out[kind], flush=True)
    for sg in [0.01, 0.05, 0.1, 0.2, 0.3]:
        torch.manual_seed(1); hh = h.clone(); hh[..., S] += sg * torch.randn_like(hh[..., S])
        out[f"noise_{sg}"] = float(run_from(hh, t0).double().mean()); print("state noise", sg, out[f"noise_{sg}"], flush=True)
    # does the loop pull noisy values back to 0/1? max deviation from {0,1} one loop after noise
    hh = h.clone(); hh[..., S] += 0.1 * torch.randn_like(hh[..., S]); h1 = M.block(hh)
    dev0 = (hh[..., S] - hh[..., S].round()).abs().max().item(); dev1 = (h1[..., S] - h1[..., S].round()).abs().max().item()
    out["snap_dev_before_after"] = [dev0, dev1]; print("deviation from 0/1 before/after one loop:", dev0, dev1)
json.dump(out, open("results/diag3.json", "w"), indent=1)
