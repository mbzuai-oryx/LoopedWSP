"""Export per-loop states of one test maze for the write-up's first figure."""
import json, numpy as np, torch
from maze import make_dataset, trajectory_w
from wsp import Looped, load_written, maze_tokens
torch.set_num_threads(2); S = slice(7, 15)
it = make_dataset(100, 15, seed=99)[4]; n = 15
T = 2 * it["dS"] + 2
tok = torch.tensor(maze_tokens(it["wall"], it["s"], it["g"]))[None]
traj = torch.tensor(trajectory_w(it, T))
def load(p, attn):
    m = Looped(attn=attn); m.load_state_dict(torch.load(p)); return m
models = {"written": load_written(None, dtype=torch.float32),
          "copy": load("results/B1_tf_nonorm.pt", "softmax"),
          "r5free": load("results/R5_tf_gather_chg.pt", "gather"),
          "r5snap": load("results/R5_tf_gather_chg.pt", "gather")}
loops = sorted(set([0, 4, 10, it["dS"], it["dS"] + it["dS"] // 2, T]))
def code(state):  # per cell: -1 unvisited, 0/1/2 residue, plus path flag
    v = state[:, :3]; vis = v.sum(-1) > 0.5
    r = torch.where(vis, v.argmax(-1), torch.full((n * n,), -1))
    return r.tolist(), (state[:, 7] > 0.5).int().tolist()
out = dict(n=n, wall=it["wall"].ravel().tolist(), s=list(map(int, it["s"])), g=list(map(int, it["g"])), dS=it["dS"], T=T, loops=loops, frames={})
with torch.no_grad():
    for name, m in models.items():
        h0 = m.embed(tok); h = torch.cat([h0[..., :7], traj[0][None], h0[..., 15:]], -1); fr = {}
        for t in range(T + 1):
            if t in loops:
                st = (h[0, :, S] > .5).float() if name != "written" else h[0, :, S].round()
                r, p = code(st); fr[t] = dict(res=r, path=p, wrong=int(((h[0, :, S] > .5).float() != traj[t]).any(-1).sum()))
            if t < T:
                h = m.block(h)
                if name == "r5snap": h = torch.cat([h0[..., :7], h[..., S].round().clamp(0, 1), h0[..., 15:]], -1)
        out["frames"][name] = fr
json.dump(out, open("results/fig_maze.json", "w"))
print({k: {t: v["wrong"] for t, v in f.items()} for k, f in out["frames"].items()}, "dS", it["dS"])
