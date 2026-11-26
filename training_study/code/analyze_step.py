"""One-step analysis of a trained model: copy baseline, accuracy on cells that change, attention pattern."""
import sys, json, math
import numpy as np, torch
from maze import make_dataset, trajectory_w, DIRS
from wsp import Looped, load_written, maze_tokens

torch.set_num_threads(2)
S = slice(7, 15)
items = make_dataset(50, 15, seed=99)
T = 2 * max(i["dS"] for i in items) + 4
tok = torch.tensor(np.stack([maze_tokens(i["wall"], i["s"], i["g"]) for i in items]))
traj = torch.tensor(np.stack([trajectory_w(i, T) for i in items])).transpose(0, 1)
n = 15


def stats(model, norm):
    res = dict(all=[0, 0], changed=[0, 0], copy_all=[0, 0])
    attn_on_nbr = []
    with torch.no_grad():
        h0 = model.embed(tok)
        for t in range(T):
            hin = torch.cat([h0[..., :7], traj[t].to(h0.dtype), h0[..., 15:]], -1)
            h1, A = model.block(hin, return_attn=True)
            good = ((h1[..., S] > 0.5).float() == traj[t + 1]).all(-1)
            ch = (traj[t + 1] != traj[t]).any(-1)
            cp = (traj[t] == traj[t + 1]).all(-1)
            res["all"][0] += good.sum().item(); res["all"][1] += good.numel()
            res["changed"][0] += (good & ch).sum().item(); res["changed"][1] += ch.sum().item()
            res["copy_all"][0] += cp.sum().item(); res["copy_all"][1] += cp.numel()
            if t == 5:
                # for each head, attention mass on each of the 4 neighbours (interior cells)
                r = torch.arange(n * n) // n; c = torch.arange(n * n) % n
                inner = ((r > 0) & (r < n - 1) & (c > 0) & (c < n - 1)).nonzero().flatten()
                for hd in range(A.shape[1]):
                    row = []
                    for dr, dc in DIRS + [(0, 0)]:
                        j = (r[inner] + dr) * n + (c[inner] + dc)
                        row.append(float(A[:, hd, inner, j].mean()))
                    attn_on_nbr.append(row)
    return {k: v[0] / v[1] for k, v in res.items()}, attn_on_nbr


out = {}
for path in sys.argv[1:]:
    if path == "written":
        m = load_written(None); name = "written"
    else:
        d = json.load(open(path.replace(".pt", ".json")))["args"]
        m = Looped(norm=d["norm"], attn=d.get("attn", "softmax")); m.load_state_dict(torch.load(path)); name = path
    s, att = stats(m, False)
    out[name] = dict(step=s, attn_dirs_DLRU_self=att)
    print(name, {k: round(v, 4) for k, v in s.items()})
    for hd, row in enumerate(att): print("  head", hd, "mass on D,L,R,U,self:", [round(x, 3) for x in row])
json.dump(out, open("results/analyze_step.json", "w"), indent=1)
