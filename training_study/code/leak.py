"""Attention mass that leaks beyond the 4 neighbours + self, by maze size, for a relative-bias model."""
import sys, json, numpy as np, torch
from maze import make_dataset, trajectory_w
from wsp import Looped, maze_tokens
torch.set_num_threads(2)
args = json.load(open(sys.argv[1].replace(".pt", ".json")))["args"]
m = Looped(attn="relbias", local=args.get("local", False)); m.load_state_dict(torch.load(sys.argv[1]))
out = {}
for n in [15, 21, 31]:
    it = make_dataset(4, n, seed=97)
    tok = torch.tensor(np.stack([maze_tokens(i["wall"], i["s"], i["g"]) for i in it]))
    T = 2 * max(i["dS"] for i in it) + 4
    traj = torch.tensor(np.stack([trajectory_w(i, T) for i in it])).transpose(0, 1)
    with torch.no_grad():
        h0 = m.embed(tok); t = min(10, T)
        _, A = m.block(torch.cat([h0[..., :7], traj[t], h0[..., 15:]], -1), return_attn=True)
    rc = tok[0, :, :2]; dr = rc[None, :, 0] - rc[:, None, 0]; dc = rc[None, :, 1] - rc[:, None, 1]
    near = ((dr.abs() + dc.abs()) <= 1).float()
    far = 1 - (A * near).sum(-1)               # (B, H, N)
    out[n] = [round(float(far[:, h].mean()), 3) for h in range(4)]
    print(f"{n}x{n}: mass outside the 4 neighbours + self, per head:", out[n])
json.dump(out, open("results/leak_E2.json", "w"))
