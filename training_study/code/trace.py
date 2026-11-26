"""Trace a free rollout of a trained block: where does it leave the clean state manifold?"""
import sys, json, numpy as np, torch
from maze import make_dataset, trajectory_w
from wsp import Looped, maze_tokens
torch.set_num_threads(2); S = slice(7, 15)
path = sys.argv[1]
args = json.load(open(path.replace(".pt", ".json")))["args"]
m = Looped(norm=args["norm"], attn=args.get("attn", "softmax")); m.load_state_dict(torch.load(path))
items = make_dataset(25, 15, seed=99); T = 2 * max(it["dS"] for it in items) + 4
tok = torch.tensor(np.stack([maze_tokens(it["wall"], it["s"], it["g"]) for it in items]))
traj = torch.tensor(np.stack([trajectory_w(it, T) for it in items])).transpose(0, 1)
rows = []
with torch.no_grad():
    h0 = m.embed(tok); h = h0
    for t in range(1, T + 1):
        h = m.block(h)
        wrong = (h[..., S] > .5).float() != traj[t]
        r = dict(t=t, static_dev=float((h[..., :7] - h0[..., :7]).abs().max()), scratch_dev=float((h[..., 15:] - h0[..., 15:]).abs().max()),
                 search_offgrid=float((h[..., S] - h[..., S].round()).abs().max()),
                 wrong_wave=int(wrong[..., :3].any(-1).sum()), wrong_ptr=int(wrong[..., 3:7].any(-1).sum()), wrong_path=int(wrong[..., 7].sum()),
                 wrong_in_walls=int((wrong.any(-1) & (tok[..., 2] == 0)).sum()))
        rows.append(r)
for r in rows[:6] + rows[9:60:10]: print(r)
json.dump(rows, open(path.replace(".pt", "_trace.json"), "w"))
