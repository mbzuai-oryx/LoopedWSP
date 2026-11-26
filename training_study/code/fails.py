"""Which held-out mazes does a train_disc model fail, and what is its first wrong bit in each?"""
import sys, json, numpy as np, torch
from maze import make_dataset, trajectory_w
from wsp import Looped, maze_tokens
torch.set_num_threads(2); S = slice(7, 15)
names = ["v0", "v1", "v2", "pD", "pL", "pR", "pU", "path"]
out = {}
for path in sys.argv[1:]:
    args = json.load(open(path.replace(".pt", ".json")))["args"]
    m = Looped(attn=args["attn"], local=args.get("local", False)); m.load_state_dict(torch.load(path))
    items = make_dataset(100, 15, seed=99); failed = []
    with torch.no_grad():
        for idx, it in enumerate(items):
            T = 2 * it["dS"] + 4
            tok = torch.tensor(maze_tokens(it["wall"], it["s"], it["g"]))[None]
            traj = torch.tensor(trajectory_w(it, T))[:, None]
            h0 = m.embed(tok); h = torch.cat([h0[..., :7], h0[..., S].round().clamp(0, 1), h0[..., 15:]], -1)
            first = None
            for t in range(1, T + 1):
                h = torch.cat([h0[..., :7], m.block(h)[..., S].round().clamp(0, 1), h0[..., 15:]], -1)
                wrong = (h[0, :, S] != traj[t, 0])
                if first is None and wrong.any():
                    cells = wrong.any(-1).nonzero().flatten().tolist()
                    c = cells[0]; r, cc = divmod(c, 15)
                    first = dict(loop=t, n_cells=len(cells), cell=(r, cc), fields=[names[k] for k in wrong[c].nonzero().flatten().tolist()],
                                 is_start=(r, cc) == tuple(it["s"]), is_goal=(r, cc) == tuple(it["g"]), dist_SG=it["dS"],
                                 on_path=bool(it["pidx"][r, cc] >= 0), path_idx=int(it["pidx"][r, cc]))
            ok = bool(((h[0, :, 14] > .5) == (traj[T, 0, :, 7] > .5)).all())
            if not ok: failed.append(dict(maze=idx, **(first or {})))
    out[path] = failed
    print(path, "failed", len(failed))
    for f in failed[:25]: print("  ", f)
json.dump(out, open("results/fails.json", "w"), indent=1)
