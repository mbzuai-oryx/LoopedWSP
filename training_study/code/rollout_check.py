import sys, json, numpy as np, torch
from maze import make_dataset, trajectory_w
from wsp import Looped, maze_tokens
torch.set_num_threads(2); S = slice(7, 15)
path = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 15
args = json.load(open(path.replace(".pt", ".json")))["args"]
m = Looped(norm=args["norm"], attn=args.get("attn", "softmax")); m.load_state_dict(torch.load(path))
items = make_dataset(100 if n <= 21 else 30, n, seed=99 if n == 15 else 98)
res = dict(init_wrong_mazes=0, snap_true0=0, snapall_true0=0, free_true0=0, first_err=[])
with torch.no_grad():
    for i in range(0, len(items), 25):
        ch = items[i:i+25]; T = 2 * max(it["dS"] for it in ch) + 4
        tok = torch.tensor(np.stack([maze_tokens(it["wall"], it["s"], it["g"]) for it in ch]))
        traj = torch.tensor(np.stack([trajectory_w(it, T) for it in ch])).transpose(0, 1)
        h0 = m.embed(tok)
        res["init_wrong_mazes"] += int(((h0[..., S] > .5).float() != traj[0]).any(-1).any(-1).sum())
        for mode in ["snap_true0", "snapall_true0", "free_true0"]:
            h = torch.cat([h0[..., :7], traj[0], h0[..., 15:]], -1); fe = torch.full((len(ch),), 10**6)
            for t in range(1, T + 1):
                h = m.block(h)
                if mode == "snap_true0": h = torch.cat([h[..., :7], h[..., S].round().clamp(0, 1), h0[..., 15:]], -1)
                if mode == "snapall_true0": h = torch.cat([h0[..., :7], h[..., S].round().clamp(0, 1), h0[..., 15:]], -1)
                bad = ((h[..., S] > .5).float() != traj[t]).any(-1).any(-1)
                fe = torch.where(bad & (fe == 10**6), torch.full_like(fe, t), fe)
            ok = ((h[..., 14] > .5) == (traj[T][..., 7] > .5)).all(-1)
            res[mode] += int(ok.sum())
            if mode == "free_true0": res["first_err"] += fe.tolist()
fe = np.array(res.pop("first_err")); res["free_median_first_err_loop"] = float(np.median(fe)); res["n"] = len(items)
print(path, n, res)
