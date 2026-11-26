"""Probe a train_disc.py model: one-step (teacher-forced) accuracy on changing cells by field,
and the loop of the first wrong bit in its own (discretized) rollout. usage: probe_disc.py MODEL.pt [SIZE]"""
import sys, json
import numpy as np, torch
from maze import make_dataset, trajectory_w
from wsp import Looped, maze_tokens
torch.set_num_threads(2); S = slice(7, 15)
path = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 15
args = json.load(open(path.replace(".pt", ".json")))["args"]
m = Looped(attn=args["attn"], local=args.get("local", False)); m.load_state_dict(torch.load(path))
items = make_dataset(50, n, seed={15: 99, 21: 98}.get(n, 97))
disc = lambda h, h0: torch.cat([h0[..., :7], h[..., S].round().clamp(0, 1), h0[..., 15:]], -1)
cnt = {}; first = []; dS = []
def add(k, v): cnt[k] = cnt.get(k, 0) + int(v)
with torch.no_grad():
    for i in range(0, 50, 25):
        ch = items[i:i + 25]; T = 2 * max(it["dS"] for it in ch) + 4
        tok = torch.tensor(np.stack([maze_tokens(it["wall"], it["s"], it["g"]) for it in ch]))
        traj = torch.tensor(np.stack([trajectory_w(it, T) for it in ch])).transpose(0, 1)
        h0 = m.embed(tok)
        for t in range(T):
            out = m.block(torch.cat([h0[..., :7], traj[t], h0[..., 15:]], -1))[..., S]
            pred = (out > .5).float(); tgt = traj[t + 1]; cur = traj[t]
            for f, sl in [("wave", slice(0, 3)), ("pointer", slice(3, 7)), ("path", slice(7, 8))]:
                chg = (tgt[..., sl] != cur[..., sl]).any(-1); w = (pred[..., sl] != tgt[..., sl]).any(-1)
                add(f + "_chg", chg.sum()); add(f + "_chg_wrong", (w & chg).sum())
                add(f + "_stay", (~chg).sum()); add(f + "_stay_wrong", (w & ~chg).sum())
        h = disc(h0, h0); fe = torch.full((len(ch),), -1)
        for t in range(1, T + 1):
            h = disc(m.block(h), h0)
            bad = (h[..., S] != traj[t]).any(-1).any(-1)
            fe = torch.where(bad & (fe < 0), torch.full_like(fe, t), fe)
        first += fe.tolist(); dS += [it["dS"] for it in ch]
res = {f: dict(changing=f"{cnt[f+'_chg_wrong']}/{cnt[f+'_chg']} wrong", staying=f"{cnt[f+'_stay_wrong']}/{cnt[f+'_stay']} wrong") for f in ["wave", "pointer", "path"]}
fe = np.array(first); res["rollout_first_wrong_loop_median"] = float(np.median(np.where(fe < 0, 10**6, fe)))
res["rollouts_never_wrong"] = int((fe < 0).sum()); res["median_dS"] = float(np.median(dS))
print(path, n, json.dumps(res))
