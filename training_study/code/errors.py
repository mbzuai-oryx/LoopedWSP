"""Break down the one-step (teacher-forced) errors of a trained model by field and by situation."""
import sys, json
import numpy as np, torch
from maze import make_dataset, trajectory_w
from wsp import Looped, maze_tokens

torch.set_num_threads(2)
S = slice(7, 15)
items = make_dataset(100, 15, seed=99)
out = {}
for path in sys.argv[1:]:
    args = json.load(open(path.replace(".pt", ".json")))["args"]
    m = Looped(norm=args["norm"], attn=args.get("attn", "softmax")); m.load_state_dict(torch.load(path))
    cnt = {}; margins = []
    def add(k, v=1): cnt[k] = cnt.get(k, 0) + v
    with torch.no_grad():
        for i in range(0, 100, 25):
            chunk = items[i:i + 25]
            T = 2 * max(it["dS"] for it in chunk) + 4
            tok = torch.tensor(np.stack([maze_tokens(it["wall"], it["s"], it["g"]) for it in chunk]))
            traj = torch.tensor(np.stack([trajectory_w(it, T) for it in chunk])).transpose(0, 1)
            h0 = m.embed(tok)
            for t in range(T):
                h1 = m.block(torch.cat([h0[..., :7], traj[t], h0[..., 15:]], -1))[..., S]
                pred = (h1 > 0.5).float(); tgt = traj[t + 1]; cur = traj[t]
                wrong = pred != tgt  # (B, N, 8)
                margins.append((h1 - 0.5).abs()[..., :].flatten())
                for f, sl in [("wave", slice(0, 3)), ("pointer", slice(3, 7)), ("path", slice(7, 8))]:
                    chg = (tgt[..., sl] != cur[..., sl]).any(-1)
                    w = wrong[..., sl].any(-1)
                    add(f"{f}:should_change", int(chg.sum())); add(f"{f}:should_change_wrong", int((w & chg).sum()))
                    add(f"{f}:should_stay", int((~chg).sum())); add(f"{f}:should_stay_wrong", int((w & ~chg).sum()))
    mg = torch.cat(margins)
    res = {k: v for k, v in cnt.items()}
    for f in ["wave", "pointer", "path"]:
        res[f"{f}:err_rate_change"] = cnt[f"{f}:should_change_wrong"] / max(cnt[f"{f}:should_change"], 1)
        res[f"{f}:err_rate_stay"] = cnt[f"{f}:should_stay_wrong"] / max(cnt[f"{f}:should_stay"], 1)
    res["margin_quantiles_0.001_0.01_0.5"] = [float(mg.quantile(q)) for q in [0.001, 0.01, 0.5]] if mg.numel() < 16_000_000 else None
    out[path] = res
    print(path)
    for f in ["wave", "pointer", "path"]:
        print(f"  {f:8s} should change: {cnt[f'{f}:should_change_wrong']}/{cnt[f'{f}:should_change']} wrong"
              f"   should stay: {cnt[f'{f}:should_stay_wrong']}/{cnt[f'{f}:should_stay']} wrong")
json.dump(out, open("results/errors.json", "w"), indent=1)
