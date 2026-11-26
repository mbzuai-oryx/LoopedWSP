"""Evaluate a train_disc.py model (discretizer in the loop) on held-out mazes of a given size.
usage: eval_disc.py results/D1_bptt_gather.pt SIZE [N] [MULT]"""
import sys, json
import numpy as np, torch
from maze import make_dataset, trajectory_w
from wsp import Looped, maze_tokens

torch.set_num_threads(2); S = slice(7, 15)
path, n = sys.argv[1], int(sys.argv[2])
N = int(sys.argv[3]) if len(sys.argv) > 3 else 100
mult = int(sys.argv[4]) if len(sys.argv) > 4 else 1
CH = int(sys.argv[5]) if len(sys.argv) > 5 else 10
SEED = int(sys.argv[6]) if len(sys.argv) > 6 else None
args = json.load(open(path.replace(".pt", ".json")))["args"]
m = Looped(attn=args["attn"], local=args.get("local", False)); m.load_state_dict(torch.load(path))
items = make_dataset(N, n, seed=SEED if SEED is not None else {15: 99, 21: 98}.get(n, 97))


def disc(h, h0):
    return torch.cat([h0[..., :7], h[..., S].round().clamp(0, 1), h0[..., 15:]], -1)


st = ro = clean = 0
with torch.no_grad():
    for i in range(0, N, CH):
        ch = items[i:i + CH]; T = 2 * max(it["dS"] for it in ch) + 4
        tok = torch.tensor(np.stack([maze_tokens(it["wall"], it["s"], it["g"]) for it in ch]))
        traj = torch.tensor(np.stack([trajectory_w(it, T) for it in ch])).transpose(0, 1)
        h0 = m.embed(tok); h = disc(h0, h0); ok = (h[..., S] == traj[0]).all(-1).all(-1)
        for t in range(1, T * mult + 1):
            h = disc(m.block(h), h0)
            if t <= T: ok &= (h[..., S] == traj[t]).all(-1).all(-1)
        fin = traj[T]
        st += int(((h[..., 14] > .5) == (fin[..., 7] > .5)).all(-1).sum())
        tgt = tok[..., 2].clone(); tgt[(fin[..., 7] > .5) & (tgt == 1)] = 4
        ro += int((m.readout(h).argmax(-1) == tgt).all(-1).sum())
        clean += int(ok.sum())
res = dict(model=path, size=n, n=N, seed=SEED, loops=f"{mult}T", solved_state=st, solved_readout=ro, every_loop_exact=clean)
print(json.dumps(res))
with open("results/eval_disc.jsonl", "a") as f: f.write(json.dumps(res) + "\n")
