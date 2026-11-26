import json, numpy as np, torch
from maze import make_dataset, trajectory_w
from wsp import Looped, maze_tokens
S = slice(7, 15); path = "results/R5_tf_gather_chg.pt"
m = Looped(attn="gather"); m.load_state_dict(torch.load(path))
items = make_dataset(100, 15, seed=99)
ch = items[:25]; T = 2 * max(it["dS"] for it in ch) + 4
tok = torch.tensor(np.stack([maze_tokens(it["wall"], it["s"], it["g"]) for it in ch]))
traj = torch.tensor(np.stack([trajectory_w(it, T) for it in ch])).transpose(0, 1)
with torch.no_grad():
    h0 = m.embed(tok); allok = torch.ones(25, dtype=torch.bool)
    for t in range(T):
        h1 = m.block(torch.cat([h0[..., :7], traj[t], h0[..., 15:]], -1))
        allok &= ((h1[..., S] > .5).float() == traj[t + 1]).all(-1).all(-1)
    print("TF all-ok mazes:", allok.nonzero().flatten().tolist())
    h = torch.cat([h0[..., :7], traj[0], h0[..., 15:]], -1)
    for t in range(1, T + 1):
        hn = m.block(h)
        hs = torch.cat([hn[..., :7], hn[..., S].round().clamp(0, 1), h0[..., 15:]], -1)
        tf = m.block(torch.cat([h0[..., :7], traj[t-1], h0[..., 15:]], -1))
        if t < 4: print(t, "input equal to TF input:", torch.equal(h, torch.cat([h0[..., :7], traj[t-1], h0[..., 15:]], -1)),
                        "static dims max diff after block:", (hn[..., :7] - h0[..., :7]).abs().max().item())
        h = hs
