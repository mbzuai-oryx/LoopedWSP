"""What does the gradient on the attention scores ask for at init? Position (neighbour) or content (visited)?

For each (query i, key j) we get g_ij = dL/ds_ij under the one-step teacher-forced loss. We average it
  (a) over pairs grouped by the key's offset from the query (dr, dc), and
  (b) over pairs grouped by the key's content (visited or not), excluding the 4 neighbours.
Positive = 'attend more here' (the gradient-descent step is -g, so we report -g)."""
import json, math
import numpy as np, torch, torch.nn.functional as F
from maze import make_dataset, trajectory_w
from wsp import Looped, maze_tokens

torch.set_num_threads(2); torch.manual_seed(0)
S = slice(7, 15); n = 15
items = make_dataset(64, n, seed=7)
rng = np.random.default_rng(0)
res_off = {}; content = {"visited_far": [], "unvisited_far": []}; nb = []
for seed in range(3):
    torch.manual_seed(seed)
    m = Looped()
    b = m.blocks[0]
    for it in items:
        T = 2 * it["dS"] + 2
        tok = torch.tensor(maze_tokens(it["wall"], it["s"], it["g"]))[None]
        traj = torch.tensor(trajectory_w(it, T))
        t = int(rng.integers(1, it["dS"]))
        h0 = m.embed(tok)
        x = torch.cat([h0[..., :7], traj[t][None], h0[..., 15:]], -1)
        q = b.Wq(x).view(1, -1, 4, 8).transpose(1, 2); k = b.Wk(x).view(1, -1, 4, 8).transpose(1, 2)
        v = b.Wv(x).view(1, -1, 4, 8).transpose(1, 2)
        s = (q @ k.transpose(-1, -2) / math.sqrt(8)).detach().requires_grad_(True)
        a = torch.softmax(s, -1); o = (a @ v).transpose(1, 2).reshape(1, -1, 32)
        h = x + b.Wo(o); h = h + b.W2(F.relu(b.W1(h)))
        loss = F.mse_loss(h[..., S], traj[t + 1][None])
        loss.backward()
        g = -s.grad[0].sum(0)  # (N, N), summed over heads
        rc = tok[0, :, :2]
        dr = rc[None, :, 0] - rc[:, None, 0]; dc = rc[None, :, 1] - rc[:, None, 1]
        # only queries whose state changes this step carry signal; normalise per such query
        chg = (traj[t + 1] != traj[t]).any(-1)
        if chg.sum() == 0: continue
        G = g[chg]; DR = dr[chg]; DC = dc[chg]
        for odr in range(-2, 3):
            for odc in range(-2, 3):
                msk = (DR == odr) & (DC == odc)
                if msk.any(): res_off.setdefault((odr, odc), []).append(float(G[msk].mean()))
        vis = (traj[t][:, :3].sum(-1) > 0)
        near = (DR.abs() + DC.abs()) <= 1
        visk = vis[None, :].expand_as(G)
        content["visited_far"].append(float(G[visk & ~near].mean()))
        content["unvisited_far"].append(float(G[~visk & ~near].mean()))
        isnb = (DR.abs() + DC.abs()) == 1
        nb.append(float(G[isnb & visk].mean()) if (isnb & visk).any() else float("nan"))

out = {"by_offset": {f"{k[0]},{k[1]}": float(np.mean(v)) for k, v in res_off.items()},
       "visited_far": float(np.mean(content["visited_far"])), "unvisited_far": float(np.mean(content["unvisited_far"])),
       "visited_neighbour": float(np.nanmean(nb))}
json.dump(out, open("results/diag4.json", "w"), indent=1)
print("mean -dL/ds by key offset (rows dr=-2..2, cols dc=-2..2), x1e6:")
for odr in range(-2, 3):
    print("  ", " ".join(f"{1e6 * out['by_offset'][f'{odr},{odc}']:8.2f}" for odc in range(-2, 3)))
print("visited key, not a neighbour: %.3g | unvisited key, not a neighbour: %.3g | visited neighbour: %.3g" %
      (out["visited_far"], out["unvisited_far"], out["visited_neighbour"]))
