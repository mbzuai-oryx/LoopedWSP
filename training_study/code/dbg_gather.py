import sys, numpy as np, torch, torch.nn.functional as F
LR=float(sys.argv[1]); CE=float(sys.argv[2]); ATT=sys.argv[3] if len(sys.argv)>3 else 'gather'
from maze import make_dataset, trajectory_w
from wsp import Looped, maze_tokens
torch.set_num_threads(2); torch.manual_seed(0)
tr = make_dataset(300, 15, seed=5)
m = Looped(attn=ATT); opt = torch.optim.Adam(m.parameters(), LR)
rng = np.random.default_rng(0); S = slice(7, 15)
def batch(items):
    T = 2 * max(i["dS"] for i in items) + 4
    tok = torch.tensor(np.stack([maze_tokens(i["wall"], i["s"], i["g"]) for i in items]))
    traj = torch.tensor(np.stack([trajectory_w(i, T) for i in items])).transpose(0, 1)
    return tok, traj, T
for step in range(1, 1501):
    items = [tr[j] for j in rng.integers(300, size=16)]
    tok, traj, T = batch(items)
    t = torch.tensor(rng.integers(0, T, size=16)); ar = torch.arange(16)
    cur, nxt = traj[t, ar], traj[t + 1, ar]
    h0 = m.embed(tok); h1 = m.block(torch.cat([h0[..., :7], cur, h0[..., 15:]], -1))
    chg = (nxt != cur).any(-1)
    loss = F.mse_loss(h1[..., S], nxt) + F.mse_loss(h0[..., S], traj[0])
    if CE > 0:
        ch = tok[..., 2].clone(); ch[(nxt[..., 7] > .5) & (ch == 1)] = 4
        loss = loss + CE * F.cross_entropy(m.readout(h1).reshape(-1, 5), ch.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 250 == 0:
        with torch.no_grad():
            pred = (h1[..., S] > .5).float()
            acc_chg = (pred == nxt).all(-1)[chg].float().mean().item() if chg.any() else float('nan')
            print(step, round(loss.item(), 5), "changed cells:", int(chg.sum()), "acc on changed", round(acc_chg, 3),
                  "upd norm", round((h1[..., S] - cur).abs().mean().item(), 4), flush=True)
