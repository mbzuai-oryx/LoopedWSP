"""The LoopedWSP maze block with a discretizer built into the loop, trained end to end.

loop step:  h <- D(block(h), h0)
  D rounds the 8 search numbers to 0/1 (straight-through: forward = round, backward = identity)
  and resets the 7 static and 20 scratch numbers to their embedding values h0.
So every state the block ever sees is a clean 0/1 search state; the only way to go wrong is a wrong bit.

modes:
  bptt   : roll out from the model's own loop-0 state for all T loops, loss after every loop, gradient
           through every loop (optionally truncated to the last --tbptt loops)
  unroll : start from the TRUE state at a random loop t, roll out --k loops, loss after every loop
Losses are on the pre-rounding search numbers (changing cells weighted up), so rounding never hides an error.
The readout is trained on a detached copy of the state (it cannot pull the block toward copying).
"""
import argparse, json, time, os
import numpy as np, torch, torch.nn.functional as F
from maze import make_dataset, trajectory_w
from wsp import Looped, maze_tokens

p = argparse.ArgumentParser()
p.add_argument("--mode", default="bptt"); p.add_argument("--k", type=int, default=8)
p.add_argument("--attn", default="softmax"); p.add_argument("--pos", default="learned")  # learned | none
p.add_argument("--lr", type=float, default=2e-3); p.add_argument("--steps", type=int, default=3000)
p.add_argument("--bs", type=int, default=16); p.add_argument("--n", type=int, default=15)
p.add_argument("--tbptt", type=int, default=0); p.add_argument("--seed", type=int, default=0)
p.add_argument("--eval_every", type=int, default=500); p.add_argument("--name", required=True)
p.add_argument("--threads", type=int, default=2); p.add_argument("--init_from", default="")
p.add_argument("--no_st", action="store_true", help="ablation: stop the gradient at the rounding")
p.add_argument("--local", action="store_true", help="relbias: mask keys outside the (2K+1)^2 window")
p.add_argument("--sizes", default="", help="comma list of training maze sizes, e.g. 11,15,21 (default: --n)")
p.add_argument("--cosine", action="store_true", help="cosine-decay the learning rate from --lr to lr/20 over --steps")
p.add_argument("--hinge", type=float, default=0.0, help="weight of a summed margin penalty on any bit within 0.25 of the rounding threshold")
p.add_argument("--focus_ties", type=float, default=0.0, help="prob. that an unroll starts just before the path passes a tie (a path cell with 2+ neighbours one step closer)")
p.add_argument("--focus", type=float, default=0.0, help="prob. that an unroll starts just before the path leaves the start cell (loop dS-2..dS+1)")
p.add_argument("--offset_aug", action="store_true", help="shift each batch to a random place in the 128x128 coordinate frame")
a = p.parse_args()
torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
os.makedirs("results", exist_ok=True)
S = slice(7, 15)

sizes = [int(x) for x in a.sizes.split(",")] if a.sizes else [a.n]
from maze import DIRS


def tie_loops(it):
    """Loops at which the path takes a step from a cell with 2+ neighbours one step closer to the goal."""
    d, pidx, out = it["dist"], it["pidx"], []
    for (r, c) in zip(*np.nonzero(pidx >= 0)):
        if d[r, c] > 0 and sum(d[r + a_, c + b_] == d[r, c] - 1 for a_, b_ in DIRS) > 1:
            out.append(int(it["dS"] + pidx[r, c] + 1))
    return out
trains = {n_: make_dataset(1000 if n_ <= 15 else 300, n_, seed=1000 + a.seed + n_) for n_ in sizes}
tests = {15: make_dataset(100, 15, seed=99), 21: make_dataset(100, 21, seed=98)}

m = Looped(attn=a.attn, local=a.local)
if a.pos == "none":   # no absolute positions at all: cells know only wall / open / start / goal
    with torch.no_grad(): m.E_row.zero_(); m.E_col.zero_()
    m.E_row.requires_grad_(False); m.E_col.requires_grad_(False)
if a.init_from:
    m.load_state_dict(torch.load(a.init_from), strict=False)
opt = torch.optim.Adam([q for q in m.parameters() if q.requires_grad], lr=a.lr)


def batch(items, T=None):
    T = T or 2 * max(i["dS"] for i in items) + 4
    tok = torch.tensor(np.stack([maze_tokens(i["wall"], i["s"], i["g"]) for i in items]))
    traj = torch.tensor(np.stack([trajectory_w(i, T) for i in items]), dtype=torch.float32).transpose(0, 1)
    return tok, traj, T


def disc(h, h0):
    x = h[..., S]
    r = x.round().clamp(0, 1)
    r = r.detach() if a.no_st else x + (r - x).detach()
    return torch.cat([h0[..., :7], r, h0[..., 15:]], -1)


def step_loss(pre, tgt, prev):
    chg = (tgt != prev).any(-1, keepdim=True).float()
    w = 1 + chg * (chg.numel() / chg.sum().clamp(min=1))
    L = ((pre - tgt) ** 2 * w).mean() / w.mean()
    if a.hinge > 0:   # every near-miss counts on its own, however rare: summed, not averaged over the grid
        L = L + a.hinge * F.relu((pre - tgt).abs() - 0.25).pow(2).sum() / pre.shape[0]
    return L


def chars_target(tok, state):
    ch = tok[..., 2].clone(); ch[(state[..., 7] > 0.5) & (ch == 1)] = 4
    return ch


@torch.no_grad()
def evaluate(items, mult=1, bs=25):
    """Free rollout of the model as it is (discretizer included), from its own loop-0 state."""
    st = ro = clean = 0
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        tok, traj, T = batch(chunk)
        h0 = m.embed(tok); h = disc(h0, h0); ok = torch.ones(len(chunk), dtype=torch.bool)
        ok &= (h[..., S] == traj[0]).all(-1).all(-1)
        for t in range(1, T * mult + 1):
            h = disc(m.block(h), h0)
            if t <= T: ok &= (h[..., S] == traj[t]).all(-1).all(-1)
        fin = traj[T]
        st += ((h[..., 14] > .5) == (fin[..., 7] > .5)).all(-1).sum().item()
        ro += (m.readout(h).argmax(-1) == chars_target(tok, fin)).all(-1).sum().item()
        clean += ok.sum().item()
    n = len(items)
    return dict(solved_state=st / n, solved_readout=ro / n, every_loop_exact=clean / n)


log = dict(args=vars(a), evals=[], loss=[])
t0 = time.time(); rng = np.random.default_rng(a.seed)
for step in range(1, a.steps + 1):
    if a.cosine:
        for g_ in opt.param_groups: g_['lr'] = a.lr * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * (step - 1) / a.steps)))
    n_ = sizes[rng.integers(len(sizes))]; tr = trains[n_]
    bs = max(4, a.bs * 225 // (n_ * n_)) if len(sizes) > 1 else a.bs   # keep cells per batch roughly constant
    items = [tr[j] for j in rng.integers(len(tr), size=bs)]
    tok, traj, T = batch(items)
    if a.offset_aug:
        tok = tok.clone(); tok[..., 0] += int(rng.integers(0, 128 - n_ + 1)); tok[..., 1] += int(rng.integers(0, 128 - n_ + 1))
    h0 = m.embed(tok)
    loss = F.mse_loss(h0[..., S], traj[0])
    if a.mode == "bptt":
        h = disc(h0, h0); hs = []
        for t in range(1, T + 1):
            if a.tbptt and t <= T - a.tbptt: h = h.detach()
            pre = m.block(h)
            loss = loss + step_loss(pre[..., S], traj[t], traj[t - 1]) / T
            h = disc(pre, h0); hs.append(h)
        ce = sum(F.cross_entropy(m.readout(x.detach()).reshape(-1, 5), chars_target(tok, traj[i + 1]).reshape(-1)) for i, x in enumerate(hs[-4:]))
    else:
        ar = torch.arange(bs); t = rng.integers(0, T, size=bs)
        if a.focus > 0:   # oversample the rare loops where the path takes its first step out of S
            pick = rng.random(bs) < a.focus
            near = np.array([it["dS"] - 2 + rng.integers(0, 4) for it in items])
            t = np.where(pick, np.clip(near, 0, T - 1), t)
        if a.focus_ties > 0:   # oversample the loops where the path passes a tie
            for i_, it in enumerate(items):
                tl = it.setdefault("_ties", tie_loops(it))
                if tl and rng.random() < a.focus_ties:
                    t[i_] = int(np.clip(tl[rng.integers(len(tl))] - 1 - rng.integers(0, 4), 0, T - 1))
        t = torch.tensor(t)
        h = torch.cat([h0[..., :7], traj[t, ar], h0[..., 15:]], -1); ce = 0.0
        for i in range(a.k):
            ti = (t + 1 + i).clamp(max=T)
            pre = m.block(h)
            loss = loss + step_loss(pre[..., S], traj[ti, ar], traj[(ti - 1), ar]) / a.k
            h = disc(pre, h0)
            ce = ce + F.cross_entropy(m.readout(h.detach()).reshape(-1, 5), chars_target(tok, traj[ti, ar]).reshape(-1)) / a.k
    loss = loss + ce
    opt.zero_grad()
    if not torch.isfinite(loss): log["diverged_at"] = step; print("diverged", step); break
    loss.backward(); gn = float(torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0))
    if not np.isfinite(gn):   # exploding gradient through the loop: skip the update, count it
        opt.zero_grad(); log["skipped"] = log.get("skipped", 0) + 1
    else:
        opt.step()
    if step % 50 == 0: log["loss"].append((step, float(loss), gn))
    if step % a.eval_every == 0 or step == a.steps:
        ev = dict(step=step, time=round(time.time() - t0), n15=evaluate(tests[15]))
        if step == a.steps or step % (a.eval_every * 2) == 0:
            ev["n15_x2"] = evaluate(tests[15], mult=2); ev["n21"] = evaluate(tests[21])
        log["evals"].append(ev); print(a.name, json.dumps(ev), flush=True)
        json.dump(log, open(f"results/{a.name}.json", "w")); torch.save(m.state_dict(), f"results/{a.name}.pt")
