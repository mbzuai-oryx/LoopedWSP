"""Diagnostics on the hand-written solution vs random init (no training needed)."""
import json, math, sys
import numpy as np, torch, torch.nn.functional as F
from maze import make_dataset, trajectory_w
from wsp import Looped, load_written, maze_tokens

torch.set_num_threads(2); torch.manual_seed(0)
out = {}
S = slice(7, 15)
items = make_dataset(40, 15, seed=99)


def batch(items, T=None):
    T = T or 2 * max(i["dS"] for i in items) + 4
    tok = torch.tensor(np.stack([maze_tokens(i["wall"], i["s"], i["g"]) for i in items]))
    traj = torch.tensor(np.stack([trajectory_w(i, T) for i in items]), dtype=torch.float64).transpose(0, 1)
    return tok, traj, T


W = load_written(None)
tok, traj, T = batch(items)

# ---- 1. scale of the written solution vs a standard init
blk = W.blocks[0]
R0 = Looped().double()
scale = {}
for (n, p), (_, p0) in zip(W.named_parameters(), R0.named_parameters()):
    scale[n] = dict(max_abs=float(p.abs().max()), rms_written=float(p.pow(2).mean().sqrt()),
                    rms_init=float(p0.pow(2).mean().sqrt()), nonzero=int((p != 0).sum()), numel=p.numel())
out["scale"] = scale
out["n_params"] = sum(p.numel() for p in W.parameters() if True)

# ---- 2. attention sharpness: logit gap between the chosen key and the runner-up, and leaked mass
with torch.no_grad():
    h = W.embed(tok)
    for _ in range(5): h = W.block(h)
    q = blk.Wq(h).view(*h.shape[:2], 4, 8).transpose(1, 2); k = blk.Wk(h).view(*h.shape[:2], 4, 8).transpose(1, 2)
    s = q @ k.transpose(-1, -2) / math.sqrt(8)
    top2 = s.topk(2, -1).values
    gap = (top2[..., 0] - top2[..., 1])
    a = torch.softmax(s, -1)
    leak = 1 - a.max(-1).values
    out["attn"] = dict(max_abs_logit=float(s.abs().max()), gap_min=float(gap.min()), gap_median=float(gap.median()),
                       leak_max=float(leak.max()),
                       softmax_grad_max=float((a * (1 - a)).max()))

# ---- 3. dynamics: does the loop correct a perturbation of the state? (written vs random init)
def perturb_growth(model, sigma, loops=40, snap_free=True):
    with torch.no_grad():
        h = model.embed(tok)
        for _ in range(10): h = model.block(h)
        hp = h.clone(); hp[..., S] += sigma * torch.randn_like(hp[..., S])
        d = []
        for t in range(loops):
            h = model.block(h); hp = model.block(hp)
            d.append(float((h[..., S] - hp[..., S]).abs().max()))
        return d
out["perturb_written"] = {str(sg): perturb_growth(W, sg) for sg in [0.05, 0.1, 0.2, 0.3, 0.4]}
for norm in [False, True]:
    torch.manual_seed(1)
    R = Looped(norm=norm).double()
    with torch.no_grad():
        h = R.embed(tok); norms = []
        for t in range(200):
            h = R.block(h); norms.append(float(h.pow(2).mean().sqrt()))
            if not math.isfinite(norms[-1]): break
    out[f"init_state_rms_{'norm' if norm else 'nonorm'}"] = norms
    out[f"perturb_init_{'norm' if norm else 'nonorm'}"] = perturb_growth(R, 1e-3)


# ---- 4. basin: solve rate of the written weights under relative weight noise; and gradient at/near it
def solve_rate(model, n=40):
    with torch.no_grad():
        lg = model(tok[:n], T)
        pred = lg.argmax(-1)
        ch = tok[:n, :, 2].clone(); on = (traj[T, :n, :, 7] > 0.5) & (ch == 1); ch[on] = 4
        return float((pred == ch).all(-1).double().mean())


def sup_loss(model, n=8):
    h = model.embed(tok[:n]); L = F.mse_loss(h[..., S], traj[0, :n])
    for t in range(1, T + 1):
        h = model.block(h); L = L + F.mse_loss(h[..., S], traj[t, :n]) / T
    return L


basin = []
for eps in [0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]:
    rates, losses, gn = [], [], {}
    for seed in range(3):
        torch.manual_seed(seed)
        M = load_written(None)
        with torch.no_grad():
            for p in M.parameters(): p.mul_(1 + eps * torch.randn_like(p))
        rates.append(solve_rate(M))
        if seed == 0:
            M.zero_grad(); L = sup_loss(M)
            if torch.isfinite(L):
                L.backward()
                gn = {n: float(p.grad.norm()) if p.grad is not None else 0.0 for n, p in M.named_parameters()}
            losses.append(float(L))
    basin.append(dict(eps=eps, solve=rates, loss=losses, grad=gn))
    print("eps", eps, rates, losses, flush=True)
out["basin"] = basin

json.dump(out, open("results/diag.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k in ("attn", "n_params")}, indent=1))
for k in ["perturb_written"]:
    for sg, d in out[k].items(): print(k, sg, [round(x, 3) for x in d[:12]], "...", round(d[-1], 4))
for k in ["init_state_rms_nonorm", "init_state_rms_norm"]:
    v = out[k]; print(k, [f"{v[i]:.3g}" for i in [0, 4, 9, 19, 49, 99, len(v) - 1] if i < len(v)], "len", len(v))
for k in ["perturb_init_nonorm", "perturb_init_norm"]:
    v = out[k]; print(k, [f"{x:.3g}" for x in v[::5]])
for n, sc in scale.items(): print(n, sc)
