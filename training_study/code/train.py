"""Train the LoopedWSP maze architecture (random init) under several supervision regimes.

modes:
  final : CE on the final readout only (the plain setup)
  bptt  : free rollout, MSE on the 8 search numbers + CE readout after EVERY loop, backprop through all loops
  tf    : one-step teacher forcing: input = true state at loop t, target = true state at loop t+1
options: --norm (pre-RMSNorm), --noise (tf input corruption), --st (straight-through rounding inside the loop)
"""
import argparse, json, time, os, math
import numpy as np, torch, torch.nn.functional as F
from maze import make_dataset, trajectory_w
from wsp import Looped, maze_tokens, load_written

p = argparse.ArgumentParser()
p.add_argument("--mode", default="bptt"); p.add_argument("--norm", action="store_true")
p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--steps", type=int, default=3000)
p.add_argument("--bs", type=int, default=16); p.add_argument("--n", type=int, default=15)
p.add_argument("--ntrain", type=int, default=1000); p.add_argument("--seed", type=int, default=0)
p.add_argument("--noise", type=float, default=0.0); p.add_argument("--st", action="store_true")
p.add_argument("--balanced", action="store_true"); p.add_argument("--clip", type=float, default=1.0)
p.add_argument("--init", default="random")   # random | written | written+noise:<eps>
p.add_argument("--hidden", type=int, default=86); p.add_argument("--width", type=int, default=35)
p.add_argument("--tbptt", type=int, default=0)  # truncate gradient to last k loops (0 = full)
p.add_argument("--eval_every", type=int, default=500); p.add_argument("--name", default=None)
p.add_argument("--chg_weight", action="store_true")
p.add_argument("--ce_w", type=float, default=1.0); p.add_argument("--keep_w", type=float, default=0.0); p.add_argument("--unroll", type=int, default=1); p.add_argument("--init_from", default=""); p.add_argument("--noise_all", type=float, default=0.0); p.add_argument("--ce_detach", action="store_true")
p.add_argument("--attn", default="softmax"); p.add_argument("--pos", default="learned")
p.add_argument("--threads", type=int, default=2); p.add_argument("--gen", default="21:40")
a = p.parse_args()
torch.set_num_threads(a.threads); torch.manual_seed(a.seed); np.random.seed(a.seed)
name = a.name or f"{a.mode}{'_norm' if a.norm else ''}_lr{a.lr}_s{a.seed}"
os.makedirs("results", exist_ok=True)

train = make_dataset(a.ntrain, a.n, seed=1000 + a.seed)
test = make_dataset(100, a.n, seed=99)
gn, gk = map(int, a.gen.split(":"))
gen = make_dataset(gk, gn, seed=98)


def batch(items, T=None):
    T = T or 2 * max(i["dS"] for i in items) + 4
    tok = torch.tensor(np.stack([maze_tokens(i["wall"], i["s"], i["g"]) for i in items]))
    traj = torch.tensor(np.stack([trajectory_w(i, T) for i in items]), dtype=torch.float32).transpose(0, 1)
    return tok, traj, T  # traj: (T+1, B, N, 8)


def chars_target(tok, state):  # readout target implied by the state: path flag -> 'o' (except S, G)
    ch = tok[..., 2].clone()
    on = (state[..., 7] > 0.5) & (ch == 1)
    ch[on] = 4
    return ch


if a.init == "random":
    m = Looped(d=a.width, hidden=a.hidden, norm=a.norm, attn=a.attn)
    if a.pos == "written_frozen":   # the written r, c, r^2+c^2 coordinates, frozen
        Wr = load_written(None, dtype=torch.float32)
        with torch.no_grad(): m.E_row.copy_(Wr.E_row); m.E_col.copy_(Wr.E_col)
        m.E_row.requires_grad_(False); m.E_col.requires_grad_(False)
else:
    m = load_written(None, dtype=torch.float32)
    if a.init.startswith("written+noise"):
        eps = float(a.init.split(":")[1])
        with torch.no_grad():
            for prm in m.parameters():
                prm.mul_(1 + eps * torch.randn_like(prm))  # relative noise on every written weight
if a.init_from:
    m.load_state_dict(torch.load(a.init_from))
opt = torch.optim.Adam([q for q in m.parameters() if q.requires_grad], lr=a.lr)
S = slice(7, 15)


def snap(h, h0):
    h = h.clone()
    h[..., S] = h[..., S].round().clamp(0, 1)
    h[..., :7] = h0[..., :7]
    h[..., 15:] = h0[..., 15:]
    return h


def st_round(x):  # straight-through rounding of the search numbers
    return x + (x.round().clamp(0, 1) - x).detach()


def ce_weight(tgt):
    if not a.balanced: return None
    w = torch.ones(5)
    cnt = torch.bincount(tgt.flatten(), minlength=5).float()
    w[4] = (cnt.sum() - cnt[4]) / cnt[4].clamp(min=1)  # path cells count as much as all others
    return w


@torch.no_grad()
def evaluate(items, snap_eval=False, mult=1, bs=25):
    solved = stable = solved_state = 0; cell_err_loops = []; first_err = []
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        tok, traj, T = batch(chunk)
        R = T * mult
        h0 = m.embed(tok); h = h0
        if snap_eval: h = snap(h, h0)
        errs = []
        for t in range(1, R + 1):
            h = m.block(h)
            if a.st: h = torch.cat([h[..., :7], h[..., S].round().clamp(0, 1), h[..., 15:]], -1)
            if snap_eval: h = snap(h, h0)
            if t <= T:
                wrong = ((h[..., S] > 0.5).float() != traj[t]).any(-1)  # (B, N)
                errs.append(wrong)
        E = torch.stack(errs)  # (T, B, N)
        cell_err_loops.append(E.float().mean((1, 2)))
        anyerr = E.any(-1)  # (T, B)
        for b in range(len(chunk)):
            idx = torch.nonzero(anyerr[:, b])
            first_err.append(int(idx[0]) + 1 if len(idx) else 10 ** 6)
        pred = m.readout(h).argmax(-1)
        tgt = chars_target(tok, traj[min(T, R)])
        solved += (pred == tgt).all(-1).sum().item()
        solved_state += ((h[..., 14] > 0.5) == (traj[min(T, R)][..., 7] > 0.5)).all(-1).sum().item()
    fe = np.array(first_err)
    return dict(solved=solved / len(items), solved_state=solved_state / len(items), cell_err=float(torch.cat([c[:20] for c in cell_err_loops]).mean()) if cell_err_loops else None,
                median_first_err=float(np.median(fe)), frac_no_err=float((fe == 10 ** 6).mean()))


@torch.no_grad()
def onestep(items, bs=25):
    """Teacher-forced: feed the TRUE state at every loop t, check the predicted state at t+1.
    Returns fraction of (cell, t) transitions exactly right, and fraction of mazes with no wrong step."""
    ok_cells = tot = 0; ok_mazes = 0; ok_chg = n_chg = 0
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        tok, traj, T = batch(chunk)
        h0 = m.embed(tok); allok = torch.ones(len(chunk), dtype=torch.bool)
        for t in range(T):
            h1 = m.block(torch.cat([h0[..., :7], traj[t], h0[..., 15:]], -1))
            good = ((h1[..., S] > 0.5).float() == traj[t + 1]).all(-1)
            ok_cells += good.sum().item(); tot += good.numel(); allok &= good.all(-1)
            chg = (traj[t + 1] != traj[t]).any(-1); ok_chg += (good & chg).sum().item(); n_chg += chg.sum().item()
        ok_mazes += allok.sum().item()
    return dict(step_cell_acc=ok_cells / tot, changed_cell_acc=ok_chg / max(n_chg, 1), step_maze_exact=ok_mazes / len(items))


log = dict(args=vars(a), evals=[], loss=[])
t0 = time.time()
rng = np.random.default_rng(a.seed)
for step in range(1, a.steps + 1):
    items = [train[j] for j in rng.integers(len(train), size=a.bs)]
    tok, traj, T = batch(items)
    h0 = m.embed(tok)
    if a.mode == "tf":
        t = torch.tensor(rng.integers(0, T, size=a.bs))
        cur = traj[t, torch.arange(a.bs)]; nxt = traj[t + 1, torch.arange(a.bs)]
        if a.noise > 0:  # corrupt the input state so the step must also correct errors
            flip = (torch.rand_like(cur) < a.noise).float()
            cur = (cur + flip * (torch.rand_like(cur) * 1.0)).clamp(0, 1)  # soft corruptions
            cur = cur + a.noise * torch.randn_like(cur) * 0.5
        h_in = torch.cat([h0[..., :7], cur, h0[..., 15:]], -1)
        if a.noise_all > 0:   # perturb every number; the target is still the clean next state
            h_in = h_in + a.noise_all * torch.randn_like(h_in)
        loss_init = F.mse_loss(h0[..., S], traj[0])
        h1 = m.block(h_in)
        if a.chg_weight:   # cells whose state changes this loop weigh as much as all the others together
            chg = (nxt != traj[t, torch.arange(a.bs)]).any(-1, keepdim=True).float()
            w = 1 + chg * (chg.numel() / chg.sum().clamp(min=1))
            loss = ((h1[..., S] - nxt) ** 2 * w).mean() / w.mean() + loss_init
        else:
            loss = F.mse_loss(h1[..., S], nxt) + loss_init
        # the scratch must be clean after the step (what the hand-written block does)
        loss = loss + F.mse_loss(h1[..., 15:], h0[..., 15:].detach()) * 0.1
        if a.keep_w > 0:   # static numbers must come back unchanged, scratch must come back clean
            loss = loss + a.keep_w * (F.mse_loss(h1[..., :7], h0[..., :7].detach()) + F.mse_loss(h1[..., 15:], h0[..., 15:].detach()))
        if a.unroll > 1:   # keep running the block on its OWN output, supervising every loop
            hh = h1; ar = torch.arange(a.bs)
            for i in range(1, a.unroll):
                ti = (t + 1 + i).clamp(max=T); prev = traj[(ti - 1), ar]; tgt_i = traj[ti, ar]
                hh = m.block(hh)
                if a.chg_weight:
                    chg = (tgt_i != prev).any(-1, keepdim=True).float()
                    w = 1 + chg * (chg.numel() / chg.sum().clamp(min=1))
                    li = ((hh[..., S] - tgt_i) ** 2 * w).mean() / w.mean()
                else:
                    li = F.mse_loss(hh[..., S], tgt_i)
                li = li + a.keep_w * (F.mse_loss(hh[..., :7], h0[..., :7].detach()) + F.mse_loss(hh[..., 15:], h0[..., 15:].detach()))
                loss = loss + li
        lg = m.readout(h1.detach() if a.ce_detach else h1)
        tgt = chars_target(tok, nxt)
        loss = loss + a.ce_w * F.cross_entropy(lg.reshape(-1, 5), tgt.reshape(-1), weight=ce_weight(tgt))
    else:
        h = h0
        loss = F.mse_loss(h[..., S], traj[0]) if a.mode == "bptt" else 0.0
        for t in range(1, T + 1):
            if a.tbptt and t <= T - a.tbptt: h = h.detach()
            h = m.block(h)
            if a.st: h = torch.cat([h[..., :7], st_round(h[..., S]), h[..., 15:]], -1)
            if a.mode == "bptt":
                tgt = chars_target(tok, traj[t])
                loss = loss + (F.mse_loss(h[..., S], traj[t]) + a.ce_w *
                               F.cross_entropy(m.readout(h.detach() if a.ce_detach else h).reshape(-1, 5), tgt.reshape(-1), weight=ce_weight(tgt))) / T
        if a.mode == "final":
            tgt = chars_target(tok, traj[T])
            loss = F.cross_entropy(m.readout(h).reshape(-1, 5), tgt.reshape(-1), weight=ce_weight(tgt))
    opt.zero_grad()
    if not torch.isfinite(loss):
        log["diverged_at"] = step; print("diverged at", step); break
    loss.backward()
    gnorm = float(torch.nn.utils.clip_grad_norm_(m.parameters(), a.clip if a.clip > 0 else 1e12))
    opt.step()
    if step % 50 == 0:
        log["loss"].append((step, float(loss), gnorm))
    if step % a.eval_every == 0 or step == a.steps:
        ev = dict(step=step, time=time.time() - t0, onestep=onestep(test), test=evaluate(test), test_snap=evaluate(test, snap_eval=True))
        if step == a.steps:
            ev["test_x2"] = evaluate(test, mult=2)
            ev["gen"] = evaluate(gen); ev["gen_snap"] = evaluate(gen, snap_eval=True)
        log["evals"].append(ev)
        print(name, json.dumps(ev), flush=True)
        json.dump(log, open(f"results/{name}.json", "w"))
torch.save(m.state_dict(), f"results/{name}.pt")
json.dump(log, open(f"results/{name}.json", "w"))
