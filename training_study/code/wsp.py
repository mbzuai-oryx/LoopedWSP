"""Torch port of the LoopedWSP generic looped transformer (maze config), loadable with the
hand-written weights or randomly initialised for training.

  h = E_row[r] + E_col[c] + E_char[ch]
  repeat R: h += Attn(h) (4 heads, d_k = d_v = 8, softmax(qk^T/sqrt(8))); h += ReLU-MLP(h) (86 neurons)
  logits = h W_out + b_out   (5 classes per cell: '#',' ','S','G','o')
"""
import math, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# The hand-written maze weights from this repository (weights/networks/maze.npz); override with WSP_WEIGHTS.
WEIGHTS = os.environ.get("WSP_WEIGHTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "weights", "networks", "maze.npz"))
CHARSET = "# SGo"


def maze_tokens(wall, s, g):
    n = wall.shape[0]
    ch = np.where(wall == 1, 0, 1)
    ch[s] = 2; ch[g] = 3
    r, c = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return np.stack([r.ravel(), c.ravel(), ch.ravel()], -1)  # (n*n, 3)


class RMS(nn.Module):
    def __init__(self, d):
        super().__init__(); self.g = nn.Parameter(torch.ones(d))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.g


class Looped(nn.Module):
    def __init__(self, d=35, heads=4, dk=8, hidden=86, max_side=128, n_char=5, n_out=5,
                 norm=False, layers=1, attn="softmax", K=2, local=False):
        super().__init__()
        self.attn, self.K, self.local = attn, K, local
        if attn == "relbias":   # learnable bias per head per 2-D offset (|dr|,|dc| <= K), one shared 'far' bias
            self.relb = nn.Parameter(torch.zeros(heads, (2 * K + 1) ** 2 + 1))
        self.d, self.H, self.dk, self.norm = d, heads, dk, norm
        self.E_row = nn.Parameter(torch.randn(max_side, d) * 0.02)
        self.E_col = nn.Parameter(torch.randn(max_side, d) * 0.02)
        self.E_char = nn.Parameter(torch.randn(n_char, d) * 0.02)
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            b = nn.Module()
            b.Wq = nn.Linear(d, heads * dk, bias=False); b.Wk = nn.Linear(d, heads * dk, bias=False)
            b.Wv = nn.Linear(d, heads * dk, bias=False); b.Wo = nn.Linear(heads * dk, d, bias=False)
            b.W1 = nn.Linear(d, hidden); b.W2 = nn.Linear(hidden, d)
            if norm:
                b.n1, b.n2 = RMS(d), RMS(d)
            self.blocks.append(b)
        self.out = nn.Linear(d, n_out)
        if norm:
            self.nf = RMS(d)

    def embed(self, tok):  # tok: (B, N, 3) long
        self.rc = tok[0, :, :2]
        return self.E_row[tok[..., 0]] + self.E_col[tok[..., 1]] + self.E_char[tok[..., 2]]

    def block(self, h, return_attn=False):
        att = None
        for b in self.blocks:
            x = b.n1(h) if self.norm else h
            B, N, _ = x.shape
            q = b.Wq(x).view(B, N, self.H, self.dk).transpose(1, 2)
            k = b.Wk(x).view(B, N, self.H, self.dk).transpose(1, 2)
            v = b.Wv(x).view(B, N, self.H, self.dk).transpose(1, 2)
            if self.attn == "gather":   # oracle routing: head h reads exactly neighbour h (D, L, R, U)
                a = self.gather_matrix(N, x.device, x.dtype).unsqueeze(0).expand(B, -1, -1, -1)
            else:
                s = q @ k.transpose(-1, -2) / math.sqrt(self.dk)
                if self.attn == "relbias":
                    oi = self.offset_index()
                    s = s + self.relb[:, oi].unsqueeze(0)
                    if self.local:   # hard window: keys more than K rows/cols away get zero attention
                        s = s.masked_fill((oi == (2 * self.K + 1) ** 2)[None, None], float("-inf"))
                a = torch.softmax(s, -1)
            att = a if return_attn else None
            o = (a @ v).transpose(1, 2).reshape(B, N, -1)
            h = h + b.Wo(o)
            x = b.n2(h) if self.norm else h
            h = h + b.W2(F.relu(b.W1(x)))
        return (h, att) if return_attn else h

    def offset_index(self):
        K = self.K
        dr = self.rc[None, :, 0] - self.rc[:, None, 0]; dc = self.rc[None, :, 1] - self.rc[:, None, 1]
        idx = (dr + K) * (2 * K + 1) + (dc + K)
        far = (dr.abs() > K) | (dc.abs() > K)
        return torch.where(far, torch.full_like(idx, (2 * K + 1) ** 2), idx)

    def gather_matrix(self, N, dev, dt):
        dr = self.rc[None, :, 0] - self.rc[:, None, 0]; dc = self.rc[None, :, 1] - self.rc[:, None, 1]
        A = torch.zeros(4, N, N, device=dev, dtype=dt)
        for h, (a_, b_) in enumerate([(1, 0), (0, -1), (0, 1), (-1, 0)]):
            A[h] = ((dr == a_) & (dc == b_)).to(dt)
        # border cells with no neighbour: read themselves
        none = A.sum(-1) == 0
        A[none] = torch.eye(N, device=dev, dtype=dt).expand(4, N, N)[none]
        return A

    def readout(self, h):
        return self.out(self.nf(h) if self.norm else h)

    def forward(self, tok, R, keep=False):
        h = self.embed(tok)
        traj = [h] if keep else None
        for _ in range(R):
            h = self.block(h)
            if keep: traj.append(h)
        return (self.readout(h), traj) if keep else self.readout(h)


def load_written(path=None, dtype=torch.float64):
    """Load the hand-written weights (sparse format of weights/run.py). Default: this repository's weights/networks/maze.npz."""
    z = np.load(path or WEIGHTS, allow_pickle=False)
    def dense(k):
        shape = tuple(int(s) for s in z[k + ".shape"])
        a = np.zeros(math.prod(shape)); a[z[k + ".idx"].astype(np.int64)] = z[k + ".val"]
        return torch.tensor(a.reshape(shape), dtype=dtype)
    m = Looped().to(dtype)
    b = m.blocks[0]
    with torch.no_grad():
        m.E_row.copy_(dense("emb.row")); m.E_col.copy_(dense("emb.col")); m.E_char.copy_(dense("emb.char"))
        b.Wq.weight.copy_(dense("L0.Wq").T); b.Wk.weight.copy_(dense("L0.Wk").T)
        b.Wv.weight.copy_(dense("L0.Wv").T); b.Wo.weight.copy_(dense("L0.Wo").T)
        b.W1.weight.copy_(dense("L0.W1").T); b.W1.bias.copy_(dense("L0.b1"))
        b.W2.weight.copy_(dense("L0.W2").T); b.W2.bias.copy_(dense("L0.b2"))
        m.out.weight.copy_(dense("out.W").T); m.out.bias.copy_(dense("out.b"))
    return m
