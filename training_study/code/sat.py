import math, numpy as np, torch
from maze import make_dataset
from wsp import Looped, load_written, maze_tokens
it = make_dataset(1, 15, seed=3)[0]; tok = torch.tensor(maze_tokens(it["wall"], it["s"], it["g"]))[None]
W = load_written(None, dtype=torch.float32)
for name in ["learned (default init)", "written coordinates, frozen"]:
    vals = []
    for seed in range(5):
        torch.manual_seed(seed); m = Looped()
        if "written" in name:
            with torch.no_grad(): m.E_row.copy_(W.E_row); m.E_col.copy_(W.E_col)
        with torch.no_grad():
            _, A = m.block(m.embed(tok), return_attn=True)
            q = m.blocks[0].Wq(m.embed(tok)); 
        vals.append((A.max(-1).values.mean().item(), (A.max(-1).values > 0.99).float().mean().item()))
    v = np.array(vals).mean(0)
    print(f"{name}: mean max attention weight {v[0]:.3f}, fraction of rows >99% on one cell {v[1]:.3f} (uniform = {1/225:.4f})")
