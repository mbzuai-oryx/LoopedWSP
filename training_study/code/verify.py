import numpy as np, torch, time
from maze import make_dataset
from wsp import load_written, maze_tokens
m = load_written(None)
for n in [11, 15, 21, 31]:
    ds = make_dataset(20, n, seed=n)
    ok = 0; t0 = time.time()
    for it in ds:
        tok = torch.tensor(maze_tokens(it["wall"], it["s"], it["g"]))[None]
        with torch.no_grad():
            pred = m(tok, 2 * n * n).argmax(-1)[0].numpy().reshape(n, n)
        tgt = (it["pidx"] >= 0)
        onpath = (pred == 4) | (pred == 2) | (pred == 3)
        ok += (onpath == tgt).all()
    print(n, ok, "/", len(ds), f"{time.time()-t0:.1f}s")
