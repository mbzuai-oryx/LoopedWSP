import numpy as np, torch
from maze import make_dataset, trajectory_w
from wsp import load_written, maze_tokens
m = load_written(None)
bad = 0
for n in [11, 15]:
    for it in make_dataset(50, n, seed=100+n):
        T = 2*it["dS"]+6
        tok = torch.tensor(maze_tokens(it["wall"], it["s"], it["g"]))[None]
        with torch.no_grad(): _, tr = m(tok, T, keep=True)
        H = torch.stack(tr)[:,0,:,7:15].numpy()
        bad += (np.round(H) != trajectory_w(it, T)).any()
        assert np.abs(torch.stack(tr)[-1,0,:,15:].numpy()).max() < 1e-6
print("mismatching mazes:", bad)
