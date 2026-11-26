import numpy as np, torch
from maze import make_dataset, trajectory
from wsp import load_written, maze_tokens
m = load_written(None)
it = make_dataset(1, 15, seed=3)[0]; n=15
tok = torch.tensor(maze_tokens(it["wall"], it["s"], it["g"]))[None]
T = 2*it["dS"]+2
with torch.no_grad(): _, tr = m(tok, T, keep=True)
H = torch.stack(tr)[:,0].numpy()   # (T+1, N, 35)
print("h0 at start cell:", np.round(H[0][it['s'][0]*n+it['s'][1]],2))
print("distinct values per dim over all loops:")
for d in range(35):
    u = np.unique(np.round(H[:,:,d],3)); print(d, len(u), u[:6], u[-3:] if len(u)>6 else "")
for shift in [0,1]:
  tgt = trajectory(it["dist"], it["ptr"], it["pidx"], T)
  best=[]
  for j in range(8):
    errs=[(np.abs(np.round(H[shift:,:,d])-tgt[:T+1-shift,:,j]).sum(),d) for d in range(35)]
    best.append(min(errs))
  print("shift",shift, best)
