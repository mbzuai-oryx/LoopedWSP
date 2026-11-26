# Training study: the copy plateau

Why gradient descent can't find the maze transformer in this repository, even with every loop supervised, and the training changes that make the same architecture solve every test maze up to 61×61.

Section 4 of the [blog post](https://mbzuai-oryx.github.io/LoopedWSP/) reports that the maze transformer's architecture, reset to random weights and trained on the 1,000 training mazes, solved 0% in every run, including runs that supervised the distance wave and pointers after every loop. This folder takes `weights/networks/maze.npz` apart to find out why, and tests fixes for each failure it finds.

The full write-up, with figures drawn from the data, is [`writeup.html`](writeup.html).

## Findings

1. **The 0% needs no long horizon.** Given the true state at every loop and asked only for the next one, the block learns to copy its input: 99.3% of cells right, 0% of the cells that change. Full backpropagation through the loop ends in the same place.
2. **Three things keep the step from being learned.**
   - Attention starts with almost no gradient toward the neighbour cells, and none that tells the four heads apart.
   - The usual answer loss, at full weight, holds the model on the copy solution.
   - Positions as lookup tables are slow to learn, and the written raw coordinates saturate attention from the first step.
3. **A learned step is not a working loop.** Small errors in the 27 unsupervised numbers per cell grow every loop, by about 1,000× over 10 loops. The written block works because it is exact, and it is fragile in one place: 0.1% noise on its position tables breaks about 40% of mazes.
4. **Rounding inside the loop fixes stability.** Straight-through gradients through the rounding diverged in all three runs we tried. Training each loop on the block's own 8-loop rollouts, with no gradient across loops, worked.
5. **Two more fixes close the gap.**
   - A local 5×5 attention window stops soft attention from leaking more weight to far cells as the grid grows.
   - The last 20% of failures were all mazes with ties, where the block had learned a distance shortcut instead of following the pointer. A margin penalty, summed over near-threshold bits, fixed them.

Trained from scratch with that recipe, three seeds each solve:

| | 15×15 | 2× loops | 21×21 | 31×31 | 61×61 | fresh 15×15 |
|---|---|---|---|---|---|---|
| Full recipe, seeds 0, 1, 2 | 100% | 100% | 100% | 30/30 | 10/10 | 100/100 |
| without the margin penalty (seed 0) | 80% | 80% | 69% | 19/30 | — | — |
| without the local window (seed 0) | 100% | 100% | 89% | 0/30 | — | — |
| without first-step oversampling (seed 0) | 100% | 100% | 100% | 30/30 | — | — |

Every solved maze is exact at every loop. The block is the paper's architecture (one block, 35 numbers per cell, 4 heads, 86 ReLU neurons), with a learned relative-position bias inside the window in place of absolute positions. It is trained only on 15×15 mazes.

**What these results are not.** Every loop is supervised with the written block's exact 8-bit state, so this is imitation of a known step, not discovery of the algorithm. The window is a hand-chosen, convolution-like prior, and much of the size transfer is built in. The ablations are one seed each. Everything ran on a laptop CPU, on 11×11 to 15×15 mazes from our own generator rather than Maze-Hard. The write-up lists every caveat.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd training_study/code         # all scripts read ../../weights/networks/maze.npz
python verify.py               # the written weights solve generated mazes (port check)
python check_traj.py           # per-loop targets match the written block at every loop
```

Train the full recipe from scratch (about 1 hour on a laptop CPU):

```bash
python train_disc.py --mode unroll --k 8 --no_st --attn relbias --pos none --local --focus 0.5 --hinge 1 \
  --steps 8000 --eval_every 2000 --lr 2e-3 --cosine --seed 0 --name S0_scratch_full
python eval_disc.py results/S0_scratch_full.pt 31 30           # 31x31, 30 mazes
python eval_disc.py results/S0_scratch_full.pt 61 10 1 1       # 61x61, 10 mazes, batch 1
python eval_disc.py results/S0_scratch_full.pt 15 100 1 10 123 # fresh 15x15 mazes, seed 123
```

The exact command for every run is in [`RUNS.md`](RUNS.md).

## Layout

| Path | Contents |
|---|---|
| `writeup.html` | The write-up, with figures drawn from the data |
| `CLAIMS.md` | Every number in the write-up, the file it comes from, and the corrections from three independent audits |
| `RUNS.md` | Every training run, its command and its last evaluation (rebuild with `python code/build_runs.py`) |
| `code/maze.py` | Maze generator, breadth-first search, and the written block's exact state after every loop (`trajectory_w`) |
| `code/wsp.py` | PyTorch port of the generic loop in `weights/run.py`, loading `weights/networks/maze.npz`, with options for relative-bias, local-window and oracle-routing attention |
| `code/train.py` | Parts 1–4 of the write-up: answer-only, full-rollout and teacher-forced training |
| `code/train_disc.py` | Parts 6–7: the rounding layer in the loop, own-rollout training, window, mixed sizes, oversampling, margin penalty |
| `code/eval_disc.py`, `probe_disc.py`, `fails.py`, `leak.py` | Free-rollout evaluation, error breakdowns, failing-maze analysis, attention leak |
| `code/diag*.py`, `sat.py`, `analyze_step.py`, `errors.py`, `trace.py`, `rollout_check.py` | Diagnostics behind sections 2–5 |
| `code/results/` | Settings and evaluations of every run (`*.json`), extra evaluations (`eval_disc.jsonl`), trained weights (`*.pt`) |
| `code/logs/` | Training logs with evaluation lines |

## Credits

Built on the maze transformer, weights and generic loop in this repository. The experiments and write-up were done with Claude Opus 5.5 in Claude Code. Everything here is covered by the repository's MIT license.
