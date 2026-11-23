# More transformers Claude wrote

Each transformer here was written by a fresh Claude Code session (Claude Opus 5.5) from a plain description of the
task, which is in [`prompts/`](../../prompts/). Each one ran once on its test set.

| Folder | Task | Test result |
|---|---|---|
| `sudoku_extreme/` | Sudoku-Extreme | 141,040 of 141,040 puzzles solved |
| `sudoku_loop_limit/` | Sudoku-Extreme, written with at most 1,000 loops | 10,000 of the first 10,000 solved |
| `graphwalks/` | GraphWalks, prompts of 256K to 1M tokens | 100% F1 on 200 problems |
| `clrs_scc/` | CLRS-Text, strongly connected components | 9,300 of 9,300 (100%) |
| `clrs_mst_prim/` | CLRS-Text, minimum spanning tree (Prim) | 16,269 of 16,400 (99.2%); the other 131 are different minimum spanning trees of the same total weight |
| `clrs_lcs_length/` | CLRS-Text, longest common subsequence | 6,000 of 6,000 (100%) |
| `satlib/` | SATLIB random 3-SAT (uf20-91, uf50-218, uuf50-218) | 2,700 of 2,700 (100%) |
| `planbench_blocksworld/` | PlanBench Blocksworld | 400 of 400 valid plans |
| `planbench_mystery_blocksworld/` | PlanBench Mystery Blocksworld | 400 of 400 valid plans |
| `planbench_logistics/` | PlanBench Logistics | 200 of 200 valid plans |
| `maze_size_1k/` | Maze-Hard, written with at most 1,000 weights: 928 | 1,000 of 1,000 |
| `maze_size_5k/` | Maze-Hard, written with at most 5,000 weights: 4,359 | 1,000 of 1,000 |
| `maze_size_10k/` | Maze-Hard, written with at most 10,000 weights: 4,380 | 1,000 of 1,000 |
| `maze_size_20k/` | Maze-Hard, written with at most 20,000 weights: 7,682 | 1,000 of 1,000 |
| `maze_size_40k/` | Maze-Hard, written with at most 40,000 weights: 8,876 | 1,000 of 1,000 |

## Run

Each folder holds one transformer (`network.json`, `weights.npz`, `tokenizer.py`) and `example_input.txt`, one
problem from its dataset. To run a transformer, give the runner its folder and an input file:

```
cd weights/more_networks
python3 runner/looped.py <folder> <input file>
```

For example, each of these runs one transformer on its example input:

```
python3 runner/looped.py clrs_lcs_length clrs_lcs_length/example_input.txt
python3 runner/looped.py clrs_mst_prim clrs_mst_prim/example_input.txt
python3 runner/looped.py clrs_scc clrs_scc/example_input.txt
python3 runner/looped.py planbench_blocksworld planbench_blocksworld/example_input.txt
python3 runner/looped.py planbench_mystery_blocksworld planbench_mystery_blocksworld/example_input.txt
python3 runner/looped.py planbench_logistics planbench_logistics/example_input.txt
python3 runner/looped.py satlib satlib/example_input.txt
python3 runner/looped.py graphwalks graphwalks/example_input.txt
python3 runner/looped.py sudoku_extreme sudoku_extreme/example_input.txt
python3 runner/looped.py sudoku_loop_limit sudoku_loop_limit/example_input.txt
python3 runner/looped.py maze_size_1k maze_size_1k/example_input.txt
```

The runner prints the answer. It needs Python 3 and numpy. GraphWalks is scored on the set of nodes in the answer, and its
transformer writes each node once per 4-character piece of its ID, so every node appears four times. The CLRS-Text and PlanBench examples take seconds on a
laptop. Sudoku and the mazes run thousands of loops and take minutes; on a GPU, add `--backend torch64`.
