# Weights

The weights of the transformers Claude wrote.

## What is here

- `networks/`: ten transformers, each as `<name>.npz` (the weights) and `<name>.json` (its shape and loop count).
- `run.py`: runs any of them. It needs only Python 3 and numpy.
- [`more_networks/`](more_networks/): the other transformers Claude wrote (Sudoku-Extreme, GraphWalks, CLRS-Text,
  SATLIB, PlanBench and the maze transformers written under a cap on the number of weights), with their own runner.

| Name | Task |
|---|---|
| `maze` | Maze-Hard: the shortest path through a maze |
| `hanoi` | Tower of Hanoi |
| `checker` | Checker Jumping |
| `boolean_expressions` | BIG-Bench Hard, Boolean Expressions |
| `dyck_languages` | BIG-Bench Hard, Dyck Languages |
| `web_of_lies` | BIG-Bench Extra Hard, Web of Lies |
| `shuffled_objects` | BIG-Bench Extra Hard, Shuffled Objects |
| `graph` | Graph search: the shortest-path length between two nodes |
| `parity` | Parity of a bit string |
| `chains` | Reasoning chains: the value at the end of a chain of assignments |

## Run

```
python run.py maze --random 30
python run.py maze maze.txt
python run.py hanoi 10
python run.py checker 5
python run.py boolean_expressions "( ( False ) ) or ( True ) is"
python run.py dyck_languages "Complete the rest of the sequence, making sure that the parentheses are closed properly. Input: [ { [ ( ) ] }"
python run.py web_of_lies question.txt
python run.py shuffled_objects question.txt
python run.py graph "0>1 1>2 2>3 5>0 ? 0 3"
python run.py parity 0110101
python run.py chains "x3 = 7; x5 = x3 + 4; x9 = x5 - 2; ? x9"
```

Each command prints the answer and checks it. `--random n` (maze, graph, parity, chains) makes a random input of
size n and runs it. A maze file has one row per line, with `#` for walls, spaces for open cells, and `S` and `G`.
On a laptop, a 30×30 maze takes about 30 seconds and 10 Hanoi disks about a second.
