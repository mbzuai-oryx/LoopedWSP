# Figure and video sources

Scripts, data and fonts that draw the blog's charts, diagrams and maze videos, kept here so the figures can be rebuilt.

- `make_figures.py` draws the charts and diagrams (base/data SVG layers for the scroll-reveal) from `data.json` and the per-chart JSON files. Run `python3 make_figures.py --only maze_training` to redraw one figure.
- `make_maze_hero.py` and `make_maze_inside.py` render the two maze videos from `maze_trace.json` and `maze_inside_trace.json` (needs Pillow and ffmpeg).
- Fonts are Inter and Caveat, both under the SIL Open Font License (`fonts/OFL.txt`, `fonts/Caveat-OFL.txt`).

