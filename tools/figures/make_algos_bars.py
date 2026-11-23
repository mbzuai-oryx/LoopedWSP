"""Graph algorithms, SAT and planning as a bar chart (was Table 7), written as an inline <svg> for index.html.
Written transformers: every test item. Claude Opus 5.5: 50 test items per task in Claude Code with no tools; its bars
carry 95% Wilson intervals. The two Claude cartoons are the ones in tools/make_concept.py (copied here).
Usage: python3 -I make_algos_bars.py <out.html>"""
import math, random, sys

ORANGE, ORANGE_T, DARK, INK, MUTED, GRID = "#eb6834", "#c4471a", "#1d1d1f", "#14213d", "#5b6475", "#e1e0d9"
GREY_FILL, GREY_EDGE, GREY_TEXT = "#ecebe7", "#a9a8a1", "#5f5e59"
GROUPS = [
    ("CLRS-Text", "graph", [("Strongly connected components", 100, 96), ("Minimum spanning tree (Prim)", 99.2, 96),
                            ("Longest common subsequence", 100, 92)]),
    ("PlanBench", "blocks", [("Blocksworld", 100, 100), ("Mystery Blocksworld", 100, 100), ("Logistics", 100, 100)]),
    ("SATLIB, random 3-SAT", "sat", [("uf20-91, satisfiable", 100, 96), ("uf50-218, satisfiable", 100, 84),
                                    ("uuf50-218, unsatisfiable", 100, 82)]),
]
N_CLAUDE = 50


def wilson(p, n, z=1.96):
    k = p / 100
    den = 1 + z * z / n
    c = (k + z * z / (2 * n)) / den
    h = z * math.sqrt(k * (1 - k) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * min(1, c + h)


def spark(cx, cy, r, seed):
    rr = random.Random(seed)
    o = []
    for i in range(11):
        a = math.radians(i * 360 / 11 + rr.uniform(-6, 6) - 90)
        L = r * rr.uniform(0.82, 1.0)
        o.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{cx + L * math.cos(a):.1f}" y2="{cy + L * math.sin(a):.1f}" '
                 f'stroke="{ORANGE}" stroke-width="{r * 0.3:.1f}" stroke-linecap="round"/>')
    o.append(f'<circle cx="{cx}" cy="{cy}" r="{r * 0.5:.1f}" fill="{ORANGE}"/>')
    return "".join(o)


def claude_dumb(cx, cy, s):
    """The confused Claude from the concept figure, scaled by s (1 = radius 40)."""
    g = [f'<g transform="translate({cx},{cy}) scale({s})">', spark(0, 0, 40, 3)]
    g.append(f'<circle cx="-9" cy="-3" r="7" fill="#fff" stroke="{DARK}" stroke-width="1.6"/><circle cx="-12.5" cy="0.5" r="3" fill="{DARK}"/>')
    g.append(f'<circle cx="9" cy="-6" r="4.6" fill="#fff" stroke="{DARK}" stroke-width="1.6"/><circle cx="10.5" cy="-8" r="2" fill="{DARK}"/>')
    g.append(f'<path d="M-9,8 Q0,6 9,10" fill="none" stroke="{DARK}" stroke-width="2.4" stroke-linecap="round"/>')
    g.append(f'<path d="M1,9 q2,9 7,5 q2,-3 -1,-6 z" fill="#e8798f" stroke="{DARK}" stroke-width="1.2"/>')
    g.append('</g>')
    return "".join(g)


def claude_villain(cx, cy, s):
    """The scheming Claude with a monocle from the concept figure, scaled by s."""
    g = [f'<g transform="translate({cx},{cy}) scale({s})">', spark(0, 0, 40, 8)]
    g.append(f'<line x1="-17" y1="-14" x2="-4" y2="-8" stroke="{DARK}" stroke-width="3.4" stroke-linecap="round"/>')
    g.append(f'<line x1="17" y1="-14" x2="4" y2="-8" stroke="{DARK}" stroke-width="3.4" stroke-linecap="round"/>')
    for ex in (-9, 9):
        g.append(f'<ellipse cx="{ex}" cy="-1" rx="5.2" ry="3.6" fill="#fff" stroke="{DARK}" stroke-width="1.4"/><circle cx="{ex + 1.5}" cy="0" r="2.1" fill="{DARK}"/>')
    g.append(f'<circle cx="9" cy="-1" r="8" fill="none" stroke="{DARK}" stroke-width="1.6"/>')
    g.append(f'<path d="M-9,9 Q1,15 11,6" fill="none" stroke="{DARK}" stroke-width="2.6" stroke-linecap="round"/><path d="M11,6 l3,-3" stroke="{DARK}" stroke-width="2.6" stroke-linecap="round"/>')
    g.append('</g>')
    return "".join(g)


def pictogram(kind, x, y):
    """A small sign for each benchmark, in a 26 x 18 box with its left edge at x, centred on y."""
    s = f'stroke="{INK}" stroke-width="1.4" fill="none" stroke-linecap="round" stroke-linejoin="round"'
    if kind == "graph":
        pts = [(x + 3, y + 5), (x + 13, y - 6), (x + 23, y + 5)]
        o = [f'<path d="M{pts[0][0]},{pts[0][1]} L{pts[1][0]},{pts[1][1]} L{pts[2][0]},{pts[2][1]} Z" {s}/>']
        o += [f'<circle cx="{px}" cy="{py}" r="3.2" fill="#fff" stroke="{INK}" stroke-width="1.4"/>' for px, py in pts]
        return "".join(o)
    if kind == "blocks":
        return (f'<rect x="{x + 2}" y="{y + 1}" width="9" height="8" rx="1.5" fill="#d7e5f6" stroke="{INK}" stroke-width="1.3"/>'
                f'<rect x="{x + 2}" y="{y - 8}" width="9" height="8" rx="1.5" fill="#fbe0d3" stroke="{INK}" stroke-width="1.3"/>'
                f'<rect x="{x + 15}" y="{y + 1}" width="9" height="8" rx="1.5" fill="#fff" stroke="{INK}" stroke-width="1.3"/>'
                f'<path d="M{x + 12},{y - 6} q6,-5 8,4" {s}/><path d="M{x + 18},{y - 3} l2,1.6 l1.2,-2.4" {s}/>')
    return (f'<rect x="{x}" y="{y - 9}" width="27" height="18" rx="4" fill="#fff" stroke="{INK}" stroke-width="1.3"/>'
            f'<text x="{x + 13.5}" y="{y + 4.5}" text-anchor="middle" class="ab-sat">a∨b</text>')


def main(out):
    W = 540
    x_lab, x0, x100 = 14, 254, 478
    sc = (x100 - x0) / 100
    bh, gap, row_gap, head_h = 11, 3, 13, 30
    y = 88
    o = []
    # legend with the two cartoons
    o.append(claude_villain(30, 30, 0.62))
    o.append(f'<rect x="60" y="23" width="22" height="11" rx="2" fill="{ORANGE}"/>')
    o.append(f'<text x="88" y="33" class="ab-leg" fill="{ORANGE_T}" font-weight="700">Written by Claude</text>')
    o.append(claude_dumb(300, 30, 0.62))
    o.append(f'<rect x="330" y="23" width="22" height="11" rx="2" fill="url(#ab-hatch)" stroke="{GREY_EDGE}" stroke-width="0.8"/>')
    o.append(f'<text x="358" y="33" class="ab-leg" fill="{GREY_TEXT}">Claude Opus 5.5 alone</text>')
    o.append(f'<line x1="0" y1="62" x2="{W}" y2="62" stroke="{GRID}" stroke-width="1"/>')
    for gname, kind, rows in GROUPS:
        o.append(pictogram(kind, x_lab, y - 6))
        o.append(f'<text x="{x_lab + 34}" y="{y - 1}" class="ab-group">{gname}</text>')
        y += head_h - 14
        for label, ours, cl in rows:
            yc = y + bh + gap / 2
            o.append(f'<text x="{x_lab + 34}" y="{yc + 4}" class="ab-task">{label}</text>')
            o.append(f'<rect x="{x0}" y="{y}" width="{ours * sc:.1f}" height="{bh}" rx="2" fill="{ORANGE}" class="ab-bar"/>')
            o.append(f'<text x="{x0 + ours * sc + 5:.1f}" y="{y + bh - 1.5}" class="ab-val" fill="{ORANGE_T}" font-weight="700">{ours:g}%</text>')
            yb = y + bh + gap
            o.append(f'<rect x="{x0}" y="{yb}" width="{cl * sc:.1f}" height="{bh}" rx="2" fill="url(#ab-hatch)" stroke="{GREY_EDGE}" stroke-width="0.8" class="ab-bar"/>')
            lo, hi = wilson(cl, N_CLAUDE)
            ym = yb + bh / 2
            o.append(f'<path d="M{x0 + lo * sc:.1f},{ym} H{x0 + hi * sc:.1f} M{x0 + lo * sc:.1f},{ym - 3.5} V{ym + 3.5} M{x0 + hi * sc:.1f},{ym - 3.5} V{ym + 3.5}" '
                     f'stroke="#4d4c48" stroke-width="1.2" fill="none"/>')
            o.append(f'<text x="{x0 + hi * sc + 5:.1f}" y="{yb + bh - 1.5}" class="ab-val" fill="{GREY_TEXT}">{cl:g}%</text>')
            y += 2 * bh + gap + row_gap
        y += 16
    # axis
    ya = y - 10
    o.append(f'<line x1="{x0}" y1="{ya}" x2="{x100}" y2="{ya}" stroke="{GRID}" stroke-width="1"/>')
    for v in (0, 50, 100):
        o.append(f'<line x1="{x0 + v * sc}" y1="78" x2="{x0 + v * sc}" y2="{ya}" stroke="{GRID}" stroke-width="{1 if v else 0.8}" stroke-dasharray="{"" if v in (0,) else "2 3"}"/>')
        o.append(f'<text x="{x0 + v * sc}" y="{ya + 16}" text-anchor="middle" class="ab-axis">{v}</text>')
    o.append(f'<text x="{(x0 + x100) / 2}" y="{ya + 34}" text-anchor="middle" class="ab-axis">Accuracy (%)</text>')
    H = ya + 42
    defs = (f'<defs><pattern id="ab-hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
            f'<rect width="5" height="5" fill="{GREY_FILL}"/><line x1="0" y1="0" x2="0" y2="5" stroke="{GREY_EDGE}" stroke-width="1.6"/></pattern></defs>')
    svg = (f'<svg class="ab-svg" viewBox="0 0 {W} {H:.0f}" role="img" aria-label="Bar chart. On nine tasks from CLRS-Text, PlanBench and SATLIB, '
           f'the transformers Claude wrote score 100%, except 99.2% on minimum spanning trees, where every miss is a tie. Claude Opus 5.5, '
           f'asked directly, scores 96, 96 and 92% on CLRS-Text, 100% on all three PlanBench tasks, and 96, 84 and 82% on SATLIB.">'
           + defs + "".join(o) + "</svg>")
    open(out, "w", encoding="utf-8").write(svg)
    print("wrote", out, f"({W}x{H:.0f})")


if __name__ == "__main__":
    main(sys.argv[1])
