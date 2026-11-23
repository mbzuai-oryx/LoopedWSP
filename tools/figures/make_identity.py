"""Section 6 figure: a trained model hunting for what its own numbers mean, against the written maze transformer,
which carries an ID card that Claude wrote. Writes an inline <svg> for index.html.
Usage: python3 -I make_identity.py <out.html>"""
import math, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_algos_bars import claude_villain  # noqa: E402  (the scheming Claude from the concept figure)

ORANGE, ORANGE_T, DARK, INK, MUTED = "#eb6834", "#c4471a", "#1d1d1f", "#14213d", "#5b6475"
W, H = 540, 560


def robot(cx, cy, happy):
    """A transformer block with a face. Unhappy: dizzy spiral eyes, wobbly mouth, a sweat drop."""
    o = []
    body = "#ffffff" if happy else "#c9d0db"
    edge = ORANGE if happy else "#6b7383"
    o.append(f'<line x1="{cx}" y1="{cy - 58}" x2="{cx}" y2="{cy - 74}" stroke="{edge}" stroke-width="3"/>')
    o.append(f'<circle cx="{cx}" cy="{cy - 79}" r="7" fill="{ORANGE if happy else "#e9c46a"}" stroke="{edge}" stroke-width="2"/>')
    o.append(f'<rect x="{cx - 48}" y="{cy - 58}" width="96" height="112" rx="20" fill="{body}" stroke="{edge}" stroke-width="3"/>')
    # the block's insides, as on a transformer diagram: attention and MLP
    o.append(f'<rect x="{cx - 34}" y="{cy + 12}" width="68" height="14" rx="4" fill="{"#fde3d6" if happy else "#e4e8ee"}"/>')
    o.append(f'<rect x="{cx - 34}" y="{cy + 30}" width="68" height="14" rx="4" fill="{"#fde3d6" if happy else "#e4e8ee"}"/>')
    o.append(f'<text x="{cx}" y="{cy + 23}" text-anchor="middle" class="id-tiny">attention</text>')
    o.append(f'<text x="{cx}" y="{cy + 41}" text-anchor="middle" class="id-tiny">MLP</text>')
    ex = (cx - 18, cx + 18)
    if happy:
        for x in ex:
            o.append(f'<path d="M{x - 8},{cy - 22} q8,-11 16,0" fill="none" stroke="{DARK}" stroke-width="3" stroke-linecap="round"/>')
        o.append(f'<path d="M{cx - 14},{cy - 6} q14,13 28,0" fill="none" stroke="{DARK}" stroke-width="3" stroke-linecap="round"/>')
        for x in (cx - 33, cx + 33):
            o.append(f'<ellipse cx="{x}" cy="{cy - 9}" rx="6" ry="3.5" fill="#f6a58a" opacity=".8"/>')
    else:
        for x in ex:
            pts = []
            for k in range(46):
                a = k * 0.42
                r = 0.9 + k * 0.2
                pts.append(f"{x + r * math.cos(a):.1f},{cy - 24 + r * math.sin(a):.1f}")
            o.append(f'<circle cx="{x}" cy="{cy - 24}" r="11" fill="#fff" stroke="{DARK}" stroke-width="1.6"/>')
            o.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{DARK}" stroke-width="1.5"/>')
        o.append(f'<path d="M{cx - 15},{cy - 2} q5,-5 10,0 t10,0 t10,0" fill="none" stroke="{DARK}" stroke-width="2.6" stroke-linecap="round"/>')
        o.append(f'<path d="M{cx + 44},{cy - 58} q-7,10 0,14 q7,-4 0,-14 z" fill="#8fc8ea"/>')
    return "".join(o)


def tangle(x0, y0, w, h, seed=4):
    """A knot of unlabeled weights: tangled curves with stray numbers."""
    r = random.Random(seed)
    o = []
    for k in range(22):
        pts = [(x0 + r.random() * w, y0 + r.random() * h) for _ in range(4)]
        d = f"M{pts[0][0]:.0f},{pts[0][1]:.0f} C{pts[1][0]:.0f},{pts[1][1]:.0f} {pts[2][0]:.0f},{pts[2][1]:.0f} {pts[3][0]:.0f},{pts[3][1]:.0f}"
        o.append(f'<path d="{d}" fill="none" stroke="{["#8d96a6", "#a9b1be", "#6b7383"][k % 3]}" stroke-width="1.3" opacity=".8"/>')
    nums = ["0.137", "−2.41", "0.008", "1.92", "−0.55", "3e−4", "0.71", "−1.06", "4.2", "−0.03", "0.29", "−7.8"]
    for k, s in enumerate(nums):
        x, y = x0 + 8 + r.random() * (w - 40), y0 + 14 + r.random() * (h - 20)
        o.append(f'<text x="{x:.0f}" y="{y:.0f}" class="id-num" transform="rotate({r.uniform(-25, 25):.0f} {x:.0f} {y:.0f})">{s}</text>')
    return "".join(o)


def main(out):
    o = []
    # ---------------- panel A: trained
    o.append(f'<rect x="0" y="0" width="{W}" height="262" rx="18" fill="#eef0f3"/>')
    o.append(f'<text x="22" y="34" class="id-title" fill="{INK}">Trained by gradient descent</text>')
    o.append(f'<text x="22" y="54" class="id-sub">Its numbers get their meaning from training</text>')
    o.append(robot(98, 176, happy=False))
    o.append(tangle(262, 110, 200, 130))
    # magnifying glass held over the tangle
    o.append(f'<line x1="146" y1="168" x2="226" y2="168" stroke="#6b7383" stroke-width="7" stroke-linecap="round"/>')
    o.append(f'<line x1="226" y1="168" x2="266" y2="162" stroke="{DARK}" stroke-width="8" stroke-linecap="round"/>')
    o.append(f'<circle cx="304" cy="160" r="36" fill="#ffffff" fill-opacity=".55" stroke="{DARK}" stroke-width="5"/>')
    o.append(f'<text x="304" y="171" text-anchor="middle" class="id-qbig" fill="{ORANGE_T}">?</text>')
    # thought bubble
    o.append(f'<path d="M206,64 h300 a14,14 0 0 1 14,14 v20 a14,14 0 0 1 -14,14 h-300 a14,14 0 0 1 -14,-14 v-20 a14,14 0 0 1 14,-14 z" fill="#fff" stroke="{DARK}" stroke-width="1.6"/>')
    for (x, y, r) in ((180, 118, 6), (166, 132, 4)):
        o.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="#fff" stroke="{DARK}" stroke-width="1.4"/>')
    o.append(f'<text x="356" y="85" text-anchor="middle" class="id-bubble">Which of my millions of numbers</text>')
    o.append(f'<text x="356" y="103" text-anchor="middle" class="id-bubble">is the distance? Who am I?</text>')
    for (x, y, s, rot) in ((488, 228, 30, 14), (30, 96, 26, -14), (232, 232, 20, -8)):
        o.append(f'<text x="{x}" y="{y}" font-size="{s}" font-weight="800" fill="#8d96a6" transform="rotate({rot} {x} {y})" class="id-q">?</text>')
    # ---------------- panel B: written
    y0 = 282
    o.append(f'<rect x="0" y="{y0}" width="{W}" height="{H - y0}" rx="18" fill="#fdeee6"/>')
    o.append(f'<text x="22" y="{y0 + 34}" class="id-title" fill="{ORANGE_T}">Written by Claude</text>')
    o.append(f'<text x="22" y="{y0 + 54}" class="id-sub">Its numbers got their meaning before any weight was set</text>')
    o.append(robot(98, y0 + 168, happy=True))
    # lanyard to the ID card
    o.append(f'<path d="M126,{y0 + 132} C170,{y0 + 116} 190,{y0 + 118} 214,{y0 + 108}" fill="none" stroke="{ORANGE}" stroke-width="3"/>')
    cx0, cy0, cw, ch = 214, y0 + 70, 306, 196
    o.append(f'<rect x="{cx0 + 4}" y="{cy0 + 5}" width="{cw}" height="{ch}" rx="12" fill="#e9cdbf"/>')
    o.append(f'<rect x="{cx0}" y="{cy0}" width="{cw}" height="{ch}" rx="12" fill="#fff" stroke="{DARK}" stroke-width="1.6"/>')
    o.append(f'<path d="M{cx0},{cy0 + 12} a12,12 0 0 1 12,-12 h{cw - 24} a12,12 0 0 1 12,12 v20 h-{cw} z" fill="{ORANGE}"/>')
    o.append(f'<text x="{cx0 + 14}" y="{cy0 + 22}" class="id-card-h">ID · the maze transformer, each cell</text>')
    rows = [("7 numbers", "my position, wall, start or goal"),
            ("3 numbers", "the distance wave"),
            ("4 numbers", "which neighbor is my next step"),
            ("1 number", "am I on the path?"),
            ("20 numbers", "scratch space for what I read"),
            ("4 heads", "read the cell below, left, right, above")]
    for k, (a, b) in enumerate(rows):
        yy = cy0 + 52 + k * 25
        o.append(f'<rect x="{cx0 + 12}" y="{yy - 12}" width="74" height="18" rx="9" fill="#fde3d6"/>')
        o.append(f'<text x="{cx0 + 49}" y="{yy + 1}" text-anchor="middle" class="id-chip">{a}</text>')
        o.append(f'<text x="{cx0 + 94}" y="{yy + 1}" class="id-row">{b}</text>')
    # Claude, who wrote the card, with a pen
    vx, vy = 488, y0 + 30
    o.append(claude_villain(vx, vy, 0.5))
    o.append(f'<line x1="{vx - 16}" y1="{vy + 14}" x2="{vx - 40}" y2="{vy + 40}" stroke="{DARK}" stroke-width="4" stroke-linecap="round"/>')
    o.append(f'<path d="M{vx - 40},{vy + 40} l-5,9 l9,-4 z" fill="{DARK}"/>')
    svg = (f'<svg class="id-svg" viewBox="0 0 {W} {H}" role="img" aria-label="Two cartoon panels. Top, a gray transformer block '
           f'trained by gradient descent has dizzy spiral eyes and holds a magnifying glass over a tangle of unlabeled numbers, '
           f'thinking: which of my millions of numbers is the distance, who am I? Bottom, the maze transformer Claude wrote is '
           f'smiling and wears an ID card, written by Claude, that says what each of a cell\'s numbers stands for: 7 numbers for '
           f'where the cell is and whether it is a wall, start or goal, 3 for the distance wave, 4 for its next step, 1 for being '
           f'on the path, 20 for scratch space, and 4 heads that read the four neighbors.">' + "".join(o) + "</svg>")
    open(out, "w", encoding="utf-8").write(svg)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
