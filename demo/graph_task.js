// Interactive figure for the graph search task (section "Generalization beyond the checked sizes").
// It draws a map in the style of the task's data: points joined by one-way links, with dead ends and decoys, a start
// point A and a goal B. Sizes follow the task: 7-12 points and routes of 1-4 steps in the checking range (1 in 4 maps
// with B unreachable), 24-48 points and routes of 8-16 steps in the test set.
// Each loop applies the rule the written transformer follows: every visited point passes "visited" along its
// outgoing links. B is visited after as many loops as the shortest route has steps; if a loop adds nothing new,
// B cannot be reached. This is the rule, simulated, not the transformer's weights.
(function () {
  "use strict";
  const root = document.getElementById("gt");
  if (!root) return;

  const W = 520, H = 330, PAD = 22;
  const ORANGE = "#eb6834", ORANGE_TEXT = "#c4471a", INK = "#14213d", MUTED = "#5b6475", LINE = "#b8b6ae";
  const reduce = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;
  const SVGNS = "http://www.w3.org/2000/svg";

  // ------------------------------------------------------------------ seeded random numbers
  function rng(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const randint = (r, lo, hi) => lo + Math.floor(r() * (hi - lo + 1));

  // ------------------------------------------------------------------ maps
  function bfs(n, edges, a) {
    const out = Array.from({ length: n }, () => []);
    edges.forEach(([u, v]) => out[u].push(v));
    const d = new Array(n).fill(-1);
    d[a] = 0;
    const q = [a];
    while (q.length) {
      const u = q.shift();
      out[u].forEach((v) => { if (d[v] < 0) { d[v] = d[u] + 1; q.push(v); } });
    }
    return d;
  }

  function makeMap(seed, big) {
    for (let tries = 0; tries < 200; tries++) {
      const r = rng(seed * 7919 + tries);
      const n = big ? randint(r, 24, 48) : randint(r, 7, 12);
      const dist = big ? randint(r, 8, Math.min(16, n - 4)) : randint(r, 1, Math.min(4, n - 2));
      const unreachable = !big && r() < 0.25;
      // the route A = 0, 1, ..., dist = B
      const edges = [];
      const has = new Set();
      const add = (u, v) => { if (u === v || has.has(u + ">" + v) || has.has(v + ">" + u)) return false; edges.push([u, v]); has.add(u + ">" + v); return true; };
      for (let i = 0; i < dist; i++) add(i, i + 1);
      // the other points: branches off the route (decoys and dead ends) and points that only lead into it
      for (let v = dist + 1; v < n; v++) {
        const kind = r();
        if (kind < 0.7) add(randint(r, 0, v - 1), v);        // reachable side point
        else add(v, randint(r, 0, v - 1));                   // a point that only leads in
      }
      // extra links, kept only when they leave the shortest route at `dist` steps
      const extra = Math.round(n * (big ? 0.55 : 0.45));
      for (let k = 0; k < extra * 4 && edges.length < n - 1 + extra; k++) {
        const u = randint(r, 0, n - 1), v = randint(r, 0, n - 1);
        if (!add(u, v)) continue;
        if (bfs(n, edges, 0)[dist] !== dist) { edges.pop(); has.delete(u + ">" + v); }
      }
      if (unreachable) {                                     // turn one link of the route around
        const i = randint(r, 0, dist - 1);
        const j = edges.findIndex(([u, v]) => u === i && v === i + 1);
        edges.splice(j, 1); has.delete(i + ">" + (i + 1));
        add(i + 1, i);
        if (bfs(n, edges, 0)[dist] >= 0) continue;
      } else if (bfs(n, edges, 0)[dist] !== dist) continue;
      return { n, edges, A: 0, B: dist, seed };
    }
    return null;
  }

  // ------------------------------------------------------------------ layout (a small force layout, A left, B right)
  function layout(map, seed) {
    const r = rng(seed * 31 + 5);
    const { n, edges, A, B } = map;
    const d = bfs(n, edges, A);
    const dmax = Math.max(1, ...d);
    const P = Array.from({ length: n }, (_, i) => ({
      x: d[i] >= 0 ? PAD + (W - 2 * PAD) * (0.08 + 0.84 * d[i] / dmax) + (r() - 0.5) * 40 : PAD + r() * (W - 2 * PAD),
      y: PAD + r() * (H - 2 * PAD),
    }));
    P[A] = { x: PAD + 14, y: H / 2 };
    P[B] = { x: W - PAD - 14, y: H / 2 };
    const k = 0.78 * Math.sqrt((W - 2 * PAD) * (H - 2 * PAD) / n);
    for (let it = 0, temp = 40; it < 420; it++, temp *= 0.992) {
      const F = P.map(() => ({ x: 0, y: 0 }));
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
        let dx = P[i].x - P[j].x, dy = P[i].y - P[j].y;
        const dd = Math.max(1, Math.hypot(dx, dy)), f = k * k / dd;
        dx /= dd; dy /= dd;
        F[i].x += dx * f; F[i].y += dy * f; F[j].x -= dx * f; F[j].y -= dy * f;
      }
      edges.forEach(([u, v]) => {
        let dx = P[u].x - P[v].x, dy = P[u].y - P[v].y;
        const dd = Math.max(1, Math.hypot(dx, dy)), f = dd * dd / k;
        dx /= dd; dy /= dd;
        F[u].x -= dx * f; F[u].y -= dy * f; F[v].x += dx * f; F[v].y += dy * f;
      });
      // keep points off links they are not part of, and pull gently toward the middle
      edges.forEach(([u, v]) => {
        const ax = P[u].x, ay = P[u].y, bx = P[v].x - ax, by = P[v].y - ay, L2 = bx * bx + by * by || 1;
        for (let w = 0; w < n; w++) {
          if (w === u || w === v) continue;
          const s = Math.max(0, Math.min(1, ((P[w].x - ax) * bx + (P[w].y - ay) * by) / L2));
          const px = P[w].x - (ax + s * bx), py = P[w].y - (ay + s * by), dd = Math.hypot(px, py) || 0.01;
          if (dd < 18) { F[w].x += px / dd * (18 - dd) * 3; F[w].y += py / dd * (18 - dd) * 3; }
        }
      });
      for (let i = 0; i < n; i++) { F[i].x += (W / 2 - P[i].x) * 0.04; F[i].y += (H / 2 - P[i].y) * 0.08; }
      for (let i = 0; i < n; i++) {
        if (i === A || i === B) continue;
        const m = Math.hypot(F[i].x, F[i].y) || 1, s = Math.min(m, temp) / m;
        P[i].x = Math.min(W - PAD, Math.max(PAD, P[i].x + F[i].x * s));
        P[i].y = Math.min(H - PAD, Math.max(PAD, P[i].y + F[i].y * s));
      }
    }
    return P;
  }

  // ------------------------------------------------------------------ the page
  root.innerHTML =
    '<svg class="gt-svg" viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="A map of points joined by one-way links, with a start point A and a goal B"></svg>' +
    '<p class="gt-status" aria-live="polite"></p>' +
    '<div class="gt-ctrl">' +
    '<button type="button" class="gt-go">▶ Play</button>' +
    '<button type="button" class="gt-step">Step</button>' +
    '<button type="button" class="gt-reset">Reset</button>' +
    '<button type="button" class="gt-new">New map</button>' +
    '<span class="gt-size" role="group" aria-label="Map size">' +
    '<button type="button" class="gt-small" aria-pressed="false">Checked size</button>' +
    '<button type="button" class="gt-big on" aria-pressed="true">Test size</button></span></div>' +
    '<p class="gt-key"><span><i class="k-a"></i>start A</span><span><i class="k-b"></i>goal B</span>' +
    '<span><i class="k-v"></i>visited, with the loop it was visited in</span><span><i class="k-n"></i>not visited yet</span></p>' +
    '<p class="gt-tokens"></p>';
  const svg = root.querySelector(".gt-svg"), status = root.querySelector(".gt-status");
  const goBtn = root.querySelector(".gt-go"), tokens = root.querySelector(".gt-tokens");
  const smallBtn = root.querySelector(".gt-small"), bigBtn = root.querySelector(".gt-big");

  let big = true, seed = 1, map, P, B, visitedAt, loop, done, timer = null;   // opens on a test-size map
  const el = (name, attrs, parent) => {
    const e = document.createElementNS(SVGNS, name);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  };

  function newMap() {
    map = makeMap(seed, big);
    P = layout(map, seed);
    B = map.B;
    reset();
  }

  function reset() {
    stop();
    visitedAt = new Array(map.n).fill(-1);
    visitedAt[map.A] = 0;
    loop = 0;
    done = false;
    tokens.textContent = "The transformer reads this map as " + map.n + " point tokens, " + map.edges.length +
      " link tokens and one question token, A to B, in shuffled order.";
    draw([]);
    say("Loop 0. Only A is visited.");
  }

  function step() {
    if (done) return;
    loop += 1;
    const fresh = [], used = [];
    map.edges.forEach(([u, v], i) => {
      if (visitedAt[u] >= 0 && visitedAt[u] < loop) {
        used.push(i);
        if (visitedAt[v] < 0 && fresh.indexOf(v) < 0) fresh.push(v);
      }
    });
    fresh.forEach((v) => { visitedAt[v] = loop; });
    const total = visitedAt.filter((x) => x >= 0).length;
    if (visitedAt[B] === loop) {
      done = true;
      say("Loop " + loop + ". B is visited, so the shortest route from A to B takes " + loop + (loop === 1 ? " step" : " steps") +
        ". Further loops change nothing.");
    } else if (!fresh.length) {
      done = true;
      say("Loop " + loop + ". No new point was visited, so B cannot be reached from A. Further loops change nothing.");
    } else {
      say("Loop " + loop + ". " + fresh.length + (fresh.length === 1 ? " new point" : " new points") + ", " + total + " visited in all.");
    }
    draw(fresh);
    if (done) { stop(); auto.done(); }
  }

  function say(s) { status.textContent = s; }

  function play() {
    if (done) reset();
    goBtn.textContent = "❚❚ Pause";
    goBtn.classList.add("on");
    const tick = () => { step(); if (!done) timer = setTimeout(tick, reduce ? 900 : 650); };
    timer = setTimeout(tick, 250);
  }
  function stop() {
    if (timer) clearTimeout(timer);
    timer = null;
    goBtn.textContent = "▶ Play";
    goBtn.classList.remove("on");
  }

  function draw(fresh) {
    svg.textContent = "";
    const defs = el("defs", {}, svg);
    [["gt-ah", LINE], ["gt-ah-on", ORANGE]].forEach(([id, col]) => {
      const m = el("marker", { id, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 5.5, markerHeight: 5.5, orient: "auto-start-reverse" }, defs);
      el("path", { d: "M0,0 L10,5 L0,10 z", fill: col }, m);
    });
    const R = map.n > 20 ? 7.5 : 10.5;
    const gE = el("g", {}, svg), gN = el("g", {}, svg);
    map.edges.forEach(([u, v]) => {
      const a = P[u], b = P[v];
      const dx = b.x - a.x, dy = b.y - a.y, dd = Math.hypot(dx, dy) || 1;
      const on = visitedAt[u] >= 0 && visitedAt[u] < loop;
      const newly = on && visitedAt[u] === loop - 1;
      el("line", {
        x1: a.x + dx / dd * R, y1: a.y + dy / dd * R, x2: b.x - dx / dd * (R + 1.5), y2: b.y - dy / dd * (R + 1.5),
        stroke: on ? ORANGE : LINE, "stroke-width": newly ? 2.6 : (on ? 1.8 : 1.3), "marker-end": "url(#" + (on ? "gt-ah-on" : "gt-ah") + ")",
        opacity: on ? 1 : 0.9,
      }, gE);
    });
    for (let i = 0; i < map.n; i++) {
      const g = el("g", { class: "gt-node" + (fresh.indexOf(i) >= 0 && !reduce ? " fresh" : ""), tabindex: i === map.A ? -1 : 0, role: "button",
        "aria-label": i === map.A ? "start A" : (i === B ? "goal B" : "point, click to make it B") }, gN);
      g.style.transformOrigin = P[i].x + "px " + P[i].y + "px";
      const vis = visitedAt[i] >= 0;
      const isA = i === map.A, isB = i === B;
      el("circle", { cx: P[i].x, cy: P[i].y, r: R, fill: isA ? INK : (vis ? ORANGE : "#fff"),
        stroke: isB ? INK : (vis || isA ? "#fff" : "#a9a79f"), "stroke-width": isB ? 2.4 : 1.4 }, g);
      const label = isA ? "A" : (vis ? String(visitedAt[i]) : "");
      if (label) {
        const t = el("text", { x: P[i].x, y: P[i].y + (R > 9 ? 4 : 3.2), "text-anchor": "middle", class: "gt-num",
          fill: "#fff", "font-size": R > 9 ? 11 : 8.5 }, g);
        t.textContent = label;
      }
      if (isB) {
        const t = el("text", { x: P[i].x, y: P[i].y - R - 5, "text-anchor": "middle", class: "gt-b" }, g);
        t.textContent = "B";
      }
      if (!isA) {
        const pick = () => { if (i !== B) { B = i; reset(); } };
        g.addEventListener("click", pick);
        g.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
      }
    }
  }

  const auto = window.wntAutoplay ? wntAutoplay(root, { play: () => play(), stop: () => stop(), playing: () => !!timer,
    next: () => { seed += 1; newMap(); } }) : { done() {} };
  goBtn.addEventListener("click", () => (timer ? stop() : play()));
  root.querySelector(".gt-step").addEventListener("click", () => { stop(); if (done) reset(); else step(); });
  root.querySelector(".gt-reset").addEventListener("click", reset);
  root.querySelector(".gt-new").addEventListener("click", () => { seed += 1; newMap(); });
  const setSize = (b) => {
    big = b;
    smallBtn.classList.toggle("on", !b); bigBtn.classList.toggle("on", b);
    smallBtn.setAttribute("aria-pressed", String(!b)); bigBtn.setAttribute("aria-pressed", String(b));
    newMap();
  };
  smallBtn.addEventListener("click", () => setSize(false));
  bigBtn.addEventListener("click", () => setSize(true));
  newMap();
})();
