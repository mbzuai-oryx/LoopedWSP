// Interactive figure for GraphWalks: a small prompt of one-way links between 10-character node names, and the question
// "which nodes are exactly N steps from the start?". It follows the rule the written GraphWalks transformer applies:
// on every loop, each link checks whether a link into its starting node has already been visited, so the visited area
// grows by one step per loop; at the end, every node exactly N steps away writes its name into the answer.
// Real prompts hold thousands of links (up to about 1.75 million characters). This is the rule, simulated.
(function () {
  "use strict";
  const root = document.getElementById("gw");
  if (!root) return;

  const W = 520, H = 170, PAD = 16;
  const ORANGE = "#eb6834", INK = "#14213d", LINE = "#c3c1ba";
  const RAMP = ["#14213d", "#1c5cab", "#3a7ed6", "#6da7ec", "#9ec5f4", "#cde2fb"];
  const reduce = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;
  const NS = "http://www.w3.org/2000/svg";

  function rng(seed) {
    let a = seed >>> 0;
    return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  }
  const randint = (r, a, b) => a + Math.floor(r() * (b - a + 1));
  const word = (r) => Array.from({ length: 10 }, () => "abcdefghijklmnopqrstuvwxyz"[randint(r, 0, 25)]).join("");

  function makeGraph(seed) {
    const r = rng(seed * 104729 + 7);
    const n = randint(r, 11, 12);
    const names = Array.from({ length: n }, () => word(r));
    const edges = [], has = new Set();
    const add = (u, v) => { if (u === v || has.has(u + ">" + v)) return; has.add(u + ">" + v); edges.push([u, v]); };
    for (let v = 1; v < n; v++) add(randint(r, Math.max(0, v - 4), v - 1), v);       // a spine of links out of node 0
    for (let k = 0; k < Math.round(n * 0.5); k++) add(randint(r, 0, n - 1), randint(r, 0, n - 1));
    for (let i = edges.length - 1; i > 0; i--) { const j = randint(r, 0, i); [edges[i], edges[j]] = [edges[j], edges[i]]; }  // prompts list links in no order
    return { n, names, edges, start: 0 };
  }

  function layout(g, seed) {
    const r = rng(seed * 17 + 3), { n, edges } = g;
    const P = Array.from({ length: n }, () => ({ x: PAD + r() * (W - 2 * PAD), y: PAD + r() * (H - 2 * PAD) }));
    P[0] = { x: PAD + 16, y: H / 2 };
    const k = 0.8 * Math.sqrt((W - 2 * PAD) * (H - 2 * PAD) / n);
    for (let it = 0, temp = 36; it < 380; it++, temp *= 0.99) {
      const F = P.map(() => ({ x: 0, y: 0 }));
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
        let dx = P[i].x - P[j].x, dy = P[i].y - P[j].y; const d = Math.max(1, Math.hypot(dx, dy)), f = k * k / d;
        dx /= d; dy /= d; F[i].x += dx * f; F[i].y += dy * f; F[j].x -= dx * f; F[j].y -= dy * f;
      }
      edges.forEach(([u, v]) => {
        let dx = P[u].x - P[v].x, dy = P[u].y - P[v].y; const d = Math.max(1, Math.hypot(dx, dy)), f = d * d / k;
        dx /= d; dy /= d; F[u].x -= dx * f; F[u].y -= dy * f; F[v].x += dx * f; F[v].y += dy * f;
      });
      for (let i = 1; i < n; i++) {
        F[i].x += (W / 2 - P[i].x) * 0.03; F[i].y += (H / 2 - P[i].y) * 0.06;
        const m = Math.hypot(F[i].x, F[i].y) || 1, s = Math.min(m, temp) / m;
        P[i].x = Math.min(W - PAD, Math.max(PAD, P[i].x + F[i].x * s));
        P[i].y = Math.min(H - PAD, Math.max(PAD, P[i].y + F[i].y * s));
      }
    }
    return P;
  }

  root.innerHTML =
    '<div class="gw-prompt" aria-label="The prompt: one-way links, then the question"><div class="gw-links"></div><p class="gw-q"></p></div>' +
    '<svg class="gw-svg" viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="The links drawn as a map"></svg>' +
    '<p class="gw-status" aria-live="polite"></p>' +
    '<div class="gw-ctrl"><button type="button" class="gw-go">▶ Play</button><button type="button" class="gw-step">Step</button>' +
    '<button type="button" class="gw-reset">Reset</button><button type="button" class="gw-new">New prompt</button>' +
    '<span class="gw-n" role="group" aria-label="Steps N">N =</span></div>';
  const linksEl = root.querySelector(".gw-links"), qEl = root.querySelector(".gw-q"), svg = root.querySelector(".gw-svg");
  const status = root.querySelector(".gw-status"), goBtn = root.querySelector(".gw-go"), nGroup = root.querySelector(".gw-n");
  const nBtns = [1, 2, 3, 4].map((k) => { const b = document.createElement("button"); b.type = "button"; b.textContent = k; nGroup.appendChild(b); return b; });

  let seed = 2, N = 3, g, P, linkAt, nodeAt, loop, done, timer = null;
  const el = (name, a, p) => { const e = document.createElementNS(NS, name); for (const k in a) e.setAttribute(k, a[k]); if (p) p.appendChild(e); return e; };

  function newGraph() {
    for (let tries = 0; tries < 50; tries++) {          // a prompt with at least one node N steps away
      g = makeGraph(seed);
      const d = bfs(g);
      if (d.some((x) => x === N)) break;
      seed++;
    }
    P = layout(g, seed);
    reset();
  }
  function bfs(gr) {
    const d = new Array(gr.n).fill(-1); d[gr.start] = 0; const q = [gr.start];
    while (q.length) { const u = q.shift(); gr.edges.forEach(([a, b]) => { if (a === u && d[b] < 0) { d[b] = d[u] + 1; q.push(b); } }); }
    return d;
  }
  function reset() {
    stop();
    linkAt = new Array(g.edges.length).fill(-1);
    nodeAt = new Array(g.n).fill(-1); nodeAt[g.start] = 0;
    loop = 0; done = false;
    nBtns.forEach((b, k) => { b.classList.toggle("on", k + 1 === N); b.setAttribute("aria-pressed", String(k + 1 === N)); });
    linksEl.innerHTML = g.edges.map(([u, v], i) => '<span class="gw-l" data-i="' + i + '">' + g.names[u] + " → " + g.names[v] + "</span>").join("") +
      '<span class="gw-more">… thousands more links in a real prompt</span>';
    qEl.innerHTML = "Which nodes are exactly <b>" + N + "</b> " + (N === 1 ? "step" : "steps") + " from <b>" + g.names[g.start] + "</b>?";
    draw();
    say("Loop 0. Only the start node, " + g.names[g.start] + ", is visited.");
  }
  function say(s) { status.textContent = s; }

  function step() {
    if (done) return;
    loop += 1;
    let fresh = 0;
    g.edges.forEach(([u, v], i) => {
      if (linkAt[i] >= 0) return;
      const ok = u === g.start || g.edges.some(([a, b], j) => b === u && linkAt[j] >= 0 && linkAt[j] < loop);
      if (ok) { linkAt[i] = loop; fresh++; if (nodeAt[v] < 0) nodeAt[v] = loop; }
    });
    if (loop >= N) {
      done = true;
      const ans = nodeAt.map((d, i) => (d === N ? g.names[i] : null)).filter(Boolean);
      say("Loop " + loop + ". The visited area reaches " + N + (N === 1 ? " step" : " steps") + ", so the nodes first visited now write their names. Answer: " + ans.join(", ") + ".");
      stop(); auto.done();
    } else {
      say("Loop " + loop + ". " + fresh + (fresh === 1 ? " link" : " links") + " found that a link into their starting node was visited.");
    }
    draw();
  }

  function draw() {
    svg.textContent = "";
    const defs = el("defs", {}, svg);
    [["gw-ah", LINE], ["gw-ah-on", ORANGE]].forEach(([id, col]) => {
      const m = el("marker", { id, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 5, markerHeight: 5, orient: "auto-start-reverse" }, defs);
      el("path", { d: "M0,0 L10,5 L0,10 z", fill: col }, m);
    });
    const R = 9;
    g.edges.forEach(([u, v], i) => {
      const a = P[u], b = P[v], dx = b.x - a.x, dy = b.y - a.y, d = Math.hypot(dx, dy) || 1, on = linkAt[i] >= 0;
      el("line", { x1: a.x + dx / d * R, y1: a.y + dy / d * R, x2: b.x - dx / d * (R + 1.5), y2: b.y - dy / d * (R + 1.5),
        stroke: on ? ORANGE : LINE, "stroke-width": on && linkAt[i] === loop ? 2.6 : (on ? 1.7 : 1.2), "marker-end": "url(#" + (on ? "gw-ah-on" : "gw-ah") + ")" }, svg);
    });
    for (let i = 0; i < g.n; i++) {
      const d = nodeAt[i], ans = done && d === N;
      el("circle", { cx: P[i].x, cy: P[i].y, r: R, fill: i === g.start ? INK : (ans ? ORANGE : (d >= 0 ? RAMP[Math.min(d, RAMP.length - 1)] : "#fff")),
        stroke: d >= 0 || i === g.start ? "#fff" : "#a9a79f", "stroke-width": 1.4 }, svg);
      if (d > 0) { const t = el("text", { x: P[i].x, y: P[i].y + 3.5, "text-anchor": "middle", class: "gw-num", fill: d >= 4 && !ans ? INK : "#fff" }, svg); t.textContent = d; }
      if (i === g.start || ans) {
        const t = el("text", { x: P[i].x, y: P[i].y - R - 4, "text-anchor": "middle", class: "gw-name", fill: ans ? "#c4471a" : INK }, svg);
        t.textContent = g.names[i];
      }
    }
    linksEl.querySelectorAll(".gw-l").forEach((s) => {
      const k = linkAt[+s.dataset.i];
      s.className = "gw-l" + (k >= 0 ? " on" : "") + (k === loop && k > 0 ? " now" : "");
    });
  }

  function play() {
    if (done) reset();
    goBtn.textContent = "❚❚ Pause"; goBtn.classList.add("on");
    const tick = () => { step(); if (!done) timer = setTimeout(tick, reduce ? 1100 : 900); };
    timer = setTimeout(tick, 300);
  }
  function stop() { if (timer) clearTimeout(timer); timer = null; goBtn.textContent = "▶ Play"; goBtn.classList.remove("on"); }

  const auto = window.wntAutoplay ? wntAutoplay(root, { play: () => play(), stop: () => stop(), playing: () => !!timer,
    next: () => { seed += 1; newGraph(); }, wait: 3200 }) : { done() {} };
  goBtn.addEventListener("click", () => (timer ? stop() : play()));
  root.querySelector(".gw-step").addEventListener("click", () => { stop(); if (done) reset(); else step(); });
  root.querySelector(".gw-reset").addEventListener("click", reset);
  root.querySelector(".gw-new").addEventListener("click", () => { seed += 1; newGraph(); });
  nBtns.forEach((b, k) => b.addEventListener("click", () => { N = k + 1; if (!bfs(g).some((x) => x === N)) newGraph(); else reset(); }));
  newGraph();
})();
