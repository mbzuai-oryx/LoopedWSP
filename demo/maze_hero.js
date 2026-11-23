// Figure 2: the transformer Claude wrote, solving the first Maze-Hard test maze, drawn from its real per-loop state
// (window.MAZE_HERO, from demo/maze_hero_data.js). A wave of distance counts spreads from the goal, one cell per loop,
// shaded light to dark; once it visits the start cell, the path grows from the start back to the goal, one cell per
// loop. It plays when scrolled into view, pauses on click, and shows the solved maze to readers who prefer less motion.
(function () {
  "use strict";
  const root = document.getElementById("mh");
  const D = window.MAZE_HERO;
  if (!root || !D) return;

  const N = 30, END = D.doneAt + 2;
  const WALL = "#22263a", OPEN = "#f1efe9", PATH = "#eb6834", FRONT = "#ffd8a8";
  const RAMP = ["#dbe9fb", "#b6d2f6", "#8db8ef", "#5f9ae4", "#3a7ed6", "#2a66b8", "#1d4e93", "#163d75"];
  const reduce = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;
  const maxReach = Math.max(...D.reach);
  const pathOrder = D.on.map((l, i) => [l, i]).filter((x) => x[0] >= 0).sort((a, b) => a[0] - b[0]).map((x) => x[1]);

  root.innerHTML = '<canvas class="mh-canvas" role="img" aria-label="A 30 by 30 maze. A wave of distances spreads out from the goal until it visits the start cell, then the path grows from the start cell back to the goal."></canvas>' +
    '<div class="mh-bar"><span class="mh-loop"></span><span class="mh-phase"></span>' +
    '<button type="button" class="mh-btn" aria-label="Pause">❚❚</button></div>' +
    '<div class="mh-track"><div class="mh-fill"></div><i style="left:' + (100 * D.startAt / END).toFixed(1) + '%"></i></div>';
  const cv = root.querySelector("canvas"), ctx = cv.getContext("2d");
  const loopEl = root.querySelector(".mh-loop"), phaseEl = root.querySelector(".mh-phase");
  const btn = root.querySelector(".mh-btn"), fill = root.querySelector(".mh-fill");

  let size = 0, cell = 0;
  function resize() {
    const w = cv.clientWidth || 500, dpr = Math.min(2, window.devicePixelRatio || 1);
    size = w; cell = w / N;
    cv.width = Math.round(w * dpr); cv.height = Math.round(w * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  const rampAt = (t) => RAMP[Math.min(RAMP.length - 1, Math.floor(t * RAMP.length))];
  const cx = (i) => (i % N + 0.5) * cell, cy = (i) => (Math.floor(i / N) + 0.5) * cell;

  function draw(L) {                         // L: loops done so far (may be fractional)
    const loop = Math.floor(L);
    ctx.clearRect(0, 0, size, size);
    for (let i = 0; i < N * N; i++) {
      const x = (i % N) * cell, y = Math.floor(i / N) * cell, ch = D.q[i];
      let col = OPEN;
      if (ch === "#") col = WALL;
      else if (D.reach[i] >= 0 && D.reach[i] <= loop) col = loop - D.reach[i] < 2 && loop < maxReach + 2 ? FRONT : rampAt(D.reach[i] / (maxReach + 1));
      ctx.fillStyle = col;
      ctx.fillRect(x, y, cell + 0.5, cell + 0.5);
    }
    // the path, from the start cell toward the goal, in the order its cells joined
    const shown = pathOrder.filter((i) => D.on[i] <= loop && (i === D.start || loop >= D.startAt));
    if (shown.length > 1) {
      ctx.strokeStyle = PATH; ctx.lineWidth = Math.max(3, cell * 0.42); ctx.lineCap = "round"; ctx.lineJoin = "round";
      ctx.beginPath();
      shown.forEach((i, k) => (k ? ctx.lineTo(cx(i), cy(i)) : ctx.moveTo(cx(i), cy(i))));
      ctx.stroke();
      const h = shown[shown.length - 1];
      if (loop < D.doneAt) { ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.arc(cx(h), cy(h), cell * 0.17, 0, 7); ctx.fill(); }
    }
    // start and goal
    const mark = (i, fillCol, ring, txt) => {
      ctx.fillStyle = fillCol; ctx.strokeStyle = ring; ctx.lineWidth = Math.max(1.5, cell * 0.12);
      ctx.beginPath(); ctx.arc(cx(i), cy(i), cell * 0.62, 0, 7); ctx.fill(); ctx.stroke();
      ctx.fillStyle = ring; ctx.font = "800 " + Math.round(cell * 0.62) + "px Inter, system-ui, sans-serif";
      ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(txt, cx(i), cy(i) + cell * 0.03);
    };
    mark(D.goal, "#fff", PATH, "G");
    mark(D.start, "#fff", WALL, "S");
    loopEl.textContent = "Loop " + Math.min(loop, D.doneAt);
    phaseEl.textContent = loop < D.startAt ? "distances spread from the goal G" :
      (loop < D.doneAt ? "the path grows from the start S" : "solved, and later loops change nothing");
    fill.style.width = (100 * Math.min(L, END) / END).toFixed(2) + "%";
  }

  // time line: the wave phase in 4.2 s, the path phase in 3.6 s, then a 2.6 s hold
  const T1 = 4200, T2 = 3600, HOLD = 2600;
  const loopAt = (ms) => (ms < T1 ? D.startAt * ms / T1 : D.startAt + (END - D.startAt) * Math.min(1, (ms - T1) / T2));
  let t0 = null, playing = false, raf = 0, pausedAt = 0, seen = false;
  function frame(now) {
    if (t0 === null) t0 = now - pausedAt;
    const ms = (now - t0) % (T1 + T2 + HOLD);
    draw(loopAt(ms));
    pausedAt = now - t0;
    raf = requestAnimationFrame(frame);
  }
  function play() { if (playing) return; playing = true; t0 = null; btn.textContent = "❚❚"; btn.setAttribute("aria-label", "Pause"); raf = requestAnimationFrame(frame); }
  function pause() { playing = false; cancelAnimationFrame(raf); btn.textContent = "▶"; btn.setAttribute("aria-label", "Play"); }
  const toggle = () => (playing ? pause() : play());
  btn.addEventListener("click", toggle);
  cv.addEventListener("click", toggle);

  resize();
  if (typeof window.MH_FRAME === "number") { draw(window.MH_FRAME); return; }   // for screenshots
  if (reduce) { draw(END); btn.textContent = "▶"; btn.setAttribute("aria-label", "Play"); }
  else draw(0);
  window.addEventListener("resize", () => { resize(); if (!playing) draw(reduce && !seen ? END : loopAt(pausedAt % (T1 + T2 + HOLD))); });
  if (!reduce && "IntersectionObserver" in window) {
    new IntersectionObserver((es) => es.forEach((e) => {
      if (e.isIntersecting && !seen) { seen = true; play(); }
      else if (!e.isIntersecting && playing) pause();
      else if (e.isIntersecting && seen && !playing && btn.dataset.user !== "1") play();
    }), { threshold: 0.35 }).observe(root);
    btn.addEventListener("click", () => { btn.dataset.user = playing ? "" : "1"; });
    cv.addEventListener("click", () => { btn.dataset.user = playing ? "" : "1"; });
  } else if (!reduce) play();
})();
