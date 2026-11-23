// Interactive figure for the Sudoku section: the rule the written Sudoku transformer applies in every loop
// (run_data/logs/blog_handoff/sudoku_report.md), simulated on puzzles from the Sudoku-Extreme test set.
// All 81 cells update together. Each cell keeps the digits it could still be; it crosses out digits already taken
// in its row, column and box, and fills a digit that has only one place left in a unit. Then one of four things
// happens: progress (keep the new candidates), stuck (save the grid and guess the smallest digit at the cell with
// the fewest candidates, marking that digit as tried in the saved copy), contradiction (bring back the last saved
// grid), or solved (nothing changes). This is the rule, simulated, not the transformer's weights.
(function () {
  "use strict";
  const root = document.getElementById("sd");
  if (!root) return;

  // puzzles from the Sudoku-Extreme test set (index in the test split, puzzle)
  const PUZZLES = [
    [20, "42...8.3....4..52..7..6.....1.......5.7...3..293...47......314.7.1..4.6....2....5"],
    [17, "....6..42..24..3..9...2..5......61....15...3.8..17...4..52....3.9.....1..7...8..."],
    [11, ".63.......8.......219468...1..7.96...9.5...3...2.1.7.......3.7....1..2.49...748.."],
    [5, "..6.5...2.......9......83...95.26.....2.3...14..7.....62......5..1....3.53..6.1.."],
    [1, ".9......3....8.9...6.7.32....4..5..85.....63......9.5..8..1...41....782..239...7."],
    [73, "8..3..6...7.1.8...3...5..1...64..9...2.......1....5.4.2..5...8......3..9..7.2.4.."],
  ];
  const STACK_MAX = 24;
  const INK = "#14213d", BLUE = "#1c5cab", ORANGE = "#eb6834", ORANGE_TEXT = "#c4471a", RED = "#e34948";
  const UNITS = [];
  for (let r = 0; r < 9; r++) UNITS.push([...Array(9).keys()].map((c) => r * 9 + c));
  for (let c = 0; c < 9; c++) UNITS.push([...Array(9).keys()].map((r) => r * 9 + c));
  for (let b = 0; b < 9; b++) UNITS.push([...Array(9).keys()].map((k) => (Math.floor(b / 3) * 3 + Math.floor(k / 3)) * 9 + (b % 3) * 3 + (k % 3)));
  const PEERS = [...Array(81).keys()].map((c) => [...new Set(UNITS.filter((u) => u.includes(c)).flat())].filter((p) => p !== c));
  const pc = (m) => { let n = 0; while (m) { m &= m - 1; n++; } return n; };
  const digitOf = (m) => Math.round(Math.log2(m)) + 1;
  const name = (c) => "row " + (Math.floor(c / 9) + 1) + ", column " + (c % 9 + 1);

  // ------------------------------------------------------------------ one loop of the rule
  function step(S) {
    const { cands } = S;
    // contradiction: an empty cell, a digit with nowhere to go in a unit, or a digit written twice in a unit
    let bad = -1, badUnit = null;
    for (let c = 0; c < 81 && bad < 0; c++) if (!cands[c]) bad = c;
    if (bad < 0) {
      for (const u of UNITS) {
        for (let d = 0; d < 9 && !badUnit; d++) {
          let any = 0, solved = 0;
          u.forEach((c) => { if (cands[c] >> d & 1) { any++; if (pc(cands[c]) === 1) solved++; } });
          if (!any || solved > 1) badUnit = u;
        }
        if (badUnit) break;
      }
    }
    if (bad >= 0 || badUnit) {
      if (!S.stack.length) return { ev: "fail" };
      const top = S.stack.pop();
      S.cands = top.cands; S.origin = top.origin;
      return { ev: "back", bad, badUnit };
    }
    if (cands.every((m) => pc(m) === 1)) return { ev: "solved" };
    const next = cands.slice(), origin = S.origin.slice();
    for (let c = 0; c < 81; c++) {
      if (pc(cands[c]) === 1) continue;
      let taken = 0;
      PEERS[c].forEach((p) => { if (pc(cands[p]) === 1) taken |= cands[p]; });
      next[c] = cands[c] & ~taken;
    }
    for (const u of UNITS) for (let d = 0; d < 9; d++) {
      const where = u.filter((c) => cands[c] >> d & 1);
      if (where.length === 1 && pc(cands[where[0]]) > 1) next[where[0]] &= 1 << d;
    }
    const changed = [];
    for (let c = 0; c < 81; c++) if (next[c] !== cands[c]) { changed.push(c); if (pc(next[c]) === 1) origin[c] = "deduced"; }
    if (changed.length) {
      const filled = changed.filter((c) => pc(next[c]) === 1).length;
      S.cands = next; S.origin = origin;
      return { ev: "progress", changed, filled, crossed: changed.reduce((n, c) => n + pc(cands[c]) - pc(next[c]), 0) - filled * 0 };
    }
    // stuck: save the grid and guess
    let cell = -1;
    for (let c = 0; c < 81; c++) if (pc(cands[c]) > 1 && (cell < 0 || pc(cands[c]) < pc(cands[cell]))) cell = c;
    const d = cands[cell] & -cands[cell];
    const saved = cands.slice(); saved[cell] &= ~d;
    S.stack.push({ cands: saved, origin: S.origin.slice() });
    S.cands = cands.slice(); S.cands[cell] = d;
    S.origin = S.origin.slice(); S.origin[cell] = "guess";
    return { ev: "guess", cell, digit: digitOf(d), left: pc(saved[cell]) };
  }

  // ------------------------------------------------------------------ the page
  root.innerHTML =
    '<svg class="sd-svg" viewBox="0 0 396 396" role="img" aria-label="A Sudoku grid. Small digits are the ones a cell could still be."></svg>' +
    '<p class="sd-status" aria-live="polite"></p>' +
    '<div class="sd-stack" aria-hidden="true"><span class="sd-stack-l">Saved grids</span><span class="sd-slots"></span><span class="sd-stack-n"></span></div>' +
    '<div class="sd-ctrl">' +
    '<button type="button" class="sd-go">▶ Play</button><button type="button" class="sd-step">Step</button>' +
    '<button type="button" class="sd-reset">Reset</button><button type="button" class="sd-new">New puzzle</button>' +
    '<span class="sd-speed" role="group" aria-label="Speed"><button type="button" class="sd-slow on" aria-pressed="true">Slow</button>' +
    '<button type="button" class="sd-fast" aria-pressed="false">Fast</button></span></div>' +
    '<p class="sd-key"><span><b style="color:' + INK + '">5</b> given</span><span><b style="color:' + BLUE + '">5</b> worked out</span>' +
    '<span><b class="sd-kg">5</b> guessed</span><span><b class="sd-kc">2 7</b> still possible</span></p>';
  const svg = root.querySelector(".sd-svg"), status = root.querySelector(".sd-status");
  const goBtn = root.querySelector(".sd-go"), slots = root.querySelector(".sd-slots"), stackN = root.querySelector(".sd-stack-n");
  const slowBtn = root.querySelector(".sd-slow"), fastBtn = root.querySelector(".sd-fast");
  for (let k = 0; k < STACK_MAX; k++) slots.appendChild(document.createElement("i"));

  let which = 0, S, loop, done, timer = null, fast = false, last = null, stats;
  const NS = "http://www.w3.org/2000/svg";
  const el = (n, a, p) => { const e = document.createElementNS(NS, n); for (const k in a) e.setAttribute(k, a[k]); if (p) p.appendChild(e); return e; };

  function reset() {
    stop();
    const q = PUZZLES[which][1];
    S = { cands: [...q].map((ch) => (ch === "." ? 0x1FF : 1 << (+ch - 1))), origin: [...q].map((ch) => (ch === "." ? "" : "given")), stack: [] };
    loop = 0; done = false; last = null; stats = { guess: 0, back: 0 };
    draw();
    say("Test puzzle " + PUZZLES[which][0] + ", " + q.replace(/\./g, "").length + " digits given. Loop 0.");
  }

  function say(s) { status.textContent = s; }

  function advance() {
    if (done) return;
    const r = step(S);
    if (r.ev === "solved" || r.ev === "fail") {
      done = true;
      say(r.ev === "solved" ? "Solved after " + loop + " loops, with " + stats.guess + (stats.guess === 1 ? " guess" : " guesses") + " and " +
        stats.back + " back-steps. Later loops change nothing." : "No saved grid is left to go back to.");
      last = null; draw(); stop(); auto.done(); return;
    }
    loop += 1;
    last = r;
    if (r.ev === "guess") {
      stats.guess++;
      say("Loop " + loop + ". Stuck, so the grid is saved and the cell at " + name(r.cell) + " guesses " + r.digit + ".");
    } else if (r.ev === "back") {
      stats.back++;
      say("Loop " + loop + ". " + (r.bad >= 0 ? "The cell at " + name(r.bad) + " has no digit left" : "A row, column or box cannot be completed") +
        ", so the last saved grid comes back, with its guess crossed out.");
    } else {
      say("Loop " + loop + ". " + (r.filled ? r.filled + (r.filled === 1 ? " cell filled" : " cells filled") + ", " : "") +
        "digits crossed out in " + r.changed.length + (r.changed.length === 1 ? " cell." : " cells."));
    }
    draw();
  }

  function draw() {
    svg.textContent = "";
    const C = 44;
    const g = el("g", { transform: "translate(0,0)" }, svg);
    const hot = new Set(last && last.ev === "progress" ? last.changed : []);
    const badSet = new Set(last && last.ev === "back" ? (last.bad >= 0 ? [last.bad] : last.badUnit || []) : []);
    for (let c = 0; c < 81; c++) {
      const x = (c % 9) * C, y = Math.floor(c / 9) * C, m = S.cands[c], o = S.origin[c];
      let bg = "#fff";
      if (o === "guess") bg = "#fde3d6";
      if (hot.has(c)) bg = "#e3eefc";
      if (badSet.has(c)) bg = "#fbd5d5";
      if (last && last.ev === "guess" && last.cell === c) bg = "#fbc6ad";
      el("rect", { x, y, width: C, height: C, fill: bg }, g);
      if (pc(m) === 1) {
        const t = el("text", { x: x + C / 2, y: y + C / 2 + 8, "text-anchor": "middle", class: "sd-d",
          fill: o === "given" ? INK : (o === "guess" ? ORANGE_TEXT : BLUE), "font-weight": o === "given" ? 800 : 600 }, g);
        t.textContent = digitOf(m);
      } else if (m) {
        for (let d = 0; d < 9; d++) if (m >> d & 1) {
          const t = el("text", { x: x + 8 + (d % 3) * 14, y: y + 12 + Math.floor(d / 3) * 13, "text-anchor": "middle", class: "sd-c" }, g);
          t.textContent = d + 1;
        }
      }
    }
    for (let k = 0; k <= 9; k++) {
      const w = k % 3 === 0 ? 2.4 : 0.8, col = k % 3 === 0 ? INK : "#cfcdc6";
      el("line", { x1: k * C, y1: 0, x2: k * C, y2: 9 * C, stroke: col, "stroke-width": w }, svg);
      el("line", { x1: 0, y1: k * C, x2: 9 * C, y2: k * C, stroke: col, "stroke-width": w }, svg);
    }
    [...slots.children].forEach((s, k) => s.classList.toggle("on", k < S.stack.length));
    stackN.textContent = S.stack.length + " of " + STACK_MAX;
  }

  function play() {
    if (done) reset();
    goBtn.textContent = "❚❚ Pause"; goBtn.classList.add("on");
    const tick = () => { advance(); if (!done) timer = setTimeout(tick, fast ? 40 : 420); };
    timer = setTimeout(tick, 150);
  }
  function stop() { if (timer) clearTimeout(timer); timer = null; goBtn.textContent = "▶ Play"; goBtn.classList.remove("on"); }

  const auto = window.wntAutoplay ? wntAutoplay(root, { play: () => play(), stop: () => stop(), playing: () => !!timer,
    next: () => { which = (which + 1) % 5; reset(); }, wait: 3200 }) : { done() {} };
  goBtn.addEventListener("click", () => (timer ? stop() : play()));
  root.querySelector(".sd-step").addEventListener("click", () => { stop(); if (done) reset(); else advance(); });
  root.querySelector(".sd-reset").addEventListener("click", reset);
  root.querySelector(".sd-new").addEventListener("click", () => { which = (which + 1) % PUZZLES.length; reset(); });
  const setFast = (f) => {
    fast = f;
    slowBtn.classList.toggle("on", !f); fastBtn.classList.toggle("on", f);
    slowBtn.setAttribute("aria-pressed", String(!f)); fastBtn.setAttribute("aria-pressed", String(f));
  };
  slowBtn.addEventListener("click", () => setFast(false));
  fastBtn.addEventListener("click", () => setFast(true));
  reset();
})();
