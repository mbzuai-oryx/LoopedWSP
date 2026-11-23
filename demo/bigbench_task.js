// Interactive figure for the four BIG-Bench transformers. One tab per task, each stepping through the rule its
// written transformer applies loop by loop (as described in the section):
//   Boolean Expressions  simplify the expression one step per loop
//   Dyck Languages       work out how deep each bracket sits, then close the most recent open bracket, one per loop
//   Web of Lies          everyone whose subject is already settled works out whether they tell the truth
//   Shuffled Objects     each swap merges its record with the record twice as far back (a doubling trick)
// The example questions are our own, in each task's format. This is the rule, simulated, not the weights.
(function () {
  "use strict";
  const root = document.getElementById("bb");
  if (!root) return;
  const reduce = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  // ------------------------------------------------------------------ Boolean Expressions
  function booleanFrames() {
    let toks = "not ( True and False ) or ( False and not True )".split(" ");
    const frames = [{ toks: toks.slice(), hi: [], note: "The question, as tokens." }];
    const isV = (x) => x === "True" || x === "False";
    const V = (b) => (b ? "True" : "False");
    for (let guard = 0; guard < 20 && toks.length > 1; guard++) {
      let done = false;
      // innermost parentheses first: find a "(" whose matching ")" encloses no other "("
      let lo = 0, hi = toks.length;
      for (let i = 0; i < toks.length; i++) if (toks[i] === "(") { const j = toks.indexOf(")", i); if (toks.slice(i + 1, j).indexOf("(") < 0) { lo = i + 1; hi = j; break; } }
      const span = toks.slice(lo, hi);
      const apply = (k, n, val, note) => { toks.splice(lo + k, n, val); frames.push({ toks: toks.slice(), hi: [lo + k], note }); done = true; };
      let k = span.findIndex((x, i) => x === "not" && isV(span[i + 1]));
      if (k >= 0) apply(k, 2, V(span[k + 1] === "False"), "not " + span[k + 1] + " becomes " + V(span[k + 1] === "False") + ".");
      if (!done) { k = span.findIndex((x, i) => x === "and" && isV(span[i - 1]) && isV(span[i + 1]) && span[i - 2] !== "not");
        if (k >= 0) apply(k - 1, 3, V(span[k - 1] === "True" && span[k + 1] === "True"), span[k - 1] + " and " + span[k + 1] + " becomes " + V(span[k - 1] === "True" && span[k + 1] === "True") + "."); }
      if (!done) { k = span.findIndex((x, i) => x === "or" && isV(span[i - 1]) && isV(span[i + 1]));
        if (k >= 0) apply(k - 1, 3, V(span[k - 1] === "True" || span[k + 1] === "True"), span[k - 1] + " or " + span[k + 1] + " becomes " + V(span[k - 1] === "True" || span[k + 1] === "True") + "."); }
      if (!done && lo > 0 && hi - lo === 1) { const v = toks[lo]; toks.splice(lo - 1, 3, v); frames.push({ toks: toks.slice(), hi: [lo - 1], note: "The brackets around " + v + " drop away." }); done = true; }
      if (!done) break;
    }
    frames[frames.length - 1].note += " That is the answer.";
    return frames.map((f) => '<div class="bb-toks">' + f.toks.map((x, i) => '<span class="bb-t' + (f.hi.includes(i) ? " hi" : "") + (isV(x) ? " v" : "") + '">' + x + "</span>").join("") + "</div>" +
      '<p class="bb-note">' + f.note + "</p>");
  }

  // ------------------------------------------------------------------ Dyck Languages
  function dyckFrames() {
    const seq = "[ { ( ) < [ ] ( ".trim().split(" ");
    const open = { "(": ")", "[": "]", "{": "}", "<": ">" };
    const depth = []; let d = 0;
    seq.forEach((c) => { if (open[c]) { d++; depth.push(d); } else { depth.push(d); d--; } });
    const stack = []; seq.forEach((c) => (open[c] ? stack.push(c) : stack.pop()));
    const closers = stack.slice().reverse().map((c) => open[c]);
    const row = (showDepth, out, hiOut) =>
      '<div class="bb-toks">' + seq.map((c, i) => '<span class="bb-t br">' + esc(c) + (showDepth ? "<i>" + depth[i] + "</i>" : "") + "</span>").join("") +
      out.map((c, i) => '<span class="bb-t out' + (i === hiOut ? " hi" : "") + '">' + esc(c) + "</span>").join("") + "</div>";
    const frames = [row(false, [], -1) + '<p class="bb-note">The sequence to close.</p>',
      row(true, [], -1) + '<p class="bb-note">Adding up openers and closers gives each bracket its depth. ' + stack.length + " are still open.</p>"];
    for (let k = 1; k <= closers.length; k++) {
      frames.push(row(true, closers.slice(0, k), k - 1) + '<p class="bb-note">Close the most recent bracket that is still open with ' + esc(closers[k - 1]) +
        (k === closers.length ? ". Every bracket is closed, so that is the answer." : ".") + "</p>");
    }
    return frames;
  }

  // ------------------------------------------------------------------ Web of Lies
  function liesFrames() {
    const P = [ // name, subject (index) or null, says truthful?
      ["Fidel", null, false], ["Sherrie", 0, true], ["Vina", 0, false], ["Kandi", 2, false], ["Delfina", 1, true], ["Raymond", 3, true]];
    const val = P.map((p) => (p[1] === null ? p[2] : null)), when = P.map((p) => (p[1] === null ? 0 : -1));
    const frames = [];
    const say = (p) => (p[1] === null ? p[0] + (p[2] ? " tells the truth." : " lies.") : p[0] + " says " + P[p[1]][0] + (p[2] ? " tells the truth." : " lies."));
    const card = (i, loop) => {
      const v = val[i], cls = v === null ? "" : (v ? " t" : " f"), now = when[i] === loop && loop > 0 ? " hi" : "";
      return '<div class="bb-card' + cls + now + '"><b>' + P[i][0] + '</b><span class="bb-say">' + say(P[i]) + '</span><span class="bb-val">' +
        (v === null ? "?" : (v ? "truthful" : "liar")) + "</span></div>";
    };
    const render = (loop, note) => '<div class="bb-cards">' + P.map((_, i) => card(i, loop)).join("") + '</div><p class="bb-note">' + note + "</p>";
    frames.push(render(0, "Does Raymond tell the truth? Only Fidel is settled at first."));
    for (let loop = 1; loop < 6; loop++) {
      const fresh = [];
      P.forEach((p, i) => { if (val[i] === null && val[p[1]] !== null && when[p[1]] < loop) { val[i] = p[2] === val[p[1]]; when[i] = loop; fresh.push(p[0]); } });
      if (!fresh.length) break;
      const last = val[5] !== null;
      frames.push(render(loop, "This loop, " + fresh.join(" and ") + (fresh.length === 1 ? " works it out, since their subject is settled." : " work it out at once, since their subjects are settled.") +
        (last ? " Raymond is " + (val[5] ? "truthful" : "a liar") + ", so the answer is " + (val[5] ? "yes." : "no.") : "")));
    }
    return frames;
  }

  // ------------------------------------------------------------------ Shuffled Objects
  function shuffleFrames() {
    const people = ["Alice", "Bob", "Claire", "Dave", "Eve"], items = ["red", "blue", "green", "yellow", "purple"];
    const swaps = [[0, 1], [2, 4], [1, 3], [0, 2], [3, 4], [1, 2], [0, 4]];
    const n = swaps.length;
    const frames = [];
    const strip = (cover, note, final) => {
      let h = '<div class="bb-swaps">';
      swaps.forEach(([a, b], i) => {
        const lo = cover[i];
        h += '<div class="bb-sw" style="--k:' + i + '"><span>' + people[a][0] + "⇄" + people[b][0] + '</span><i style="--w:' + (i - lo + 1) + '" title="covers swaps ' + (lo + 1) + " to " + (i + 1) + '"></i><em>' + (lo + 1) + "–" + (i + 1) + "</em></div>";
      });
      h += "</div>";
      if (final) {
        const hold = items.slice();
        swaps.forEach(([a, b]) => { const t = hold[a]; hold[a] = hold[b]; hold[b] = t; });
        h += '<div class="bb-hold">' + people.map((p, i) => '<span><b>' + p + "</b> " + hold[i] + "</span>").join("") + "</div>";
      }
      return h + '<p class="bb-note">' + note + "</p>";
    };
    let cover = swaps.map((_, i) => i);
    frames.push(strip(cover, "Alice, Bob, Claire, Dave and Eve start with red, blue, green, yellow and purple balls, then make 7 swaps. Each swap knows only itself.", false));
    for (let loop = 1, span = 1; span < n; loop++, span *= 2) {
      cover = cover.map((lo, i) => (i - span >= 0 ? cover[i - span] : lo));
      const full = cover[n - 1] === 0;
      frames.push(strip(cover, "Loop " + loop + ". Each swap merges its record with the one " + span + (span === 1 ? " place" : " places") + " back, so each now covers up to " + Math.min(2 * span, n) + " swaps." +
        (full ? " The last one covers all 7, so it knows who holds what." : ""), full));
    }
    return frames;
  }

  const TASKS = [["Boolean Expressions", booleanFrames], ["Dyck Languages", dyckFrames], ["Web of Lies", liesFrames], ["Shuffled Objects", shuffleFrames]];
  root.innerHTML = '<div class="bb-tabs" role="tablist">' + TASKS.map((t, i) => '<button type="button" role="tab" data-i="' + i + '">' + t[0] + "</button>").join("") + "</div>" +
    '<div class="bb-stage" aria-live="polite"></div>' +
    '<div class="bb-ctrl"><button type="button" class="bb-go">▶ Play</button><button type="button" class="bb-step">Step</button>' +
    '<button type="button" class="bb-reset">Reset</button><span class="bb-loop"></span></div>';
  const stage = root.querySelector(".bb-stage"), goBtn = root.querySelector(".bb-go"), loopEl = root.querySelector(".bb-loop");
  const tabs = [...root.querySelectorAll(".bb-tabs button")];
  let task = 0, frames = [], k = 0, timer = null;

  function show() {
    stage.innerHTML = frames[k];
    loopEl.textContent = k === 0 ? "start" : "loop " + k + " of " + (frames.length - 1);
  }
  function choose(i) {
    stop(); task = i; frames = TASKS[i][1](); k = 0;
    tabs.forEach((b, j) => { b.classList.toggle("on", j === i); b.setAttribute("aria-selected", String(j === i)); });
    show();
  }
  function step() { if (k < frames.length - 1) { k++; show(); } if (k >= frames.length - 1) { const was = !!timer; stop(); if (was) auto.done(); } }
  function play() {
    if (k >= frames.length - 1) { k = 0; show(); }
    goBtn.textContent = "❚❚ Pause"; goBtn.classList.add("on");
    const tick = () => { step(); if (k < frames.length - 1) timer = setTimeout(tick, reduce ? 1600 : 1300); };
    timer = setTimeout(tick, 700);
  }
  function stop() { if (timer) clearTimeout(timer); timer = null; goBtn.textContent = "▶ Play"; goBtn.classList.remove("on"); }
  const auto = window.wntAutoplay ? wntAutoplay(root, { play: () => play(), stop: () => stop(), playing: () => !!timer,
    next: () => choose((task + 1) % TASKS.length), wait: 2800 }) : { done() {} };
  tabs.forEach((b, i) => b.addEventListener("click", () => choose(i)));
  goBtn.addEventListener("click", () => (timer ? stop() : play()));
  root.querySelector(".bb-step").addEventListener("click", () => { stop(); if (k >= frames.length - 1) { k = 0; show(); } else step(); });
  root.querySelector(".bb-reset").addEventListener("click", () => { stop(); k = 0; show(); });
  choose(0);
})();
