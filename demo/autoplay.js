// Autoplay for the interactive figures: a figure starts playing when it scrolls into view, pauses when it leaves,
// and after each finished run moves on to the next example and plays again. The first click or key press inside a
// figure hands it back to the reader for good. Readers who prefer less motion get no autoplay.
// Use: const auto = wntAutoplay(root, { play, stop, playing, next }); then call auto.done() when a run finishes.
(function () {
  "use strict";
  window.wntAutoplay = function (root, o) {
    const reduce = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (!root || reduce) return { done() {} };
    let on = true, visible = false, t = null;
    const off = () => { on = false; clearTimeout(t); };
    root.addEventListener("pointerdown", off, true);
    root.addEventListener("keydown", off, true);
    function setVisible(v) {
      if (v === visible) return;
      visible = v;
      if (!on) return;
      if (visible) { if (!o.playing()) o.play(); }
      else { clearTimeout(t); o.stop(); }
    }
    // in view: at least a third of the figure, or of the window for a figure taller than the window, is on screen
    function check() {
      const r = root.getBoundingClientRect(), h = window.innerHeight || document.documentElement.clientHeight;
      const seen = Math.min(r.bottom, h) - Math.max(r.top, 0);
      setVisible(seen > Math.min(r.height, h) / 3);
    }
    if ("IntersectionObserver" in window) {
      new IntersectionObserver(() => check(), { threshold: [0, 0.2, 0.35, 0.5, 0.8, 1] }).observe(root);
    }
    let pending = false;
    const onScroll = () => { if (!pending) { pending = true; requestAnimationFrame(() => { pending = false; check(); }); } };
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    setTimeout(check, 0);
    return {
      done() {
        if (!on) return;
        clearTimeout(t);
        t = setTimeout(() => { if (on && visible) { o.next(); o.play(); } }, o.wait || 2600);
      },
    };
  };
})();
