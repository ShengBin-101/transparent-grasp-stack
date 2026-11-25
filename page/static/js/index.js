// Project page behaviour. The copy-BibTeX function follows the Academic Project Page Template's index.js
// (CC BY-SA 4.0); the rest is new: in-view video playback with pause buttons, the depth slider and
// chart tooltips. No libraries.
(function () {
  "use strict";
  var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var ICON_PAUSE = '<svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><rect x="3" y="2" width="3.5" height="12" rx="1"/><rect x="9.5" y="2" width="3.5" height="12" rx="1"/></svg>';
  var ICON_PLAY = '<svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M4 2.5v11a.8.8 0 001.2.7l9-5.5a.8.8 0 000-1.4l-9-5.5A.8.8 0 004 2.5z"/></svg>';

  // Videos: muted loops that play only while on screen. A visitor who pauses one keeps it paused.
  // With reduced motion nothing plays on its own and the native controls are shown instead.
  var videos = Array.prototype.slice.call(document.querySelectorAll("video[data-play]"));
  function setIcon(v) {
    var b = v.parentNode.querySelector(".pp");
    if (!b) return;
    var playing = !v.paused;
    b.innerHTML = playing ? ICON_PAUSE : ICON_PLAY;
    b.setAttribute("aria-label", playing ? "Pause video" : "Play video");
  }
  videos.forEach(function (v) {
    if (reduce) { v.controls = true; return; }
    var b = document.createElement("button");
    b.type = "button";
    b.className = "pp";
    b.addEventListener("click", function () {
      if (v.paused) { v.dataset.userPaused = ""; v.play().catch(function () {}); }
      else { v.dataset.userPaused = "1"; v.pause(); }
    });
    v.parentNode.appendChild(b);
    v.addEventListener("play", function () { setIcon(v); });
    v.addEventListener("pause", function () { setIcon(v); });
    setIcon(v);
  });
  if (!reduce && "IntersectionObserver" in window) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        var v = e.target;
        if (e.isIntersecting && v.dataset.userPaused !== "1") v.play().catch(function () {});
        else if (!e.isIntersecting) v.pause();
      });
    }, { threshold: 0.35 });
    videos.forEach(function (v) { io.observe(v); });
  }

  // Depth slider
  var cmp = document.getElementById("compare");
  if (cmp) {
    var range = cmp.querySelector("input");
    var set = function () { cmp.style.setProperty("--pos", range.value + "%"); };
    range.addEventListener("input", set);
    set();
  }

  // The interactive point-cloud viewer loads only when asked (3.9 MB)
  var v3 = document.getElementById("viewer3d");
  if (v3) {
    v3.querySelector(".load3d").addEventListener("click", function (e) {
      var f = document.createElement("iframe");
      f.src = e.currentTarget.dataset.src;
      f.title = "Interactive point clouds: the camera's depth and the repaired depth of each pipeline";
      f.setAttribute("allowfullscreen", "");
      v3.innerHTML = "";
      v3.appendChild(f);
      f.focus();
    });
  }

  // Chart tooltips: hover or keyboard focus on a row
  var tip = document.getElementById("tip");
  function show(row, x, y) {
    tip.textContent = row.dataset.tip;
    tip.classList.add("on");
    var r = tip.getBoundingClientRect();
    var left = Math.min(Math.max(8, x + 14), window.innerWidth - r.width - 8);
    var top = y - r.height - 12 < 8 ? y + 18 : y - r.height - 12;
    tip.style.left = left + "px";
    tip.style.top = top + "px";
  }
  function hide() { tip.classList.remove("on"); }
  Array.prototype.forEach.call(document.querySelectorAll(".row[data-tip]"), function (row) {
    row.setAttribute("aria-label", row.dataset.tip);
    row.addEventListener("pointermove", function (e) { show(row, e.clientX, e.clientY); });
    row.addEventListener("pointerleave", hide);
    row.addEventListener("focus", function () {
      var r = row.getBoundingClientRect();
      show(row, r.left + r.width / 2, r.top);
    });
    row.addEventListener("blur", hide);
  });
  window.addEventListener("scroll", hide, { passive: true });

  // Copy BibTeX
  var copy = document.getElementById("copy-bib");
  if (copy) {
    copy.addEventListener("click", function () {
      var text = document.getElementById("bibtex").textContent;
      var label = copy.querySelector("span");
      var done = function () { label.textContent = "Copied"; setTimeout(function () { label.textContent = "Copy"; }, 2000); };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, function () { fallback(text); done(); });
      } else { fallback(text); done(); }
    });
  }
  function fallback(text) {
    var ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.style.position = "absolute";
    ta.style.left = "-9999px";
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand("copy"); } catch (e) { /* nothing more to try */ }
    document.body.removeChild(ta);
  }
})();
