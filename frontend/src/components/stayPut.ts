import { useEffect } from "react";

// What was clicked stays where it is on screen while the page around it changes (a card
// reloading above it, a message appearing): for HOLD_MS after the press, each frame scrolls its
// drift back, which the browser's own anchoring does not always do. Scrolling on purpose (the
// wheel, a touch, a key that scrolls, the scrollbar) lets it go.
const HOLD_MS = 2500;
// The keys that scroll the page, or move the focus, which scrolls to it.
const SCROLL_KEYS = new Set(["ArrowUp", "ArrowDown", "PageUp", "PageDown", "Home", "End", " ", "Tab"]);
let held: { element: Element; top: number; until: number } | null = null;
let frame = 0;

function release() {
  held = null;
  cancelAnimationFrame(frame);
}

function step() {
  if (!held) return;
  if (performance.now() > held.until || !held.element.isConnected) {
    release();
    return;
  }
  const drift = held.element.getBoundingClientRect().top - held.top;
  if (Math.abs(drift) >= 1) window.scrollBy(0, drift);
  frame = requestAnimationFrame(step);
}

/** Holds `element` where it is on screen for a moment: see above. */
export function holdInPlace(element: Element) {
  release();
  held = { element, top: element.getBoundingClientRect().top, until: performance.now() + HOLD_MS };
  frame = requestAnimationFrame(step);
}

/** Every press on the page holds what it pressed in place; one on the scrollbar (the document
 * itself) lets go, and an element marked `data-moves-page` scrolls on purpose. */
export function useStayPut() {
  useEffect(() => {
    function down(event: PointerEvent) {
      const target = event.target instanceof Element ? event.target : null;
      if (!target || target === document.documentElement || target === document.body) {
        release();
        return;
      }
      if (target.closest("[data-moves-page]")) return;
      holdInPlace(target);
    }
    function key(event: KeyboardEvent) {
      if (SCROLL_KEYS.has(event.key)) release();
    }
    const inputs = ["wheel", "touchstart"];
    document.addEventListener("pointerdown", down, true);
    window.addEventListener("keydown", key);
    for (const input of inputs) window.addEventListener(input, release, { passive: true });
    return () => {
      document.removeEventListener("pointerdown", down, true);
      window.removeEventListener("keydown", key);
      for (const input of inputs) window.removeEventListener(input, release);
      release();
    };
  }, []);
}
