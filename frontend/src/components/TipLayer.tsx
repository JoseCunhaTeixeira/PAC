import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { TooltipLines } from "./HoverTooltip";

// Every element with a `data-tip` says it on hover, as the plots' own tooltips do: its first line
// in bold, then one bullet a line (the lines split on "\n"). One layer for the whole app, so that
// every hover looks alike; the innermost element's tip wins. `data-tip-align="right"` opens it
// leftward (an element at a card's right edge), `data-tip-place="above"` over the element,
// `data-tip-tone="danger"` says what is wrong (a number field's issue) in red. A tip that would
// leave the window's bottom opens over its element.

interface Shown {
  lines: string[];
  x: number;
  top: number; // the element's edges
  bottom: number;
  align: "center" | "right";
  above: boolean;
  danger: boolean;
}

// A tip's distance from its element, and the least from the window's edges (px).
const GAP = 8;
const EDGE = 8;

export function TipLayer() {
  const [shown, setShown] = useState<Shown | null>(null);
  const tip = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function over(event: MouseEvent) {
      const target = event.target instanceof Element ? event.target : null;
      const element = target?.closest<HTMLElement>("[data-tip]") ?? null;
      const text = element?.dataset.tip;
      if (!element || !text) {
        setShown(null);
        return;
      }
      const box = element.getBoundingClientRect();
      const right = element.dataset.tipAlign === "right";
      const above = element.dataset.tipPlace === "above";
      setShown({
        lines: text.split("\n").filter((line) => line.trim()),
        x: right ? box.right : box.left + box.width / 2,
        top: box.top,
        bottom: box.bottom,
        align: right ? "right" : "center",
        above,
        danger: element.dataset.tipTone === "danger",
      });
    }
    function hide() {
      setShown(null);
    }
    document.addEventListener("mouseover", over);
    document.addEventListener("mouseleave", hide);
    window.addEventListener("scroll", hide, true);
    window.addEventListener("mousedown", hide, true);
    return () => {
      document.removeEventListener("mouseover", over);
      document.removeEventListener("mouseleave", hide);
      window.removeEventListener("scroll", hide, true);
      window.removeEventListener("mousedown", hide, true);
    };
  }, []);

  // Measured once drawn, before the screen shows it: under the element unless it would leave
  // the window's bottom and fits over it.
  useLayoutEffect(() => {
    const height = tip.current?.offsetHeight;
    if (!shown || shown.above || height === undefined) return;
    if (shown.bottom + GAP + height > window.innerHeight - EDGE && shown.top - GAP - height >= EDGE) {
      setShown({ ...shown, above: true });
    }
  }, [shown]);

  if (!shown) return null;
  // Kept on screen: centred under (or over) the element, clamped to the window's edges.
  const half = 150;
  const left = shown.align === "right" ? undefined : Math.min(Math.max(shown.x, half + 8), window.innerWidth - half - 8);
  return (
    <div
      ref={tip}
      className={shown.danger ? "tip-layer danger" : "tip-layer"}
      role="tooltip"
      style={{
        left,
        right: shown.align === "right" ? Math.max(8, window.innerWidth - shown.x) : undefined,
        top: shown.above ? undefined : shown.bottom + GAP,
        bottom: shown.above ? window.innerHeight - (shown.top - GAP) : undefined,
        transform: shown.align === "right" ? undefined : "translateX(-50%)",
      }}
    >
      <TooltipLines lines={shown.lines} />
    </div>
  );
}
