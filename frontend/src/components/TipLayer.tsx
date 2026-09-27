import { useEffect, useState } from "react";
import { TooltipLines } from "./HoverTooltip";

// Every element with a `data-tip` says it on hover, as the plots' own tooltips do: its first line
// in bold, then one bullet a line (the lines split on "\n"). One layer for the whole app, so that
// every hover looks alike; the innermost element's tip wins. `data-tip-align="right"` opens it
// leftward (an element at a card's right edge), `data-tip-place="above"` over the element,
// `data-tip-tone="danger"` says what is wrong (a number field's issue) in red.

interface Shown {
  lines: string[];
  x: number;
  y: number;
  align: "center" | "right";
  above: boolean;
  danger: boolean;
}

export function TipLayer() {
  const [shown, setShown] = useState<Shown | null>(null);

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
        y: above ? box.top - 8 : box.bottom + 8,
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

  if (!shown) return null;
  // Kept on screen: centred under (or over) the element, clamped to the window's edges.
  const half = 150;
  const left = shown.align === "right" ? undefined : Math.min(Math.max(shown.x, half + 8), window.innerWidth - half - 8);
  return (
    <div
      className={shown.danger ? "tip-layer danger" : "tip-layer"}
      role="tooltip"
      style={{
        left,
        right: shown.align === "right" ? Math.max(8, window.innerWidth - shown.x) : undefined,
        top: shown.above ? undefined : shown.y,
        bottom: shown.above ? window.innerHeight - shown.y : undefined,
        transform: shown.align === "right" ? undefined : "translateX(-50%)",
      }}
    >
      <TooltipLines lines={shown.lines} />
    </div>
  );
}
