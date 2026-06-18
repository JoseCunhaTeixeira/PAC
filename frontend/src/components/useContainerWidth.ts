import { useEffect, useRef, useState } from "react";

// Tracks the live content width of a wrapper element via ResizeObserver, so
// canvases can rasterize at exactly their displayed size instead of being
// scaled by the browser after the fact (which blurs canvas-drawn text).
export function useContainerWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width;
      if (w) setWidth(w);
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return [ref, width] as const;
}
