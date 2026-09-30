import { useEffect, useRef, type ReactNode } from "react";

/** Its children, never shorter than they have been since `reset` or their width changed:
 * stepping from a unit to the next, a shorter one does not pull the page's end up over what is
 * being looked at, which would move the page. Another `reset` (another run, another stage)
 * lets them be as tall as they are. */
export function Steady({ reset, children }: { reset: string; children: ReactNode }) {
  const outer = useRef<HTMLDivElement>(null);
  const inner = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const box = outer.current;
    const content = inner.current;
    if (!box || !content) return;
    let floor = 0;
    let width = -1;
    const observer = new ResizeObserver(() => {
      const rect = content.getBoundingClientRect();
      if (rect.width !== width) {
        floor = 0;
        width = rect.width;
      }
      floor = Math.max(floor, rect.height);
      box.style.minHeight = `${floor}px`;
    });
    observer.observe(content);
    return () => {
      observer.disconnect();
      box.style.minHeight = "";
    };
  }, [reset]);
  return (
    <div ref={outer}>
      <div ref={inner} style={{ display: "flow-root" }}>
        {children}
      </div>
    </div>
  );
}
