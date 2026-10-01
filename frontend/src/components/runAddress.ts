import { useEffect, useEffectEvent } from "react";
import { useSearchParams } from "react-router-dom";

/** A run named in the page's address (?run=<profile>/<run>, a link from the assistant's answer):
 * `choose` applies it once, then the address is cleared, the page's own choice from then on. */
export function useRunFromAddress(choose: (run: string) => void) {
  const [params, setParams] = useSearchParams();
  const asked = params.get("run");
  const apply = useEffectEvent(choose);
  useEffect(() => {
    if (!asked) return;
    Promise.resolve().then(() => {
      apply(asked);
      setParams(
        (previous) => {
          const next = new URLSearchParams(previous);
          next.delete("run");
          return next;
        },
        { replace: true },
      );
    });
  }, [asked, setParams]);
}
