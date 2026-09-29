import { useEffect } from "react";

/** The browser tab's name: "PAC - " and the page's. */
export function usePageTitle(name: string): void {
  useEffect(() => {
    document.title = `PAC - ${name}`;
  }, [name]);
}
