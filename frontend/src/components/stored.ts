import { useEffect, useState } from "react";

// What a page keeps when it is left: its choices and its last job, in this tab's session storage.
// The page opened again finds them; another tab, or a new session, starts afresh, and so does a
// browser that refuses the storage (a private window).

export function readStored<T>(key: string): T | undefined {
  try {
    const raw = sessionStorage.getItem(key);
    return raw === null ? undefined : (JSON.parse(raw) as T);
  } catch {
    return undefined;
  }
}

export function writeStored(key: string, value: unknown): void {
  try {
    if (value === undefined || value === null || value === "") sessionStorage.removeItem(key);
    else sessionStorage.setItem(key, JSON.stringify(value));
  } catch {
    // refused: the page just starts afresh next time
  }
}

/** useState, kept in this tab's session storage under `key`. */
export function useStoredState<T>(key: string, initial: T) {
  const [value, setValue] = useState<T>(() => readStored<T>(key) ?? initial);
  useEffect(() => {
    writeStored(key, value);
  }, [key, value]);
  return [value, setValue] as const;
}
