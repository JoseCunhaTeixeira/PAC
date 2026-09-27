import { useEffect, useState } from "react";

interface Loaded<T> {
  url: string;
  data: T | null;
  error: string | null;
}

/** GET `url`'s JSON; null `url` fetches nothing. `data` belongs to `url`: while another loads,
 * the last one's stays for `stale`, and `data` is null. `version` fetches it again. */
export function useJson<T>(url: string | null, version: unknown = 0) {
  const [loaded, setLoaded] = useState<Loaded<T> | null>(null);

  useEffect(() => {
    if (!url) return;
    let cancelled = false;
    fetch(url)
      .then(async (res) => {
        if (!res.ok) {
          const body = await res.json().catch(() => null);
          throw new Error(body?.detail ?? `HTTP ${res.status}`);
        }
        return res.json() as Promise<T>;
      })
      .then((data) => {
        if (!cancelled) setLoaded({ url, data, error: null });
      })
      .catch((err) => {
        if (!cancelled) setLoaded({ url, data: null, error: err instanceof Error ? err.message : String(err) });
      });
    return () => {
      cancelled = true;
    };
  }, [url, version]);

  const current = loaded && loaded.url === url ? loaded : null;
  return {
    data: current?.data ?? null,
    error: current?.error ?? null,
    loading: url !== null && current === null,
    stale: loaded?.data ?? null,
  };
}
