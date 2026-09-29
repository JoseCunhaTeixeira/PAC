import { useEffect, useState } from "react";

interface Loaded<T> {
  url: string;
  data: T | null;
  error: string | null;
  status: number | null; // a failed answer's HTTP status
}

/** GET `url`'s JSON; null `url` fetches nothing. `data` belongs to `url`: while another loads,
 * the last one's stays for `stale`, and `data` is null. `version` fetches it again. `missing`:
 * the server has none (404, as a section of fewer than two windows); any other failure is an
 * `error` only, to be said as one. */
export function useJson<T>(url: string | null, version: unknown = 0) {
  const [loaded, setLoaded] = useState<Loaded<T> | null>(null);

  useEffect(() => {
    if (!url) return;
    let cancelled = false;
    fetch(url)
      .then(async (res) => {
        if (!res.ok) {
          const body = await res.json().catch(() => null);
          throw Object.assign(new Error(body?.detail ?? `HTTP ${res.status}`), { status: res.status });
        }
        return res.json() as Promise<T>;
      })
      .then((data) => {
        if (!cancelled) setLoaded({ url, data, error: null, status: null });
      })
      .catch((err) => {
        if (!cancelled) {
          const status = (err as { status?: number }).status ?? null;
          setLoaded({ url, data: null, error: err instanceof Error ? err.message : String(err), status });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [url, version]);

  const current = loaded && loaded.url === url ? loaded : null;
  return {
    data: current?.data ?? null,
    error: current?.error ?? null,
    missing: current?.status === 404,
    loading: url !== null && current === null,
    stale: loaded?.data ?? null,
  };
}
