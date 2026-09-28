import { useEffect, useState } from "react";
import { API } from "./api";

// sigpipe's processing settings for a mode, fitted to a profile (GET /presets/{mode}): the
// values a form starts from, and each stage's methods with their own values.
export interface PresetDefaults {
  values: Record<string, Record<string, unknown>>;
  methods: Record<string, Record<string, Record<string, unknown>>>;
}

export function usePreset(mode: string, profile: string) {
  const [preset, setPreset] = useState<PresetDefaults | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch(`${API}/presets/${mode}?profile=${encodeURIComponent(profile)}`)
      .then(async (res) => {
        const body = await res.json().catch(() => null);
        if (!res.ok) throw new Error(body?.detail ?? `HTTP ${res.status}`);
        return body as PresetDefaults;
      })
      .then((data) => {
        if (!cancelled) setPreset(data);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, [mode, profile]);

  return { preset, error };
}

// A stage's form state: the values of all its methods (switching method then keeps sensible
// numbers), overlaid with the preset's own method and values.
export function stage<T>(preset: PresetDefaults, name: string): T {
  const methods = Object.values(preset.methods[name] ?? {});
  return Object.assign({}, ...methods, preset.values[name]) as T;
}

// A bound may be empty (null, or NaN in its field): none, no stand-in value.
export interface MutingState { method: string; tmin: number | null; tmax: number | null; vmin: number | null; vmax: number | null; width: number | null; taper: number | null; }
// Empty (null, or NaN once emptied): 0, no shift (the user, 2026-09-28).
export interface TriggerState { t0: number | null; }

/** The trigger a form starts from: the preset's when given, else empty (the user, 2026-09-28:
 * unfilled at start, 0). */
export function triggerDefault(preset: PresetDefaults): TriggerState {
  const given = stage<TriggerState>(preset, "trigger").t0;
  return { t0: given !== null && given !== undefined ? given : null };
}
export interface FilteringState { method: string; fmin: number; fmax: number; order: number; }
export interface StackingState { method: string; nu: number; n: number; }
export interface SelectionState { method: string; threshold: number; vmin: number | null; vmax: number | null; }
export interface WhiteningState { method: string; fmin: number; fmax: number; taper_width_Hz: number; }
