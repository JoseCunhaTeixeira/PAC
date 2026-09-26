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

export interface MutingState { method: string; tmin: number; tmax: number; vmin: number; vmax: number; taper: number; }
export interface FilteringState { method: string; fmin: number; fmax: number; order: number; }
export interface WindowState { method: string; vmin: number; vmax: number; taper: number; }
export interface StackingState { method: string; nu: number; n: number; }
export interface SelectionState { method: string; threshold: number; vmin: number; vmax: number; }
export interface WhiteningState { method: string; fmin: number; fmax: number; taper_width_Hz: number; }
