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
export interface MutingState { method: string; tmin: number | null; tmax: number | null; vmin: number | null; vmax: number | null; width: number; taper: number; }
// Null: each record's own trigger, from its file.
export interface TriggerState { t0: number | null; }

/** The trigger a form starts from: the preset's when given; else the files' when they all say
 * the same, 0 when none says one, null (each file's own) when they differ. */
export function triggerDefault(preset: PresetDefaults, triggers: (number | null)[] | undefined): TriggerState {
  const given = stage<TriggerState>(preset, "trigger").t0;
  if (given !== null && given !== undefined) return { t0: given };
  const all = triggers ?? [];
  const known = all.filter((t): t is number => t !== null);
  const said = [...new Set(known)];
  if (known.length === 0) return { t0: 0 };
  return { t0: said.length === 1 && known.length === all.length ? said[0] : null };
}
export interface FilteringState { method: string; fmin: number; fmax: number; order: number; }
export interface StackingState { method: string; nu: number; n: number; }
export interface SelectionState { method: string; threshold: number; vmin: number | null; vmax: number | null; }
export interface WhiteningState { method: string; fmin: number; fmax: number; taper_width_Hz: number; }
