export function buildFilteringParams(m: { method: string; fmin: number; fmax: number; order: number; }) {
  switch (m.method) {
    case "iir":
      return { method: "iir", fmin: m.fmin, fmax: m.fmax, order: m.order };
    default:
      return { method: "none" };
  }
}


// A value left empty (NaN in its field, or null) is sent as none.
export const bound = (value: number | null | undefined) =>
  value !== null && value !== undefined && Number.isFinite(value) ? value : null;


export function buildMutingParams(m: { method: string; tmin: number | null; tmax: number | null; vmin: number | null; vmax: number | null; width: number | null; taper: number | null; }) {
  switch (m.method) {
    case "mute":
      // The width left empty left out: sigpipe's default, one sample; the taper empty, none.
      return {
        method: "mute",
        tmin: bound(m.tmin),
        tmax: bound(m.tmax),
        vmin: bound(m.vmin),
        vmax: bound(m.vmax),
        ...(bound(m.width) !== null ? { width: bound(m.width) } : {}),
        taper: bound(m.taper) ?? 0,
      };
    default:
      return { method: "none" };
  }
}


// The windows' settings sent: a shot distance left empty left out, so that sigpipe's own default
// applies (none; 0 and 1,000 m in a sigpipe from before 2026-09-28, which refuses null).
export function buildMaswParams(m: { length: number; step: number; distance_min: number | null; distance_max: number | null; }) {
  const near = bound(m.distance_min);
  const far = bound(m.distance_max);
  return {
    length: m.length,
    step: m.step,
    ...(near !== null ? { distance_min: near } : {}),
    ...(far !== null ? { distance_max: far } : {}),
  };
}


// The trigger is part of the muting: its shift with the muting on (emptied: 0; null, untouched
// on files that differ: each record's own, from its file), none with it off.
export function buildTriggerParams(t: { t0: number | null; }, muting: { method: string; }) {
  if (muting.method !== "mute") return { t0: 0 };
  return { t0: t.t0 === null ? null : (bound(t.t0) ?? 0) };
}


export function buildNormalizationParams(m: { method: string; }) {
  switch (m.method) {
    case "onebit":
      return { method : "onebit"};
    default:
      return { method: "none"};
  }
}


export function buildSelectionParams(m: { method: string; threshold?: number; vmin?: number | null; vmax?: number | null; }) {
  switch (m.method) {
    case "fk":
      return { method : "fk", threshold : m.threshold, vmin: bound(m.vmin), vmax: bound(m.vmax) };
    default:
      return { method: "none"};
  }
}

export function buildStackingParams(m: { method: string; nu?: number; n?: number; }) {
  switch (m.method) {
    case "linear":
      return { method : "linear"}
    case "phase_weighted":
      return { method : "phase_weighted", nu : m.nu };
    case "root":
      return { method : "root", n : m.n };
    default:
      return { method: "none"};
  }
}


export function buildWhiteningParams(m: { method: string; fmin?: number; fmax?: number; taper_width_Hz?: number; }) {
  switch (m.method) {
    case "onebit":
      return { method : "onebit"};
    case "onebit_apod":
      return { method : "onebit_apod", fmin : m.fmin, fmax : m.fmax, taper_width_Hz : m.taper_width_Hz };
    default:
      return { method: "none"};
  }
}
