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


export function buildMutingParams(m: { method: string; tmin: number | null; tmax: number | null; vmin: number | null; vmax: number | null; width: number; taper: number; }) {
  switch (m.method) {
    case "mute":
      return { method: "mute", tmin: bound(m.tmin), tmax: bound(m.tmax), vmin: bound(m.vmin), vmax: bound(m.vmax), width: m.width, taper: m.taper };
    default:
      return { method: "none" };
  }
}


// The trigger is part of the muting: its shift with the muting on (empty: each record's own,
// from its file), none with it off.
export function buildTriggerParams(t: { t0: number | null; }, muting: { method: string; }) {
  return { t0: muting.method === "mute" ? bound(t.t0) : 0 };
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
