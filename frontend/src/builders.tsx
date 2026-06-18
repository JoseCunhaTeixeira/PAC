export function buildFilteringParams(m: { method: any; fmin: any; fmax: any; order: any; }) {
  switch (m.method) {
    case "iir":
      return { method: "iir", fmin: m.fmin, fmax: m.fmax, order: m.order };
    default:
      return { method: "none" };
  }
}


export function buildMutingParams(m: { method: any; tmin: any; tmax: any; vmin: any; vmax: any; taper: any; }) {
  switch (m.method) {
    case "mute":
      return { method: "mute", tmin: m.tmin, tmax: m.tmax, vmin: m.vmin, vmax: m.vmax, taper: m.taper };
    default:
      return { method: "none" };
  }
}


export function buildNormalizationParams(m: { method: any; }) {
  switch (m.method) {
    case "onebit":
      return { method : "onebit"};
    default:
      return { method: "none"};
  }
}


export function buildSelectionParams(m: { method: any; threshold?: number; vmin?: number; vmax?: number; }) {
  switch (m.method) {
    case "fk":
      return { method : "fk", threshold : m.threshold, vmin: m.vmin, vmax: m.vmax };
    default:
      return { method: "none"};
  }
}

export function buildStackingParams(m: { method: any; nu?: number; n?: number; }) {
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


export function buildWhiteningParams(m: { method: any; fmin?: number; fmax?: number; taper_width_Hz?: number; }) {
  switch (m.method) {
    case "onebit":
      return { method : "onebit"};
    case "onebit_apod":
      return { method : "onebit_apod", fmin : m.fmin, fmax : m.fmax, taper_width_Hz : m.taper_width_Hz };
    default:
      return { method: "none"};
  }
}