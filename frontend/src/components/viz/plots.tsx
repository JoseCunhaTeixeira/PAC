import { useTheme } from "../../theme";
import { LinePlot, type PlotArea, type PlotRef, type PlotSeries } from "./LinePlot";
import { num } from "./format";
import { vizPalette } from "./palette";
import type { FitCurve, SoilColumn, VsProfile } from "./types";

// The plots of a selected window's card: its model's Vs against depth, with the depth its data
// inform; its picked curve against the one its model gives back; its soil column.

const MODEL_LABELS: Record<string, string> = {
  best: "best layered model",
  smooth_best: "smooth best model",
  median: "median layered model",
  smooth_median: "smooth median model",
  ensemble: "median of the ensemble",
};

/** Points of a step profile: each layer's value from its top to the next one's, the last down
 * to `bottom`. */
function steps(tops: number[], values: number[], bottom: number): [number, number][] {
  const points: [number, number][] = [];
  tops.forEach((top, i) => {
    const next = i + 1 < tops.length ? tops[i + 1] : bottom;
    points.push([values[i], top], [values[i], next]);
  });
  return points;
}

export function VsProfilePlot({ profile }: { profile: VsProfile }) {
  const palette = vizPalette(useTheme());
  const bottom = Math.max(profile.bottom, profile.tops[profile.tops.length - 1] ?? 0);
  const low = profile.vs.map((vs, i) => vs - profile.std[i]);
  const high = profile.vs.map((vs, i) => vs + profile.std[i]);
  const left = steps(profile.tops, low, bottom);
  const right = steps(profile.tops, high, bottom).reverse();
  const xs = [...low, ...high, ...profile.layered_vs];
  const xLo = Math.min(...xs);
  const xHi = Math.max(...xs);
  const areas: PlotArea[] = [{ color: palette.seriesSoft, polygon: [...left, ...right] }];
  const refs: PlotRef[] = [];
  const informed = profile.informed;
  if (informed !== null && informed < bottom) {
    // Below the depth the data inform, the samples spread as the prior does.
    areas.push({
      color: palette.selected,
      polygon: [
        [xLo - (xHi - xLo), informed],
        [xHi + (xHi - xLo), informed],
        [xHi + (xHi - xLo), bottom * 1.1],
        [xLo - (xHi - xLo), bottom * 1.1],
      ],
    });
    refs.push({
      axis: "y",
      at: informed,
      label: informed > 0 ? `informed to ${num(informed)} m` : "not informed",
      color: palette.status.warn,
    });
  }
  const series: PlotSeries[] = [
    {
      label: "Layered median",
      color: palette.muted,
      points: steps(profile.layered_tops, profile.layered_vs, bottom),
      dash: [5, 4],
      width: 1.4,
    },
    {
      label: MODEL_LABELS[profile.model] ?? profile.model,
      color: palette.series,
      points: steps(profile.tops, profile.vs, bottom),
      width: 2.2,
    },
  ];
  return (
    <div>
      <p className="viz-plot-title">Vs profile</p>
      <LinePlot
        series={series}
        areas={areas}
        refs={refs}
        xLabel="Vs (m/s)"
        yLabel="Depth (m)"
        yDown
        height={320}
        xRange={[Math.max(0, xLo - (xHi - xLo) * 0.08), xHi + (xHi - xLo) * 0.08]}
        yRange={[0, bottom * 1.04]}
        resetKey={profile}
      />
      <div className="viz-legend-inline">
        <span style={{ color: palette.series }}>
          <i />
          {MODEL_LABELS[profile.model] ?? profile.model}
        </span>
        <span>
          <i className="area" style={{ background: palette.seriesSoft }} />± its spread
        </span>
        <span style={{ color: palette.muted }}>
          <i className="dashed" />
          layered median
        </span>
        {informed !== null && informed < bottom && (
          <span>
            <i className="area" style={{ background: palette.selected, outline: `1px solid ${palette.faint}` }} />
            not informed by the data
          </span>
        )}
      </div>
    </div>
  );
}

export type CurveAxis = "frequency" | "wavelength";

export function CurveFitPlot({ curve, axis, modelled }: { curve: FitCurve; axis: CurveAxis; modelled: string }) {
  const palette = vizPalette(useTheme());
  const along = (f: number, v: number) => (axis === "frequency" ? f : v / f);
  const observed: PlotSeries = {
    label: `Picked ${curve.label}`,
    color: palette.series,
    points: curve.observed_fs.map((f, i) => [along(f, curve.observed_vs[i]), curve.observed_vs[i]]),
    errors: curve.observed_err.length ? curve.observed_err : undefined,
    dots: true,
    line: false,
  };
  const predicted: PlotSeries = {
    label: modelled,
    color: palette.status.fail,
    points: curve.predicted_fs.map((f, i) => [along(f, curve.predicted_vs[i]), curve.predicted_vs[i]]),
    dash: [6, 4],
    width: 2,
  };
  return (
    <div>
      <p className="viz-plot-title">Picked and modelled curve</p>
      <LinePlot
        series={[observed, predicted]}
        xLabel={axis === "frequency" ? "Frequency (Hz)" : "Wavelength (m)"}
        yLabel="Phase velocity (m/s)"
        height={320}
        resetKey={`${axis}-${curve.label}`}
      />
      <div className="viz-legend-inline">
        <span style={{ color: palette.series }}>● picked, ± its uncertainty</span>
        <span style={{ color: palette.status.fail }}>
          <i className="dashed" />
          {modelled}
        </span>
      </div>
    </div>
  );
}

// Soils in the colours geologists draw them with; any other a neutral tone.
const SOIL_COLOURS: [RegExp, string][] = [
  [/clay/i, "#c9a27e"],
  [/silt/i, "#d8c49a"],
  [/sand/i, "#eed98f"],
  [/gravel/i, "#c7b9a4"],
  [/peat|organic/i, "#8f7a5a"],
  [/rock|bedrock|marl|chalk|lime/i, "#a7adb5"],
  [/fill|made/i, "#b9b0c9"],
];

function soilColour(soil: string): string {
  return SOIL_COLOURS.find(([pattern]) => pattern.test(soil))?.[1] ?? "#cfd4da";
}

export function SoilColumnView({ column }: { column: SoilColumn }) {
  const height = 280;
  const total = column.thicknesses_m.reduce((a, b) => a + b, 0) || 1;
  // Each layer as tall as its thickness, at least a label's height; the half-space a fifth
  // more; all scaled into the column.
  const wanted = column.soils.map((_, i) => {
    const thickness = column.thicknesses_m[i];
    return Math.max(20, ((thickness ?? total * 0.2) / (total * 1.2)) * height);
  });
  const scale = height / wanted.reduce((a, b) => a + b, 0);
  const boxes = wanted.map((h) => h * scale);
  const tops = boxes.map((_, i) => boxes.slice(0, i).reduce((a, b) => a + b, 0));
  // The water table in the box of the layer it lies in.
  let water: number | null = null;
  if (column.water_table_m !== null) {
    let depth = 0;
    for (let i = 0; i < column.soils.length; i++) {
      const thickness = column.thicknesses_m[i] ?? Infinity;
      if (column.water_table_m <= depth + thickness || i === column.soils.length - 1) {
        const within = Number.isFinite(thickness) ? (column.water_table_m - depth) / thickness : 0;
        water = tops[i] + Math.min(1, Math.max(0, within)) * boxes[i];
        break;
      }
      depth += thickness;
    }
  }
  return (
    <div>
      <p className="viz-plot-title">Soil column</p>
      <div className="viz-column-wrap" style={{ height }}>
        <div className="viz-column" style={{ height }}>
          {column.soils.map((soil, i) => (
            <div key={i} style={{ background: soilColour(soil), height: boxes[i] }}>
              <span>{soil}</span>
              <span>
                {column.thicknesses_m[i] !== undefined ? `${num(column.thicknesses_m[i])} m` : "below"}
                {column.ns[i] !== undefined ? ` · N ${column.ns[i]}` : ""}
              </span>
            </div>
          ))}
        </div>
        {water !== null && column.water_table_m !== null && (
          <div className="viz-water" style={{ top: water }}>
            <span>water table {num(column.water_table_m)} m</span>
          </div>
        )}
      </div>
    </div>
  );
}
