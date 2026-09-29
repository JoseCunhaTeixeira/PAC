import { useTheme } from "../../theme";
import { useState, type ReactNode } from "react";
import { PlotBox } from "../kit";
import type { Range } from "../useZoom";
import { LinePlot, type PlotArea, type PlotRef, type PlotSeries } from "./LinePlot";
import { DEPTH_INFORMED_TIP, MODEL_LABELS, num } from "./format";
import { vizPalette } from "./palette";
import type { FitCurve, SoilColumn, VsProfile } from "./types";
import { PlotHead } from "./ui";

// The plots of a selected window's card: its model's Vs against depth, with the depth its data
// inform; its picked curve against the one its model gives back; its soil column.

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
  const { spread_depths: depths, spread_low: low, spread_high: high, uncertainty } = profile;
  const xs = [...low, ...high, ...profile.vs];
  const xLo = Math.min(...xs);
  const xHi = Math.max(...xs);
  // One zoom in depth for the profile and its uncertainty beside it, back to all of it with
  // another window.
  const [depth, setDepth] = useState<{ key: VsProfile; y: Range | null }>({ key: profile, y: null });
  const depthLink = {
    y: depth.key === profile ? depth.y : null,
    setY: (y: Range | null) => setDepth({ key: profile, y }),
  };
  // The kept models' 10-90 %, as the curve's band: down along the 90th, back up the 10th.
  const areas: PlotArea[] = depths.length
    ? [
        {
          color: palette.seriesSoft,
          polygon: [
            ...depths.map((z, i): [number, number] => [high[i], z]),
            ...depths.map((z, i): [number, number] => [low[i], z]).reverse(),
          ],
        },
      ]
    : [];
  const refs: PlotRef[] = [];
  const veils: PlotArea[] = [];
  const informed = profile.informed;
  if (informed !== null && informed < bottom) {
    // Below the depth the data inform, the kept models' Vs uncertainty over the limit.
    const veil = (lo: number, hi: number): PlotArea => ({
      color: palette.selected,
      polygon: [
        [lo, informed],
        [hi, informed],
        [hi, bottom * 1.1],
        [lo, bottom * 1.1],
      ],
    });
    areas.push(veil(xLo - (xHi - xLo), xHi + (xHi - xLo)));
    veils.push(veil(-1_000, 1_000));
    refs.push({
      axis: "y",
      at: informed,
      label: informed > 0 ? `informed to ${num(informed)} m` : "not informed",
      color: palette.status.warn,
    });
  }
  const series: PlotSeries[] = [
    {
      label: MODEL_LABELS[profile.model] ?? profile.model,
      color: palette.series,
      points: steps(profile.tops, profile.vs, bottom),
      width: 2.2,
    },
  ];
  // Their relative uncertainty U, %.
  const uncertain = depths.map((z, i): [number, number] => [uncertainty[i], z]);
  const uncertainMax = Math.max(10, ...uncertainty);
  const unlabelled = refs.map((ref) => ({ ...ref, label: "" }));
  const uncertaintyColour = palette.uncertainty;
  // Where they place layer boundaries: the share of them per interface_dz, as steps.
  const boundaries = profile.interfaces.flatMap((share, k): [number, number][] => [
    [share, k * profile.interface_dz],
    [share, (k + 1) * profile.interface_dz],
  ]);
  const boundaryMax = Math.max(10, ...profile.interfaces);
  const interfaceColour = palette.interfaces;
  return (
    <PlotBox>
      <div>
        <PlotHead title="Vs profile" />
        <div className="viz-profile-pair">
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
            depthLink={depthLink}
          />
          {uncertain.length > 0 && (
            <LinePlot
              series={[{ label: "uncertainty", color: uncertaintyColour, points: uncertain, width: 1.8 }]}
              areas={veils}
              refs={unlabelled}
              xLabel="Uncertainty (%)"
              yLabel="Depth (m)"
              yDown
              height={320}
              xRange={[0, uncertainMax * 1.08]}
              yRange={[0, bottom * 1.04]}
              resetKey={profile}
              minWidth={90}
              depthLink={depthLink}
              yAxis={false}
            />
          )}
          {boundaries.length > 0 && (
            <LinePlot
              series={[{ label: "interfaces", color: interfaceColour, points: boundaries, width: 1.6 }]}
              areas={veils}
              refs={unlabelled}
              xLabel="Interfaces (%)"
              yLabel="Depth (m)"
              yDown
              height={320}
              xRange={[0, boundaryMax * 1.08]}
              yRange={[0, bottom * 1.04]}
              resetKey={profile}
              minWidth={90}
              depthLink={depthLink}
              yAxis={false}
            />
          )}
        </div>
        {/* Two lines: the models, then what is read from them. */}
        <div className="viz-legend-rows">
          <div className="viz-legend-inline">
            <span style={{ color: palette.series }}>
              <i />
              {MODEL_LABELS[profile.model] ?? profile.model}
            </span>
            {depths.length > 0 && (
              <span>
                <i className="area" style={{ background: palette.seriesSoft }} />
                10–90 % of the models
              </span>
            )}
          </div>
          <div className="viz-legend-inline">
            {uncertain.length > 0 && (
              <span
                style={{ color: uncertaintyColour }}
                data-tip={"Vs uncertainty\nU = (P90 − P10) / (2 P50) of the kept models"}
              >
                <i />
                uncertainty
              </span>
            )}
            {boundaries.length > 0 && (
              <span
                style={{ color: interfaceColour }}
                data-tip={
                  "Interfaces\nThe share of the kept models placing a layer boundary in each " +
                  `${profile.interface_dz} m\nA peak: where they agree one lies; low and flat: anywhere`
                }
              >
                <i />
                interfaces
              </span>
            )}
            {informed !== null && informed < bottom && (
              <span data-tip={DEPTH_INFORMED_TIP}>
                <i className="area" style={{ background: palette.selected, outline: `1px solid ${palette.faint}` }} />
                not informed by the data
              </span>
            )}
          </div>
        </div>
      </div>
    </PlotBox>
  );
}

export type CurveAxis = "frequency" | "wavelength";

export function CurveFitPlot({
  curve,
  axis,
  modelled,
  aside,
}: {
  curve: FitCurve;
  axis: CurveAxis;
  modelled: string;
  /** On the title's line, at the right: the axis's switch. */
  aside?: ReactNode;
}) {
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
  // The kept models' spread, as the saved density figure draws it: up along their 90th percentile,
  // back along their 10th.
  const spreadFs = curve.spread_fs ?? [];
  const low = curve.spread_low ?? [];
  const high = curve.spread_high ?? [];
  const order = spreadFs.map((_, i) => i).sort((a, b) => spreadFs[a] - spreadFs[b]);
  const areas: PlotArea[] = order.length
    ? [
        {
          color: palette.modelledSoft,
          polygon: [
            ...order.map((i): [number, number] => [along(spreadFs[i], high[i]), high[i]]),
            ...[...order].reverse().map((i): [number, number] => [along(spreadFs[i], low[i]), low[i]]),
          ],
        },
      ]
    : [];
  return (
    <PlotBox>
      <div>
        <PlotHead title="Picked and modelled curve">{aside}</PlotHead>
        <LinePlot
          series={[observed, predicted]}
          areas={areas}
          xLabel={axis === "frequency" ? "Frequency (Hz)" : "Wavelength (m)"}
          yLabel="Phase velocity (m/s)"
          height={320}
          resetKey={`${axis}-${curve.label}`}
        />
        {/* Two lines: the picks, then the models. */}
        <div className="viz-legend-rows">
          <div className="viz-legend-inline">
            <span style={{ color: palette.series }}>● picked, ± its uncertainty</span>
          </div>
          <div className="viz-legend-inline">
            <span style={{ color: palette.status.fail }}>
              <i className="dashed" />
              {modelled}
            </span>
            {areas.length > 0 && (
              <span>
                <i className="area" style={{ background: palette.modelledSoft }} />
                10–90 % of the models
              </span>
            )}
          </div>
        </div>
      </div>
    </PlotBox>
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
      <div className="viz-row viz-plot-head">
        <p className="viz-plot-title">Soil column</p>
      </div>
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
