import type { Theme } from "../../theme";
import { afmhotR, purples } from "../colormaps";
import type { PartState, Status, Use } from "./types";

// Visualization's canvas colours, which cannot take CSS variables: the states (pass green,
// warn amber, fail red, none grey, as the page's badges), one series (blue) for values and the
// shots a window stacks, and the categorical colours of the chains in a fixed order. Both modes
// are chosen for their own surface.
export function vizPalette(theme: Theme) {
  const dark = theme === "dark";
  const status: Record<Status, string> = {
    pass: dark ? "#34c46a" : "#1a9e4b",
    warn: dark ? "#f0b429" : "#d98a04",
    fail: dark ? "#f06a6a" : "#d63c3c",
    none: dark ? "#4a4d55" : "#d5d6da",
  };
  const series = dark ? "#4f9cf5" : "#2a78d6";
  // By hand: the app's indigo (index.css's --accent), as the rails and badges say it.
  const hand = dark ? "#6671ec" : "#4f5bd5";
  return {
    status,
    hand,
    /** A cell's part, by its state. */
    part: (state: PartState) => (state === "hand" ? hand : status[state]),
    series,
    seriesSoft: dark ? "rgba(79, 156, 245, 0.28)" : "rgba(42, 120, 214, 0.18)",
    // The models' spread around their curve, drawn in the curve's red.
    modelledSoft: dark ? "rgba(240, 106, 106, 0.3)" : "rgba(214, 60, 60, 0.2)",
    // How a window uses a shot: stacked, stacked with some traces left out, or left out (why).
    use: {
      used: series,
      part: series,
      excluded: status.fail,
      failed: status.fail,
      inside: dark ? "#6b6f78" : "#a9abb2",
      near: dark ? "#6b6f78" : "#a9abb2",
      far: dark ? "#6b6f78" : "#a9abb2",
      traces: status.warn,
    } satisfies Record<Use, string>,
    fail: status.fail,
    band: dark ? "rgba(52, 196, 106, 0.16)" : "rgba(26, 158, 75, 0.1)",
    reach: dark ? "rgba(79, 156, 245, 0.12)" : "rgba(42, 120, 214, 0.08)",
    limit: dark ? "#9a988f" : "#898781",
    grid: dark ? "#2a2c31" : "#ecebe6",
    reference: dark ? "#3a3d44" : "#e4e4e8", // a total a value is read against (the model's depth)
    faint: dark ? "#3a3d44" : "#e4e4e8",
    ink: dark ? "#e9e9ec" : "#1a1a1a",
    muted: dark ? "#aaadb3" : "#5a5a60",
    selected: dark ? "rgba(255, 255, 255, 0.08)" : "rgba(15, 15, 20, 0.05)",
    selectedEdge: dark ? "#f2f2f4" : "#1a1a1a",
    hover: dark ? "rgba(255, 255, 255, 0.28)" : "rgba(11, 11, 11, 0.22)",
    // The profile's U and interfaces, each in its section's colours (afmhot, purples).
    uncertainty: `rgb(${afmhotR(dark ? 0.6 : 0.65).join(", ")})`,
    interfaces: `rgb(${purples(dark ? 0.55 : 0.8).join(", ")})`,
    chains: dark
      ? ["#4f9cf5", "#e06a3a", "#22b07d", "#d4a017", "#e0689a", "#3fa33f", "#9a8cf0", "#ef7a7a"]
      : ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
  };
}

export type VizPalette = ReturnType<typeof vizPalette>;
