import { HandIcon, LassoIcon, ZoomIcon } from "./icons";
import type { SegmentOption } from "./kit";
import type { DragTool } from "./plotBox";

// What a drag can do in a box's plots, said by an icon and a hover (kit's BoxTools).

const ZOOM: SegmentOption<DragTool> = {
  value: "zoom",
  label: <ZoomIcon size={14} />,
  title: "Zoom\nDrag a box to zoom in; the wheel zooms too",
};

const HAND: SegmentOption<DragTool> = {
  value: "pan",
  label: <HandIcon size={14} />,
  title: "Hand\nDrag to move the view; the wheel, or a drag along an axis, zooms",
};

/** Every plot's tools: the hand, the default, and the zoom box. */
export const DRAG_TOOLS = [HAND, ZOOM];

/** The picking's: the lasso first, its default. */
export const PICK_TOOLS: SegmentOption<DragTool>[] = [
  { value: "lasso", label: <LassoIcon size={14} />, title: "Lasso\nDraw around a mode: the picker follows its ridge" },
  HAND,
  ZOOM,
];
