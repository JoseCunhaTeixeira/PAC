import {
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
} from "react";
import { CopyIcon, GripIcon, LockIcon, StrataIcon, TrashIcon } from "./icons";
import { NumberInput } from "./kit";
import {
  copied,
  moved,
  removed,
  type Layers,
  type ThicknessLayer,
  type VsLayer,
} from "./layers";

// The seismic inversion's layered model, top down: each layer's Vs and thickness as a range the
// chains sample, or fixed at one value; the half-space last, with no thickness.
// A layer above the half-space is dragged to another place, copied below itself, or removed.

// A layer dragged by its handle: its place, the place it would take, how far it went, and the
// rows' boxes when the drag began. The rows stay in place, shifted by transforms, until the drop:
// moving the dragged row in the page would lose the pointer.
interface Drag {
  from: number;
  to: number;
  startY: number;
  dy: number;
  tops: number[];
  heights: number[];
}

export function LayerTable({
  layers,
  onChange,
}: {
  layers: Layers;
  onChange: (next: Layers) => void;
}) {
  const [drag, setDrag] = useState<Drag | null>(null);
  const body = useRef<HTMLTableSectionElement>(null);
  const count = layers.vs.length; // the half-space included
  const movable = count - 1; // the layers above it

  const setVs = (i: number, change: Partial<VsLayer>) =>
    onChange({
      ...layers,
      vs: layers.vs.map((layer, k) =>
        k === i ? { ...layer, ...change } : layer,
      ),
    });
  const setThickness = (i: number, change: Partial<ThicknessLayer>) =>
    onChange({
      ...layers,
      thickness: layers.thickness.map((layer, k) =>
        k === i ? { ...layer, ...change } : layer,
      ),
    });

  // A value fixed at the middle of its range, or sampled again within its range.
  function fixThickness(i: number) {
    const layer = layers.thickness[i];
    const middle =
      Math.round((layer.thickness_min + layer.thickness_max) * 5) / 10;
    setThickness(i, {
      thickness_fixed: layer.thickness_fixed != null ? null : middle,
    });
  }

  function fixVs(i: number) {
    const layer = layers.vs[i];
    const middle = Math.round((layer.vs_min + layer.vs_max) / 20) * 10;
    setVs(i, { vs_fixed: layer.vs_fixed != null ? null : middle });
  }

  function grab(event: ReactPointerEvent<HTMLButtonElement>, i: number) {
    // Frozen while a job runs: a disabled button still gets pointer events.
    const frozen = event.currentTarget.matches(":disabled");
    if (event.button !== 0 || movable < 2 || frozen) return;
    const rows = body.current?.querySelectorAll("tr");
    if (!rows) return;
    const boxes = [...rows]
      .slice(0, movable)
      .map((row) => row.getBoundingClientRect());
    event.currentTarget.setPointerCapture(event.pointerId);
    setDrag({
      from: i,
      to: i,
      startY: event.clientY,
      dy: 0,
      tops: boxes.map((box) => box.top),
      heights: boxes.map((box) => box.height),
    });
  }

  function slide(event: ReactPointerEvent<HTMLButtonElement>) {
    if (!drag) return;
    const { from, tops, heights } = drag;
    const last = movable - 1;
    // Kept among the layers above the half-space.
    const dy = Math.min(
      Math.max(event.clientY - drag.startY, tops[0] - tops[from]),
      tops[last] + heights[last] - heights[from] - tops[from],
    );
    // A row passed once the dragged one's middle reaches its own: the ends too, where it stops.
    const middle = tops[from] + dy + heights[from] / 2;
    let to = from;
    for (let k = from + 1; k <= last; k++)
      if (middle >= tops[k] + heights[k] / 2) to = k;
    for (let k = from - 1; k >= 0; k--)
      if (middle <= tops[k] + heights[k] / 2) to = k;
    setDrag({ ...drag, to, dy });
  }

  function drop() {
    if (!drag) return;
    if (drag.to !== drag.from) onChange(moved(layers, drag.from, drag.to));
    setDrag(null);
  }

  // The handle focused: the arrows move its layer, and the focus follows it; Escape ends a drag.
  function key(event: ReactKeyboardEvent<HTMLButtonElement>, i: number) {
    if (event.key === "Escape" && drag) {
      setDrag(null);
      return;
    }
    const to =
      event.key === "ArrowUp"
        ? i - 1
        : event.key === "ArrowDown"
          ? i + 1
          : null;
    if (to === null) return;
    event.preventDefault();
    if (to < 0 || to >= movable) return;
    onChange(moved(layers, i, to));
    requestAnimationFrame(() => {
      const handles =
        body.current?.querySelectorAll<HTMLButtonElement>(".layer-handle");
      handles?.[to]?.focus();
    });
  }

  // Where a row is drawn while a layer is dragged: the layer under the pointer, the rows between
  // its place and the one it would take shifted by its height.
  function offset(i: number): number {
    if (!drag) return 0;
    const { from, to, dy, heights } = drag;
    if (i === from) return dy;
    if (from < to && i > from && i <= to) return -heights[from];
    if (to < from && i >= to && i < from) return heights[from];
    return 0;
  }

  return (
    <div className="table-wrap">
      <table className={"layers" + (drag ? " sorting" : "")}>
        <thead>
          <tr>
            <th />
            <th colSpan={3}>Thickness (m)</th>
            <th colSpan={3}>Vs (m/s)</th>
          </tr>
          <tr className="sub">
            <th />
            <th />
            <th>min</th>
            <th>max</th>
            <th />
            <th>min</th>
            <th>max</th>
          </tr>
        </thead>
        <tbody ref={body}>
          {layers.vs.map((vs, i) => {
            const halfSpace = i === count - 1;
            const thickness = layers.thickness[i];
            const dy = offset(i);
            const name = halfSpace ? "Half-space" : `Layer ${i + 1}`;
            return (
              <tr
                key={i}
                className={drag?.from === i ? "dragged" : undefined}
                style={dy ? { transform: `translateY(${dy}px)` } : undefined}
              >
                <td>
                  <span className="layer-name">
                    {halfSpace ? (
                      <span className="layer-handle still">
                        <GripIcon size={14} />
                        <StrataIcon size={14} />
                        {name}
                      </span>
                    ) : (
                      <button
                        type="button"
                        className={
                          "layer-handle" + (movable < 2 ? " still" : "")
                        }
                        aria-label={`${name}: drag, or arrow keys, to move it`}
                        data-tip={
                          drag || movable < 2
                            ? undefined
                            : "Drag to move"
                        }
                        onPointerDown={(event) => grab(event, i)}
                        onPointerMove={slide}
                        onPointerUp={drop}
                        onPointerCancel={() => setDrag(null)}
                        onKeyDown={(event) => key(event, i)}
                      >
                        <GripIcon size={14} />
                        <StrataIcon size={14} />
                        {name}
                      </button>
                    )}
                    {!halfSpace && (
                      <span className="layer-actions">
                        <button
                          type="button"
                          className="ghost icon"
                          aria-label={`Copy ${name.toLowerCase()}`}
                          data-tip="Copy below"
                          onClick={() => onChange(copied(layers, i))}
                        >
                          <CopyIcon size={13} />
                        </button>
                        <button
                          type="button"
                          className="ghost icon layer-remove"
                          aria-label={`Remove ${name.toLowerCase()}`}
                          data-tip={
                            movable < 2
                              ? "Remove\nOne layer must stay"
                              : "Remove"
                          }
                          disabled={movable < 2}
                          onClick={() => onChange(removed(layers, i))}
                        >
                          <TrashIcon size={13} />
                        </button>
                      </span>
                    )}
                  </span>
                </td>
                {halfSpace || !thickness ? (
                  <td colSpan={3} className="faint center">
                    —
                  </td>
                ) : (
                  <>
                    <td className="lock">
                      <FixToggle
                        on={thickness.thickness_fixed != null}
                        what="thickness"
                        onToggle={() => fixThickness(i)}
                      />
                    </td>
                    {thickness.thickness_fixed != null ? (
                      <td colSpan={2}>
                        <NumberInput
                          className="layer-fixed"
                          aria-label={`${name}: thickness, fixed`}
                          min={0.1}
                          unit="m"
                          step={0.1}
                          value={thickness.thickness_fixed}
                          onChange={(v) =>
                            setThickness(i, { thickness_fixed: v })
                          }
                        />
                      </td>
                    ) : (
                      <>
                        <td>
                          <NumberInput
                            min={0.1}
                            unit="m"
                            step={0.1}
                            value={thickness.thickness_min}
                            onChange={(v) =>
                              setThickness(i, { thickness_min: v })
                            }
                          />
                        </td>
                        <td>
                          <NumberInput
                            min={0.1}
                            gt={thickness.thickness_min}
                            unit="m"
                            step={0.1}
                            value={thickness.thickness_max}
                            onChange={(v) =>
                              setThickness(i, { thickness_max: v })
                            }
                          />
                        </td>
                      </>
                    )}
                  </>
                )}
                <td className="lock">
                  <FixToggle
                    on={vs.vs_fixed != null}
                    what="Vs"
                    onToggle={() => fixVs(i)}
                  />
                </td>
                {vs.vs_fixed != null ? (
                  <td colSpan={2}>
                    <NumberInput
                      className="layer-fixed"
                      aria-label={`${name}: Vs, fixed`}
                      min={10}
                      unit="m/s"
                      step={10}
                      value={vs.vs_fixed}
                      onChange={(v) => setVs(i, { vs_fixed: v })}
                    />
                  </td>
                ) : (
                  <>
                    <td>
                      <NumberInput
                        min={10}
                        unit="m/s"
                        step={10}
                        value={vs.vs_min}
                        onChange={(v) => setVs(i, { vs_min: v })}
                      />
                    </td>
                    <td>
                      <NumberInput
                        min={10}
                        gt={vs.vs_min}
                        unit="m/s"
                        step={10}
                        value={vs.vs_max}
                        onChange={(v) => setVs(i, { vs_max: v })}
                      />
                    </td>
                  </>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/** A value's lock: fixed at one value, not sampled by the chains; or sampled within its range. */
function FixToggle({
  on,
  what,
  onToggle,
}: {
  on: boolean;
  what: string;
  onToggle: () => void;
}) {
  return (
    <button
      type="button"
      className={"ghost icon layer-lock" + (on ? " on" : "")}
      aria-pressed={on}
      aria-label={on ? `Sample the ${what} again` : `Fix the ${what}`}
      data-tip={
        on
          ? `Free the ${what}\nSampled within a range`
          : `Fix the ${what}\nOne value, not sampled`
      }
      onClick={onToggle}
    >
      <LockIcon size={13} />
    </button>
  );
}
