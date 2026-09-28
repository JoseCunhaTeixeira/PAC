import { useCallback, useEffect, useState } from "react";
import { API } from "./api";
import {
  DispersionImageCanvas,
  type DispersionImage,
  type DragMode,
} from "./components/DispersionImageCanvas";
import {
  ModeHead,
  PseudoSectionCanvas,
  type PseudoSection,
} from "./components/PseudoSectionCanvas";
import {
  ArrowLeftIcon,
  ArrowRightIcon,
  CheckIcon,
  CrosshairIcon,
  ImageIcon,
  LassoIcon,
  RulerIcon,
  SparklesIcon,
  SpectrumIcon,
  StrataIcon,
  TrashIcon,
  ZoomIcon,
} from "./components/icons";
import { Badge, Callout, Card, Empty, Page, Segmented } from "./components/kit";
import { RunSelect } from "./components/RunSelect";
import {
  PositionRail,
  RailLegend,
  type RailCell,
  type RailTone,
} from "./components/PositionRail";
import { useStoredState } from "./components/stored";
import { judged, useReceivers, useStageStates, xmidKey } from "./components/railStates";
import { num } from "./components/viz/format";

const LABEL_PATTERN = /^[A-Z]{1,3}[0-9]+$/;
// The rail's parts' states in words, in the legend's order: a window's image's (G2), its
// curve's (G3, G4; or by hand).
const IMAGE_TONES: [RailTone, string][] = [
  ["pass", "passed"],
  ["warn", "flagged"],
  ["fail", "rejected"],
  ["none", "not checked"],
];
const CURVE_TONES: [RailTone, string][] = [
  ["pass", "passed"],
  ["warn", "flagged"],
  ["fail", "rejected"],
  ["auto", "not checked"],
  ["hand", "by hand"],
  ["none", "not picked"],
];
const LABEL_PARTS = /^([A-Z]{1,3})([0-9]+)$/;
// The line of the checks' hover naming the modes picked, which the page says as the picks are now.
const MODES_LINE = /^\d+ modes? picked: /;

function sanitizeLabel(raw: string): string {
  return raw
    .toUpperCase()
    .replace(/[^A-Z0-9]/g, "")
    .slice(0, 8);
}

function labelPrefix(currentLabel: string): string {
  return currentLabel.match(LABEL_PARTS)?.[1] ?? "M";
}

// Next free number for a given wave-type prefix (e.g. "M" or "L"), based on
// what's actually picked -- not just the picked count -- so it lands on the
// right number after deletions or mixed wave types leave gaps.
function nextLabel(prefix: string, curves: { label: string }[]): string {
  const taken = curves
    .map((c) => c.label.match(LABEL_PARTS))
    .filter((m): m is RegExpMatchArray => m !== null && m[1] === prefix)
    .map((m) => Number(m[2]));
  const next = taken.length > 0 ? Math.max(...taken) + 1 : 0;
  return `${prefix}${next}`;
}

// A position's picks, and who made them: PACo's picker, automatically, or a person, by hand.
interface PositionPicks {
  xmid: number;
  labels: string[];
  picked_by?: "auto" | "hand" | null;
}

export default function DispersionPickingPage() {
  // The run and the position, kept when the page is left.
  const [folder, setFolder] = useStoredState("pac.picking.folder", "");

  const [xmids, setXmids] = useState<number[]>([]);
  const [xmid, setXmid] = useStoredState<number | null>(
    "pac.picking.xmid",
    null,
  );

  const [image, setImage] = useState<DispersionImage | null>(null);
  const [pendingPolygon, setPendingPolygon] = useState<
    [number, number][] | null
  >(null);
  const [label, setLabel] = useState("M0");
  // What a drag on the image does; the zoom itself stays from one position to the next.
  const [dragMode, setDragMode] = useStoredState<DragMode>(
    "pac.picking.drag",
    "lasso",
  );

  const [labelCounts, setLabelCounts] = useState<Record<string, number>>({});
  const [pseudoMode, setPseudoMode] = useStoredState<
    "frequency" | "wavelength"
  >("pac.picking.axis", "frequency");
  const [pseudoSections, setPseudoSections] = useState<
    Record<string, PseudoSection>
  >({});
  const [positionPicks, setPositionPicks] = useState<PositionPicks[]>([]);
  // Each window's checks, as Visualization shows them; again after every pick and deletion.
  const [picks, setPicks] = useState(0);
  const { states, loading: statesLoading } = useStageStates(
    folder,
    "dispersion",
    picks,
  );
  // The line's receivers: a window's cell is their spacing wide, whatever the step.
  const { receivers, loading: receiversLoading } = useReceivers(folder);

  const [error, setError] = useState<string | null>(null);
  // The window's M0 being picked automatically.
  const [autoPicking, setAutoPicking] = useState(false);
  // Starts true so the first render after picking a folder shows "Loading…"
  // instead of flashing "No positions found" before the effect below runs.
  const [loadingXmids, setLoadingXmids] = useState(true);

  const loadPseudoSection = useCallback(
    (folderName: string, labelValue: string) => {
      fetch(
        `${API}/dispersion_pseudo_section/${encodeURIComponent(folderName)}/${encodeURIComponent(labelValue)}`,
      )
        .then(async (res) => {
          if (!res.ok) {
            const body = await res.json().catch(() => null);
            throw new Error(body?.detail ?? `HTTP ${res.status}`);
          }
          return res.json();
        })
        .then((data: PseudoSection) =>
          setPseudoSections((prev) => ({ ...prev, [labelValue]: data })),
        )
        // Best-effort per label: a failure here shouldn't blank out the rest
        // of the page (picking, other labels).
        .catch(() => {});
    },
    [],
  );

  const refreshLabels = useCallback(
    (folderName: string) => {
      fetch(`${API}/dispersion_image_labels/${encodeURIComponent(folderName)}`)
        .then(async (res) => {
          if (!res.ok) {
            const body = await res.json().catch(() => null);
            throw new Error(body?.detail ?? `HTTP ${res.status}`);
          }
          return res.json();
        })
        .then((data: Record<string, number>) => {
          setLabelCounts(data);
          setPseudoSections((prev) =>
            Object.fromEntries(
              Object.entries(prev).filter(([lbl]) => lbl in data),
            ),
          );
          Object.entries(data)
            .filter(([, count]) => count >= 2)
            .forEach(([labelValue]) =>
              loadPseudoSection(folderName, labelValue),
            );
        })
        .catch((err) =>
          setError(err instanceof Error ? err.message : String(err)),
        );
    },
    [loadPseudoSection],
  );

  const refreshPositionPicks = useCallback((folderName: string) => {
    fetch(
      `${API}/dispersion_picks_by_position/${encodeURIComponent(folderName)}`,
    )
      .then(async (res) => {
        if (!res.ok) {
          const body = await res.json().catch(() => null);
          throw new Error(body?.detail ?? `HTTP ${res.status}`);
        }
        return res.json();
      })
      .then((data: PositionPicks[]) => setPositionPicks(data))
      .catch((err) =>
        setError(err instanceof Error ? err.message : String(err)),
      );
  }, []);

  useEffect(() => {
    // The position is the run's own: choosing another run clears it (chooseRun).
    Promise.resolve().then(() => {
      setImage(null);
      setLabelCounts({});
      setPseudoSections({});
      setPositionPicks([]);
    });
    if (!folder) {
      Promise.resolve().then(() => setXmids([]));
      return;
    }
    Promise.resolve().then(() => setLoadingXmids(true));
    fetch(`${API}/xmids/${encodeURIComponent(folder)}`)
      .then(async (res) => {
        if (!res.ok) {
          const body = await res.json().catch(() => null);
          throw new Error(body?.detail ?? `HTTP ${res.status}`);
        }
        return res.json();
      })
      .then((data: number[]) => setXmids(data))
      .catch((err) =>
        setError(err instanceof Error ? err.message : String(err)),
      )
      .finally(() => setLoadingXmids(false));
    refreshLabels(folder);
    refreshPositionPicks(folder);
  }, [folder, refreshLabels, refreshPositionPicks]);

  function loadImage(folderName: string, xmidValue: number) {
    Promise.resolve().then(() => setError(null));
    fetch(
      `${API}/dispersion_images/${encodeURIComponent(folderName)}/${xmidValue}`,
    )
      .then(async (res) => {
        if (!res.ok) {
          const body = await res.json().catch(() => null);
          throw new Error(body?.detail ?? `HTTP ${res.status}`);
        }
        return res.json();
      })
      .then((data: DispersionImage) => {
        setImage(data);
        setPendingPolygon(null);
        setLabel(nextLabel("M", data.curves));
      })
      .catch((err) =>
        setError(err instanceof Error ? err.message : String(err)),
      );
  }

  useEffect(() => {
    if (folder && xmid !== null) loadImage(folder, xmid);
  }, [folder, xmid]);

  // The window's M0 picked automatically, as the assistant picks it: its M0 replaced, its other
  // modes kept.
  function handleAutoPick() {
    if (folder === "" || xmid === null) return;
    setAutoPicking(true);
    setError(null);
    fetch(`${API}/dispersion_images/${encodeURIComponent(folder)}/${xmid}/pick/auto`, { method: "POST" })
      .then(async (res) => {
        if (!res.ok) {
          const body = await res.json().catch(() => null);
          throw new Error(body?.detail ?? `HTTP ${res.status}`);
        }
        return res.json();
      })
      .then((data: DispersionImage) => {
        setImage(data);
        setLabel(nextLabel(labelPrefix(label), data.curves));
        refreshLabels(folder);
        refreshPositionPicks(folder);
        setPicks((n) => n + 1);
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setAutoPicking(false));
  }

  function handlePick() {
    if (!pendingPolygon || folder === "" || xmid === null) return;
    setError(null);
    fetch(
      `${API}/dispersion_images/${encodeURIComponent(folder)}/${xmid}/pick/lasso`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ polygon: pendingPolygon, label }),
      },
    )
      .then(async (res) => {
        if (!res.ok) {
          const body = await res.json().catch(() => null);
          throw new Error(body?.detail ?? `HTTP ${res.status}`);
        }
        return res.json();
      })
      .then((data: DispersionImage) => {
        setImage(data);
        setPendingPolygon(null);
        setLabel(nextLabel(labelPrefix(label), data.curves));
        refreshLabels(folder);
        refreshPositionPicks(folder);
        setPicks((n) => n + 1);
      })
      .catch((err) =>
        setError(err instanceof Error ? err.message : String(err)),
      );
  }

  function handleDelete(curveLabel: string) {
    if (folder === "" || xmid === null) return;
    setError(null);
    fetch(
      `${API}/dispersion_images/${encodeURIComponent(folder)}/${xmid}/pick/${encodeURIComponent(curveLabel)}`,
      {
        method: "DELETE",
      },
    )
      .then(async (res) => {
        if (!res.ok) {
          const body = await res.json().catch(() => null);
          throw new Error(body?.detail ?? `HTTP ${res.status}`);
        }
        return res.json();
      })
      .then((data: DispersionImage) => {
        setImage(data);
        setLabel(nextLabel(labelPrefix(label), data.curves));
        refreshLabels(folder);
        refreshPositionPicks(folder);
        setPicks((n) => n + 1);
      })
      .catch((err) =>
        setError(err instanceof Error ? err.message : String(err)),
      );
  }

  // ← and → step along the line, unless a field has the focus.
  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      const target = event.target as HTMLElement | null;
      if (target && ["INPUT", "SELECT", "TEXTAREA"].includes(target.tagName))
        return;
      if (xmids.length === 0) return;
      event.preventDefault();
      const at = xmid === null ? -1 : xmids.indexOf(xmid);
      const next =
        event.key === "ArrowRight"
          ? Math.min(xmids.length - 1, at + 1)
          : Math.max(0, at - 1);
      setXmid(xmids[next]);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [xmids, xmid, setXmid]);

  const picked = positionPicks.filter((position) => position.labels.length > 0);
  // Each window's curve, as its picks are now: by hand the user's, an automatic one by its
  // checks' verdict, none without a curve; and, the assistant having checked the images, its
  // image apart, above (as Visualization's line); the hover as Visualization's.
  const cells: RailCell[] = positionPicks.map((position) => {
    const state = states.get(xmidKey(position.xmid));
    const verdict = judged(state);
    const hasCurve = position.labels.length > 0;
    const hand = hasCurve && position.picked_by === "hand";
    const checked = state?.parts?.at(-1);
    const curve: RailTone = !hasCurve
      ? "none"
      : hand
        ? "hand"
        : checked === "pass" || checked === "warn" || checked === "fail"
          ? checked
          : "auto";
    const image = state?.parts && state.parts.length > 1 ? state.parts[0] : undefined;
    const tone: RailTone = image === undefined ? curve : hand ? "hand" : (verdict ?? curve);
    // The hover as Visualization's (its image, then its curve, apart), its curve and its modes
    // as the picks are now.
    const lines = state
      ? state.hover.filter((line) => !MODES_LINE.test(line))
      : [
          `xmid ${num(position.xmid, 4)} m`,
          hasCurve ? (hand ? "Curve: by hand" : "Curve: picked automatically") : "No curve",
        ];
    const at = lines.findIndex((line) => line.startsWith("Curve") || line === "No curve");
    const now = !hasCurve ? "No curve" : hand ? "Curve: by hand" : undefined;
    if (now !== undefined && at >= 0) lines[at] = now;
    const n = position.labels.length;
    if (hasCurve)
      lines.splice(
        at >= 0 ? at + 1 : lines.length,
        0,
        `${n} mode${n > 1 ? "s" : ""} picked: ${position.labels.join(", ")}`,
      );
    return {
      xmid: position.xmid,
      tone,
      title: lines.join("\n"),
      parts: image === undefined ? undefined : [image === "hand" ? "none" : image, curve],
      modes: position.labels,
    };
  });
  // What the rail's colours say, each part's states present: the image's, the curve's.
  const imageTones = new Set(cells.flatMap((cell) => (cell.parts ? [cell.parts[0]] : [])));
  const curveTones = new Set(cells.map((cell) => (cell.parts ? cell.parts[1] : cell.tone)));
  const legend = [
    ...(imageTones.size
      ? [{ title: "Image", items: IMAGE_TONES.filter(([tone]) => imageTones.has(tone)) }]
      : []),
    { title: "Curve", items: CURVE_TONES.filter(([tone]) => curveTones.has(tone)) },
  ];
  const index = xmid === null ? -1 : xmids.indexOf(xmid);
  const chooseRun = useCallback(
    (next: string) => {
      setXmid(null);
      setFolder(next);
    },
    [setFolder, setXmid],
  );

  return (
    <Page
      icon={<CrosshairIcon size={24} />}
      title="Dispersion picking"
      subtitle="Pick the curves on each dispersion image"
      art="picking"
      actions={<RunSelect folder={folder} onChange={chooseRun} />}
    >
      {error && <Callout tone="error">{error}</Callout>}
      {folder && !loadingXmids && xmids.length === 0 && (
        <Callout tone="warn">This run has no window.</Callout>
      )}

      {folder && positionPicks.length > 0 && (
        <div className="stack">
          <Card
            icon={<RulerIcon size={17} />}
            title="Windows"
            hint="Click one, or ← →."
            aside={
              <RailLegend groups={legend} modes={picked.some((position) => position.labels.length > 1)}>
                <Badge
                  tone={
                    picked.length === positionPicks.length ? "ok" : "neutral"
                  }
                >
                  {picked.length}/{positionPicks.length} picked
                </Badge>
              </RailLegend>
            }
          >
            {/* Once the checks and the receivers are in, so that no cell changes colour or size
                as they come. */}
            {!statesLoading && !receiversLoading && (
              <PositionRail
                cells={cells}
                receivers={receivers}
                isActive={(x) => x === xmid}
                onClick={setXmid}
              />
            )}
          </Card>

          {!image && (
            <Empty
              icon={<CrosshairIcon size={22} />}
              title="Choose a window"
            />
          )}

          {image && xmid !== null && (
            <div className="pick-grid">
              <Card
                icon={<ImageIcon size={17} />}
                title={`xmid ${xmid.toFixed(2)} m`}
                hint={
                  dragMode === "lasso"
                    ? "Lasso a mode: the picker follows its ridge."
                    : "Drag to zoom; double-click to go back."
                }
                aside={
                  <>
                    <button
                      type="button"
                      className="secondary icon"
                      data-tip="Previous window (←)"
                      disabled={index <= 0}
                      onClick={() => setXmid(xmids[index - 1])}
                    >
                      <ArrowLeftIcon size={16} />
                    </button>
                    <button
                      type="button"
                      className="secondary icon"
                      data-tip="Next window (→)"
                      disabled={index < 0 || index >= xmids.length - 1}
                      onClick={() => setXmid(xmids[index + 1])}
                    >
                      <ArrowRightIcon size={16} />
                    </button>
                    <Segmented
                      size="sm"
                      label="What a drag does"
                      value={dragMode}
                      onChange={setDragMode}
                      options={[
                        {
                          value: "lasso",
                          label: (
                            <>
                              <LassoIcon size={14} /> Lasso
                            </>
                          ),
                        },
                        {
                          value: "zoom",
                          label: (
                            <>
                              <ZoomIcon size={14} /> Zoom
                            </>
                          ),
                        },
                      ]}
                    />
                  </>
                }
              >
                <DispersionImageCanvas
                  image={image}
                  pendingPolygon={pendingPolygon}
                  onLassoComplete={setPendingPolygon}
                  dragMode={dragMode}
                />
                <div className="pick-bar" aria-label="Pick a new curve">
                  <button
                    type="button"
                    className={`pick-step${pendingPolygon ? " done" : dragMode === "lasso" ? " current" : ""}`}
                    data-tip={
                      "Lasso a ridge\nDraw around one mode: the picker follows its ridge"
                    }
                    onClick={() => setDragMode("lasso")}
                  >
                    <i>{pendingPolygon ? <CheckIcon size={12} /> : 1}</i> Lasso
                    a ridge
                  </button>
                  <ArrowRightIcon size={14} />
                  <label
                    className={`pick-step${pendingPolygon && LABEL_PATTERN.test(label) ? " done" : pendingPolygon ? " current" : ""}`}
                    data-tip={
                      "Name it\nM0: the fundamental mode, M1: the next…"
                    }
                  >
                    <i>
                      {pendingPolygon && LABEL_PATTERN.test(label) ? (
                        <CheckIcon size={12} />
                      ) : (
                        2
                      )}
                    </i>{" "}
                    Name it
                    <input
                      value={label}
                      onChange={(e) => setLabel(sanitizeLabel(e.target.value))}
                    />
                  </label>
                  <ArrowRightIcon size={14} />
                  <button
                    type="button"
                    onClick={handlePick}
                    disabled={!pendingPolygon || !LABEL_PATTERN.test(label)}
                  >
                    <span className="pick-number">3</span> Pick
                  </button>
                  <button
                    type="button"
                    className="ghost"
                    onClick={() => setPendingPolygon(null)}
                    disabled={!pendingPolygon}
                  >
                    Clear lasso
                  </button>
                  {/* Apart from the lasso's steps, and wrapped with its "or". */}
                  <span className="pick-auto">
                    or
                    <button
                      type="button"
                      className="tinted"
                      onClick={handleAutoPick}
                      disabled={autoPicking}
                      data-tip="Replaces this window's M0"
                    >
                      <SparklesIcon size={14} /> {autoPicking ? "Picking…" : "Auto-pick M0"}
                    </button>
                  </span>
                </div>
              </Card>

              <Card icon={<SpectrumIcon size={17} />} title="Picked curves">
                {image.curves.length === 0 ? (
                  <p className="faint">None yet.</p>
                ) : (
                  <ul className="curve-list">
                    {image.curves.map((curve) => (
                      <li key={curve.label}>
                        <span className="curve-dot" />
                        <strong>{curve.label}</strong>
                        <span className="faint">{curve.fs.length} points</span>
                        <button
                          type="button"
                          className="danger icon"
                          data-tip={`Delete ${curve.label}`}
                          onClick={() => handleDelete(curve.label)}
                        >
                          <TrashIcon size={15} />
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
              </Card>
            </div>
          )}

          {Object.keys(labelCounts).length > 0 && (
            <Card
              icon={<StrataIcon size={17} />}
              title="Pseudo-sections"
              hint="Click a column to open its window."
              aside={
                <Segmented
                  size="sm"
                  label="Vertical axis"
                  value={pseudoMode}
                  onChange={setPseudoMode}
                  options={[
                    { value: "frequency", label: "Frequency" },
                    { value: "wavelength", label: "Wavelength" },
                  ]}
                />
              }
            >
              <div className="stack">
                {Object.entries(labelCounts).map(([lbl, count]) => (
                  <div key={lbl}>
                    <ModeHead
                      label={lbl}
                      count={count}
                      total={xmids.length}
                      unit="windows"
                    />
                    {count < 2 ? (
                      <p className="faint">Needs 2 picked windows.</p>
                    ) : (
                      pseudoSections[lbl] && (
                        <PseudoSectionCanvas
                          section={pseudoSections[lbl]}
                          mode={pseudoMode}
                          height={220}
                          marker={xmid ?? undefined}
                          onPick={(position) =>
                            setXmid(
                              xmids.reduce(
                                (best, x) =>
                                  Math.abs(x - position) <
                                  Math.abs(best - position)
                                    ? x
                                    : best,
                                xmids[0],
                              ),
                            )
                          }
                        />
                      )
                    )}
                  </div>
                ))}
              </div>
            </Card>
          )}
        </div>
      )}
    </Page>
  );
}
