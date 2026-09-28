import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { API, type Acquisition } from "./api";
import { DispersionPanel } from "./components/viz/DispersionPanel";
import { InversionPanel } from "./components/viz/InversionPanel";
import { neighbours } from "./components/viz/cells";
import { PetroPanel } from "./components/viz/PetroPanel";
import { ProfilePlot } from "./components/viz/ProfilePlot";
import { Gather, RecordsPanel } from "./components/viz/RecordsPanel";
import { RECORDS_ONLY, RunPicker } from "./components/viz/RunPicker";
import { RunSummary } from "./components/viz/RunSummary";
import type {
  Overview,
  PartState,
  ProfileRuns,
  RecordCard,
  RunCard,
  StageKey,
  Status,
  Use,
  WindowSources,
} from "./components/viz/types";
import { Empty, ErrorBox, Skeleton } from "./components/viz/ui";
import { useJson } from "./components/viz/useJson";
import { readStored, writeStored } from "./components/stored";
import { EyeIcon, PulseIcon } from "./components/icons";
import { Page, SelectField, Stat } from "./components/kit";
import { capitalized, STATE_MEANINGS, xmidOf } from "./components/viz/format";
import { lineExtent } from "./components/viz/line";
import { vizPalette } from "./components/viz/palette";
import { useTheme } from "./theme";
import type { Range } from "./components/useZoom";

// Visualization: one run at a time. Its card (who made it, its windows' length and step, the
// shots they stack, every setting and why), its line from above, and its stages as tabs: the
// selected window's (or record's) card with its plots, then the line's sections. What is shown
// lives in the address (run, stage, window, record), so a reload or a link shows it again.

const TABS: { key: StageKey; label: string }[] = [
  { key: "records", label: "Records" },
  { key: "dispersion", label: "Dispersion" },
  { key: "inversion", label: "Seismic inversion" },
  { key: "petro", label: "Petrophysics" },
];

const STATUSES: Status[] = ["pass", "warn", "fail", "none"];
// A part's states in the legend's order, and what each part checks, on its title's hover.
const PART_STATES: PartState[] = ["pass", "warn", "fail", "hand", "none"];
const PART_TIPS: Record<string, string> = {
  Image: "Image\nG2: the dispersion image's check",
  Curve: "Curve\nG3 and G4: the curve's checks, alone and along the line",
};

// What a window's (or record's) state implies for what comes after it, said on the line plot's
// hover: the badges' meanings, in short. None for a window with nothing: its own line says so.
const IMPLIES = Object.fromEntries(
  Object.entries(STATE_MEANINGS).map(([stage, meanings]) => [
    stage,
    Object.fromEntries(
      Object.entries(meanings)
        .filter(([, one]) => one.short)
        .map(([status, one]) => [status, one.short]),
    ),
  ]),
) as Record<StageKey, Partial<Record<Status, string>>>;

const at = (folder: string) => encodeURIComponent(folder);
// Where the page was: its address's query, for this tab.
const VIEW_KEY = "pac.visualization.view";

function RecordsOnly({ profile }: { profile: string }) {
  const acquisition = useJson<Acquisition>(
    `${API}/acquisitions/${at(profile)}`,
  );
  const [file, setFile] = useState<string | null>(null);
  if (acquisition.error) return <ErrorBox message={acquisition.error} />;
  if (!acquisition.data) return <Skeleton height={200} />;
  const files = acquisition.data.files;
  if (files.length === 0)
    return <Empty>The profile's folder holds no record.</Empty>;
  const shown = file && files.includes(file) ? file : files[0];
  return (
    <section className="viz-card">
      {/* As the computing pages count them, the record shown chosen beside. */}
      <div className="viz-records-head">
        <Stat label="Records" value={files.length} />
        <SelectField
          label="Record"
          value={shown}
          onChange={setFile}
          icon={<PulseIcon size={15} />}
        >
          {files.map((one) => (
            <option key={one} value={one}>
              {one}
            </option>
          ))}
        </SelectField>
      </div>
      <Gather profile={profile} file={shown} />
    </section>
  );
}

// The selected window's shots in the legend, by how it uses them: only the uses it has, and
// not those it leaves out by the geometry (inside it, out of its reach: grey, said on hover).
const SHOT_KEYS: { uses: Use[]; label: string; hollow?: boolean }[] = [
  { uses: ["used"], label: "stacked" },
  { uses: ["part"], label: "stacked, some traces left out", hollow: true },
  { uses: ["excluded"], label: "rejected" },
  { uses: ["failed"], label: "failed" },
  { uses: ["traces"], label: "too few receivers" },
];

function LineLegend({
  overview,
  windows,
  uses,
  meanings,
}: {
  overview: Overview | null;
  windows: boolean;
  /** How the selected window uses its shots; null: no window selected, or no shot. */
  uses: Set<Use> | null;
  /** What each state means, as the cards' badges say it. */
  meanings: Partial<Record<Status, string>>;
}) {
  // The states' colours, then the marks of the selected window: its shots and its receivers.
  const palette = vizPalette(useTheme());
  return (
    <div className="viz-line-legend">
      {overview &&
        (overview.parts?.length
          ? // The cells' parts apart (a window's image, its curve): a titled group each.
            overview.parts.map((part) => (
              <span key={part.title} className="viz-legend-group">
                <b data-tip={PART_TIPS[part.title]}>{part.title}</b>
                {PART_STATES.filter((state) => part.legend[state]).map((state) => (
                  <span key={state}>
                    <i className={`viz-swatch ${state}`} />
                    {part.legend[state]}
                  </span>
                ))}
              </span>
            ))
          : STATUSES.filter((status) => overview.legend[status]).map((status) => (
              <span
                key={status}
                data-tip={`${capitalized(overview.legend[status] ?? "")}\n${meanings[status] ?? ""}`}
              >
                <i className={`viz-swatch ${status}`} />
                {overview.legend[status]}
              </span>
            )))}
      {windows && uses && (
        <>
          {SHOT_KEYS.filter((key) => key.uses.some((use) => uses.has(use))).map(
            (key) => (
              <span key={key.label}>
                <span style={{ color: palette.use[key.uses[0]] }}>
                  {key.hollow ? "☆" : "★"}
                </span>{" "}
                {key.label}
              </span>
            ),
          )}
          <span>
            <span style={{ color: palette.series }}>▼</span> the window's
            receivers
          </span>
        </>
      )}
      {!windows && (
        <span>
          <i className="viz-swatch series" />{" "}
          windows stacking the record
        </span>
      )}
    </div>
  );
}

export default function VisualizationPage() {
  const [params, setParams] = useSearchParams();
  // Opened from the menu (no address of its own): where it was, for this tab; kept as it goes.
  useEffect(() => {
    const here = params.toString();
    if (here) {
      writeStored(VIEW_KEY, here);
      return;
    }
    const last = readStored<string>(VIEW_KEY);
    if (last)
      Promise.resolve().then(() =>
        setParams(new URLSearchParams(last), { replace: true }),
      );
  }, [params, setParams]);
  const runs = useJson<ProfileRuns[]>(`${API}/quality/runs`);
  const profiles = runs.data ?? [];
  const newest = profiles
    .flatMap((one) => one.runs)
    .filter((run) => run.started_at !== null)
    .sort((a, b) => (b.started_at ?? "").localeCompare(a.started_at ?? ""))[0];
  const folder = params.get("run") ?? newest?.folder ?? RECORDS_ONLY;
  const profile =
    params.get("profile") ??
    profiles.find((one) => one.runs.some((run) => run.folder === folder))
      ?.profile ??
    profiles[0]?.profile ??
    "";

  function update(changes: Record<string, string | null>) {
    setParams(
      (previous) => {
        const next = new URLSearchParams(previous);
        for (const [key, value] of Object.entries(changes)) {
          if (value === null) next.delete(key);
          else next.set(key, value);
        }
        return next;
      },
      { replace: true },
    );
  }

  const card = useJson<RunCard>(
    folder ? `${API}/quality/run/${at(folder)}` : null,
  );
  const run = card.data;
  const stages = run?.stages ?? [];
  const furthest =
    [...stages]
      .reverse()
      .find((stage) => stage.key !== "records" && stage.done > 0)?.key ??
    "dispersion";
  const tabParam = params.get("tab") as StageKey | null;
  const tab: StageKey =
    tabParam && stages.some((stage) => stage.key === tabParam)
      ? tabParam
      : furthest;
  const overview = useJson<Overview>(
    run ? `${API}/quality/${tab}/overview/${at(folder)}` : null,
  );
  const cells = useMemo(() => overview.data?.cells ?? [], [overview.data]);
  const records = tab === "records";

  const firstResult = cells.find((cell) => cell.status !== "none") ?? cells[0];
  const windowKey =
    params.get("x") ??
    (!records && firstResult
      ? firstResult.key
      : (run?.windows[0]?.key ?? null));
  const recordKey =
    params.get("rec") ?? (records && firstResult ? firstResult.key : null);
  const sources = useJson<WindowSources>(
    run && run.run_id !== null && windowKey && !records
      ? `${API}/quality/sources/${at(folder)}/${xmidOf(windowKey)}`
      : null,
  );
  const recordCard = useJson<RecordCard>(
    records && recordKey
      ? `${API}/quality/records/card/${at(folder)}/${encodeURIComponent(recordKey)}`
      : null,
  );

  const select = (key: string) => update(records ? { rec: key } : { x: key });

  // One zoom along the line for the line plot and the plots aligned under it, back to the
  // whole line with another run, and on the records' stage with another shot (the user).
  const extent = useMemo((): Range => (run ? lineExtent(run) : [0, 1]), [run]);
  const [zoomed, setZoomed] = useState<{
    folder: string;
    rec: string | null;
    x: Range | null;
  }>({
    folder,
    rec: null,
    x: null,
  });
  const lineX =
    zoomed.folder === folder && (!records || zoomed.rec === recordKey)
      ? zoomed.x
      : null;
  const setLineX = (x: Range | null) =>
    setZoomed({ folder, rec: records ? recordKey : null, x });

  // ← and → step to the previous and next unit with a result, but in a form's fields.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
      if (e.altKey || e.ctrlKey || e.metaKey) return;
      const target = e.target as HTMLElement | null;
      if (target?.closest("input, select, textarea, [contenteditable='true']"))
        return;
      const { before, after } = neighbours(
        cells,
        records ? recordKey : windowKey,
      );
      const next = e.key === "ArrowLeft" ? before : after;
      if (!next) return;
      e.preventDefault();
      setParams(
        (previous) => {
          const updated = new URLSearchParams(previous);
          updated.set(records ? "rec" : "x", next.key);
          return updated;
        },
        { replace: true },
      );
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [cells, records, recordKey, windowKey, setParams]);

  const panel = !run ? null : tab === "records" ? (
    <RecordsPanel
      folder={folder}
      card={recordCard.data}
      cardError={recordCard.error}
      overview={overview.data}
      overviewError={overview.error}
      onSelect={select}
      extent={extent}
      xZoom={lineX}
      onXZoom={setLineX}
    />
  ) : tab === "dispersion" ? (
    <DispersionPanel
      folder={folder}
      selected={windowKey}
      overview={overview.data}
      overviewError={overview.error}
      sources={sources.data}
      onSelect={select}
    />
  ) : tab === "inversion" ? (
    <InversionPanel
      folder={folder}
      selected={windowKey}
      overview={overview.data}
      overviewError={overview.error}
      onSelect={select}
    />
  ) : (
    <PetroPanel
      folder={folder}
      selected={windowKey}
      overview={overview.data}
      overviewError={overview.error}
      onSelect={select}
    />
  );

  return (
    <div className="viz">
      <Page
        icon={<EyeIcon size={24} />}
        title="Visualization"
        subtitle="Review a run's windows, checks and models"
        art="visualization"
        actions={
          profiles.length > 0 && (
            <RunPicker
              profiles={profiles}
              profile={profile}
              folder={folder}
              onChange={(nextProfile, nextFolder) =>
                update({
                  profile: nextProfile,
                  run: nextFolder,
                  tab: null,
                  x: null,
                  rec: null,
                })
              }
            />
          )
        }
      >
        {runs.error && <ErrorBox message={runs.error} />}
        {!runs.data && !runs.error && <Skeleton height={160} />}
        {runs.data && profiles.length === 0 && (
          <Empty>
            No profile in the input folder and no run in the output folder yet.
          </Empty>
        )}
        {runs.data && folder === RECORDS_ONLY && profile && (
          <RecordsOnly profile={profile} />
        )}

        {folder !== RECORDS_ONLY && card.error && (
          <ErrorBox message={card.error} />
        )}
        {folder !== RECORDS_ONLY && !run && !card.error && (
          <Skeleton height={190} />
        )}
        {run && (
          <>
            <RunSummary card={run} />
            <nav className="viz-tabs" aria-label="Stages">
              {TABS.filter((one) =>
                stages.some((stage) => stage.key === one.key),
              ).map((one) => {
                const stage = stages.find((item) => item.key === one.key);
                return (
                  <button
                    key={one.key}
                    type="button"
                    className={`viz-tab${tab === one.key ? " active" : ""}`}
                    onClick={() => update({ tab: one.key })}
                  >
                    {one.label}
                    {stage && (
                      <small
                        data-tip={`${one.label}\n${stage.done} of ${stage.total} ${one.key === "records" ? "records used" : "windows with a result"}`}
                      >
                        {stage.done}/{stage.total}
                      </small>
                    )}
                  </button>
                );
              })}
            </nav>
            {(run.windows.length > 0 ||
              Object.keys(run.sources).length > 0) && (
              <section className="viz-card">
                <ProfilePlot
                  card={run}
                  cells={overview.data?.cells ?? null}
                  implies={IMPLIES[tab]}
                  mode={records ? "records" : "windows"}
                  selected={records ? recordKey : windowKey}
                  sources={records ? null : sources.data}
                  stacking={records ? (recordCard.data?.windows ?? null) : null}
                  onSelect={select}
                  onShot={
                    run.records
                      ? (name) => update({ tab: "records", rec: name })
                      : undefined
                  }
                  onWindow={(key) => update({ tab: "dispersion", x: key })}
                  xZoom={lineX}
                  onXZoom={setLineX}
                />
                <LineLegend
                  overview={overview.data}
                  windows={!records}
                  uses={
                    !records && sources.data?.shots.length
                      ? new Set(sources.data.shots.map((shot) => shot.use))
                      : null
                  }
                  meanings={Object.fromEntries(
                    Object.entries(STATE_MEANINGS[tab]).map(([status, one]) => [
                      status,
                      one.full,
                    ]),
                  )}
                />
              </section>
            )}
            <div className="viz-section">{panel}</div>
          </>
        )}
      </Page>
    </div>
  );
}
