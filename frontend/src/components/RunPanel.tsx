import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { API } from "../api";
import { CheckIcon, PlayIcon, StopIcon } from "./icons";
import { runningJob } from "./jobs";
import { readStored, writeStored } from "./stored";

interface WindowError {
  xmid: number;
  error_type: string;
  message: string;
  traceback: string;
}

export interface Job {
  id: string;
  kind: "processing" | "inversion" | "petro_inversion";
  mode?: string | null; // a processing job's mode
  target: string; // a processing job's profile, an inversion's run folder
  state: string; // running, succeeded, failed, stopped
  stopping?: boolean; // a stop was asked, and the job has not ended yet
  completed: number;
  total: number;
  elapsed: number | null;
  error: string | null;
  errors: WindowError[];
  run?: string | null; // a processing job's run, <profile>/<run_id>, once it has ended
}

// The job a page starts, by where it posts it.
const KIND_OF: Record<string, Job["kind"]> = {
  "/run": "processing",
  "/inversion/run": "inversion",
  "/petro_inversion/run": "petro_inversion",
};

function formatDuration(s: number): string {
  if (s < 60) return `${s.toFixed(1)} s`;
  if (s < 3600) {
    const m = Math.floor(s / 60);
    const sec = Math.round(s % 60);
    return `${m} min ${sec} s`;
  }
  const h = Math.floor(s / 3600);
  const m = Math.round((s % 3600) / 60);
  return `${h} h ${m} min`;
}

function normalizeJob(j: Job): Job {
  return { ...j, errors: j.errors ?? [] };
}

/** The page's main action, in a bar that stays at the bottom of the screen: what will run
 * (`summary`), the button, the job's progress and its Stop, its result, and what to do next
 * (`after`). A page left while its job ran finds it again when it comes back. */
export function RunPanel({
  config,
  runUrl = "/run",
  itemLabel = "windows",
  itemLabelSingular = "window",
  label = "Compute",
  summary,
  missing = [],
  onDone,
  onRunning,
  after,
}: {
  config: unknown;
  runUrl?: string;
  itemLabel?: string;
  itemLabelSingular?: string;
  label?: string;
  summary?: ReactNode;
  /** What the run still needs: the button waits for it. */
  missing?: string[];
  onDone?: (job: Job) => void;
  /** Whether the page's job runs: its settings frozen meanwhile. */
  onRunning?: (running: boolean) => void;
  /** Links shown once the job has results: it succeeded, or was stopped with some kept. */
  after?: (job: Job) => ReactNode;
}) {
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);
  // Read through a ref: pages pass a fresh callback on every render.
  const onDoneRef = useRef(onDone);
  useEffect(() => {
    onDoneRef.current = onDone;
  }, [onDone]);
  const kind = KIND_OF[runUrl];
  const mode = (config as { mode?: unknown } | null)?.mode;
  // What the page runs on: a computing page's profile, an inversion page's run.
  const target = (config as { profile?: unknown; folder?: unknown } | null)?.profile ?? (config as { folder?: unknown } | null)?.folder;
  // The last job this page started, kept for this tab: found again with its result.
  const jobKey = `pac.job.${kind}.${typeof mode === "string" ? mode : "any"}`;

  const stopPolling = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  const poll = useCallback((id: string) => {
    stopPolling();
    pollRef.current = window.setInterval(() => {
      fetch(`${API}/jobs/${id}`)
        .then(async (res) => {
          if (!res.ok) {
            const body = await res.json().catch(() => null);
            throw new Error(body?.detail ?? `HTTP ${res.status}`);
          }
          return res.json();
        })
        .then((j: Job) => {
          const nj = normalizeJob(j);
          setJob(nj);
          if (nj.state !== "running") {
            stopPolling();
            onDoneRef.current?.(nj);
          }
        })
        .catch((err) => {
          setError(err instanceof Error ? err.message : String(err));
          stopPolling();
        });
    }, 1000);
  }, []);

  useEffect(() => stopPolling, []);

  // Back on the page: the job of its kind still running (a computing page's, of its mode), else
  // the last one this page started, with its result, when it ran on what the page shows.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      let found = await runningJob(kind, mode);
      const last = readStored<string>(jobKey);
      if (!found && last) {
        const res = await fetch(`${API}/jobs/${last}`).catch(() => null);
        const kept = res?.ok ? ((await res.json()) as Job) : null;
        found = kept && kept.target === target ? kept : null;
      }
      if (cancelled || !found) return;
      setJob(normalizeJob(found));
      if (found.state === "running") poll(found.id);
    })();
    return () => {
      cancelled = true;
    };
  }, [kind, mode, target, jobKey, poll]);

  function compute() {
    setError(null);
    setJob(null);
    stopPolling();
    fetch(`${API}${runUrl}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    })
      .then(async (res) => {
        if (res.status === 422) {
          const body = await res.json();
          const msg = Array.isArray(body.detail)
            ? body.detail
                .map((e: { loc: (string | number)[]; msg: string }) => `${e.loc.slice(1).join(".")}: ${e.msg}`)
                .join("; ")
            : String(body.detail);
          throw new Error("Invalid settings: " + msg);
        }
        if (!res.ok) {
          const body = await res.json().catch(() => null);
          throw new Error(body?.detail ?? `HTTP ${res.status}`);
        }
        return res.json();
      })
      .then((j: Job) => {
        const nj = normalizeJob(j);
        writeStored(jobKey, nj.id);
        setJob(nj);
        poll(nj.id);
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }

  // At once: what had finished kept, nothing half-written left.
  function stop() {
    if (!job) return;
    fetch(`${API}/jobs/${job.id}/stop`, { method: "POST" })
      .then((res) => (res.ok ? (res.json() as Promise<Job>) : Promise.reject(new Error(`HTTP ${res.status}`))))
      .then((j) => setJob(normalizeJob(j)))
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }

  const running = job?.state === "running";
  const onRunningRef = useRef(onRunning);
  useEffect(() => {
    onRunningRef.current = onRunning;
  }, [onRunning]);
  useEffect(() => {
    onRunningRef.current?.(running);
  }, [running]);
  const pct = job && job.total > 0 ? Math.round((job.completed / job.total) * 100) : 0;
  const failed = job?.errors.length ?? 0;
  // A stopped processing job keeps a run only when windows had finished.
  const kept = job?.state === "stopped" && job.completed > 0 && (job.kind !== "processing" || !!job.run);
  const results = job?.state === "succeeded" || kept;

  return (
    <>
      <div className="run-bar">
        <div className="run-bar-info">
          {running ? (
            <>
              <div className="progress" aria-label="Progress">
                <span style={{ width: `${pct}%` }} />
              </div>
              <span>
                <b>
                  {job.completed} / {job.total}
                </b>{" "}
                {itemLabel}
              </span>
            </>
          ) : job?.state === "succeeded" ? (
            <span className={failed ? "run-warn" : "run-ok"}>
              <CheckIcon size={16} />
              <b>
                {job.total - failed}/{job.total}
              </b>{" "}
              {itemLabel} computed{job.elapsed != null && ` in ${formatDuration(job.elapsed)}`}
              {failed > 0 && `, ${failed} failed`}
            </span>
          ) : job?.state === "stopped" ? (
            <span className="run-warn">
              <StopIcon size={14} />
              {kept ? (
                <>
                  Stopped: <b>{job.completed}/{job.total}</b> {itemLabel} kept
                </>
              ) : (
                "Stopped: nothing kept"
              )}
            </span>
          ) : job?.state === "failed" ? (
            <span className="run-bad">
              Failed{job.elapsed != null && ` after ${formatDuration(job.elapsed)}`}: {job.error}
            </span>
          ) : missing.length > 0 ? (
            <span>Missing: {missing.join(", ")}.</span>
          ) : (
            summary
          )}
          {results && job && after?.(job)}
        </div>
        <div className="run-bar-actions">
          {running && (
            <button
              type="button"
              className="secondary large"
              onClick={stop}
              disabled={job.stopping}
              data-tip={"Stop\nAt once: what finished is kept\nNothing half-written is kept"}
            >
              {job.stopping ? (
                <>
                  <span className="spinner" /> Stopping…
                </>
              ) : (
                <>
                  <StopIcon size={14} /> Stop
                </>
              )}
            </button>
          )}
          <button type="button" className="large" onClick={compute} disabled={running || missing.length > 0}>
            {running ? (
              <>
                <span className="spinner" /> Computing…
              </>
            ) : (
              <>
                <PlayIcon size={15} /> {job ? `${label} again` : label}
              </>
            )}
          </button>
        </div>
      </div>

      {error && <div className="callout error run-error">{error}</div>}

      {failed > 0 && job && (
        <div className="card run-errors">
          <strong>
            {failed} {failed > 1 ? itemLabel : itemLabelSingular} failed
          </strong>
          {job.errors.map((e, i) => (
            <details key={i} className="fold">
              <summary>
                xmid {e.xmid.toFixed(2)}: {e.error_type}: {e.message}
              </summary>
              <pre>{e.traceback}</pre>
            </details>
          ))}
        </div>
      )}
    </>
  );
}
