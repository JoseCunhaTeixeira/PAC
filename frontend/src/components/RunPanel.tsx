import { useEffect, useRef, useState } from "react";
import { API } from "../api";

interface Job {
  id: string;
  state: string;
  completed: number;
  total: number;
  elapsed: number | null;
  error: string | null;
}

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

export function RunPanel({ config }: { config: unknown }) {
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);

  const stopPolling = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  useEffect(() => stopPolling, []);

  function poll(id: string) {
    pollRef.current = window.setInterval(() => {
      fetch(`${API}/jobs/${id}`)
        .then((res) => {
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          return res.json();
        })
        .then((j: Job) => {
          setJob(j);
          if (j.state !== "running") stopPolling();
        })
        .catch((err) => {
          setError(String(err));
          stopPolling();
        });
    }, 1000);
  }

  function compute() {
    setError(null);
    setJob(null);
    stopPolling();
    fetch(`${API}/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    })
      .then(async (res) => {
        if (res.status === 422) {
          const body = await res.json();
          const msg = body.detail
            .map((e: { loc: (string | number)[]; msg: string }) =>
              `${e.loc.slice(1).join(".")}: ${e.msg}`,
            )
            .join("; ");
          throw new Error("Invalid config — " + msg);
        }
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((j: Job) => {
        setJob(j);
        poll(j.id);
      })
      .catch((err) => setError(String(err)));
  }

  const running = job?.state === "running";
  const pct = job && job.total > 0 ? Math.round((job.completed / job.total) * 100) : 0;

  return (
    <div>
      <button onClick={compute} disabled={running}>
        {running ? "Computing…" : "Compute"}
      </button>
      {error && <p style={{ color: "crimson" }}>{error}</p>}
      {job && (
        <div style={{ marginTop: 10 }}>
          {running && (
            <>
              <div style={{ background: "#eee", borderRadius: 6, height: 14, maxWidth: 400, overflow: "hidden" }}>
                <div style={{ width: `${pct}%`, background: "#ff4b4b", height: "100%" }} />
              </div>
              <p>{job.completed} / {job.total} windows</p>
            </>
          )}
          {job.state === "succeeded" && (
            <p>
              ✓ Done — {job.total} windows computed
              {job.elapsed != null && ` in ${formatDuration(job.elapsed)}`}.
            </p>
          )}
          {job.state === "failed" && (
            <p style={{ color: "crimson" }}>
              ✗ Failed{job.elapsed != null && ` after ${formatDuration(job.elapsed)}`}: {job.error}
            </p>
          )}
        </div>
      )}
    </div>
  );
}