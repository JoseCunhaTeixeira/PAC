import { useEffect, useState } from "react";
import { API } from "../api";
import type { Job } from "./RunPanel";

// What runs, for the side menu to show beside each page: PAC's jobs (a processing job by its
// mode) and the assistant's conversations still answering. Asked every few seconds.

export interface Running {
  processing: ReadonlySet<string>; // the modes a processing job runs in
  inversion: boolean;
  petro: boolean;
  assistant: boolean;
}

const NOTHING: Running = { processing: new Set(), inversion: false, petro: false, assistant: false };
const EVERY_MS = 3000;

async function running(assistant: boolean): Promise<Running> {
  const jobs = await fetch(`${API}/jobs`)
    .then((res) => (res.ok ? (res.json() as Promise<Job[]>) : []))
    .catch(() => [] as Job[]);
  const sessions = assistant
    ? await fetch(`${API}/agent/sessions`)
        .then((res) => (res.ok ? (res.json() as Promise<{ busy: boolean }[]>) : []))
        .catch(() => [])
    : [];
  const live = jobs.filter((job) => job.state === "running");
  return {
    processing: new Set(live.filter((job) => job.kind === "processing").map((job) => job.mode ?? "")),
    inversion: live.some((job) => job.kind === "inversion"),
    petro: live.some((job) => job.kind === "petro_inversion"),
    assistant: sessions.some((one) => one.busy),
  };
}

/** What runs now, asked every few seconds; the assistant's only where PAC has it. */
export function useRunning(assistant: boolean): Running {
  const [now, setNow] = useState<Running>(NOTHING);
  useEffect(() => {
    let cancelled = false;
    const ask = () =>
      running(assistant).then((found) => {
        if (!cancelled) setNow(found);
      });
    void ask();
    const timer = window.setInterval(() => void ask(), EVERY_MS);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [assistant]);
  return now;
}

/** Whether the page at `to` has work running. */
export function runsAt(to: string, now: Running): boolean {
  switch (to) {
    case "/active":
      return now.processing.has("active");
    case "/passive":
      return now.processing.has("passive");
    case "/passive-active":
      return now.processing.has("passive-active");
    case "/seismic_inversion":
      return now.inversion;
    case "/petro_inversion":
      return now.petro;
    case "/assistant":
      return now.assistant;
    default:
      return false;
  }
}
