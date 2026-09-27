import { API } from "../api";
import type { Job } from "./RunPanel";

// PAC's jobs as the pages find them again: a page left while its job ran shows it on its return.

/** The latest job of `kind` still running (a processing job, of `mode` when given), or null. */
export async function runningJob(kind: Job["kind"], mode?: unknown): Promise<Job | null> {
  try {
    const res = await fetch(`${API}/jobs`);
    if (!res.ok) return null;
    const jobs = (await res.json()) as Job[];
    return (
      [...jobs]
        .reverse()
        .find((one) => one.kind === kind && one.state === "running" && (mode === undefined || one.mode === mode)) ?? null
    );
  } catch {
    return null;
  }
}
