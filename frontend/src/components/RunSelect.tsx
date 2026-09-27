import { useEffect } from "react";
import { API } from "../api";
import { Callout } from "./kit";
import { RunPicker } from "./viz/RunPicker";
import type { ProfileRuns } from "./viz/types";
import { useJson } from "./viz/useJson";

// A run to work on, as Visualization chooses it: a profile, then one of its runs. With none
// chosen yet, the newest run of all.

export function RunSelect({
  folder,
  onChange,
  disabled = false,
}: {
  folder: string;
  onChange: (folder: string) => void;
  /** Frozen: a job runs on the run chosen. */
  disabled?: boolean;
}) {
  const runs = useJson<ProfileRuns[]>(`${API}/quality/runs`);
  const profiles = runs.data ?? [];
  const newest = profiles
    .flatMap((one) => one.runs)
    .sort((a, b) => (b.started_at ?? "").localeCompare(a.started_at ?? ""))[0]?.folder;

  useEffect(() => {
    if (!folder && newest) onChange(newest);
  }, [folder, newest, onChange]);

  if (runs.error) return <Callout tone="error">{runs.error}</Callout>;
  if (runs.data && !newest) return <Callout tone="warn" title="No run yet">Compute a profile first.</Callout>;
  const profile = profiles.find((one) => one.runs.some((run) => run.folder === folder))?.profile ?? "";
  return (
    <RunPicker
      profiles={profiles}
      profile={profile}
      folder={folder}
      onChange={(_, next) => onChange(next)}
      recordsOnly={false}
      disabled={disabled}
    />
  );
}
