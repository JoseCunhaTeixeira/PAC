import { ClockIcon, FolderIcon } from "../icons";
import { SelectField } from "../kit";
import { runLabel } from "./format";
import type { ProfileRuns } from "./types";

// Which run to show: a profile, then one of its runs, the newest first. A profile without a run
// shows its records alone. Every page that reads a run chooses it with these two.

export const RECORDS_ONLY = "";

export function RunPicker({
  profiles,
  profile,
  folder,
  onChange,
  recordsOnly = true,
  disabled = false,
}: {
  profiles: ProfileRuns[];
  profile: string;
  folder: string;
  onChange: (profile: string, folder: string) => void;
  /** Offer a profile's records without a run (Visualization's). */
  recordsOnly?: boolean;
  /** Frozen: a job runs on the run chosen. */
  disabled?: boolean;
}) {
  const shown = recordsOnly ? profiles : profiles.filter((one) => one.runs.length > 0);
  const current = shown.find((one) => one.profile === profile);
  return (
    <>
      <SelectField
        label="Profile"
        value={profile}
        icon={<FolderIcon size={15} />}
        disabled={disabled}
        onChange={(value) => {
          const next = shown.find((one) => one.profile === value);
          onChange(value, next?.runs[0]?.folder ?? RECORDS_ONLY);
        }}
      >
        {profile === "" && <option value="">Choose a profile…</option>}
        {shown.map((one) => (
          <option key={one.profile} value={one.profile}>
            {one.profile}
            {one.runs.length ? ` (${one.runs.length} run${one.runs.length > 1 ? "s" : ""})` : ""}
          </option>
        ))}
      </SelectField>
      {current && (
        <SelectField
          label="Run"
          value={folder}
          icon={<ClockIcon size={15} />}
          disabled={disabled}
          onChange={(value) => onChange(profile, value)}
        >
          {current.runs.map((run) => (
            <option key={run.folder} value={run.folder}>
              {runLabel(run)}
            </option>
          ))}
          {recordsOnly && current.records && <option value={RECORDS_ONLY}>Records only</option>}
        </SelectField>
      )}
    </>
  );
}
