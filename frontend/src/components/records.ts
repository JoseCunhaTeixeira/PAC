import type { Acquisition } from "../api";

// What a profile's records set the settings' limits by. Files that differ (their lengths, their
// triggers): the least of each is the reference (the user, 2026-09-28), so that every record
// keeps to the limits; the pages list every value the files say.

/** The shortest record, s: its last sample's time. */
export function shortestRecord(acquisition: Acquisition): number {
  const durations = acquisition.durations.filter((d) => Number.isFinite(d));
  return durations.length ? Math.min(...durations) : 0;
}

/** One sample, s (the records share their sampling: sigpipe refuses a profile otherwise). */
export function sampleOf(acquisition: Acquisition): number {
  return 1 / (acquisition.sampling_frequencies[0] || 1);
}

/** Where the records' data end once the muting moves them by their trigger, the first to end:
 * each record's length less its shift, the trigger typed (emptied, 0) or, left to the files,
 * its own (0 when its file says none). As sigpipe checks a preset (its resolving.py). */
export function dataEnd(acquisition: Acquisition, t0: number | null): number {
  const ends = acquisition.durations.map((duration, i) => {
    const shift = t0 !== null ? (Number.isFinite(t0) ? t0 : 0) : (acquisition.triggers?.[i] ?? 0);
    return duration - shift;
  });
  const finite = ends.filter((end) => Number.isFinite(end));
  return finite.length ? Math.min(...finite) : 0;
}

/** Each value once, from the least, as the pages list the files' lengths and triggers. */
export function distinct(values: readonly number[]): number[] {
  return [...new Set(values.filter((v) => Number.isFinite(v)).map((v) => +v.toFixed(4)))].sort((a, b) => a - b);
}
