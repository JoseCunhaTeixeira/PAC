import type { Acquisition } from "../api";

// What a profile's records set the settings' limits by. Files that differ (their lengths, their
// triggers): the least of each is the reference, so that every record keeps to the limits; the
// pages list every value the files say.

/** The shortest record, s: its last sample's time. */
export function shortestRecord(acquisition: Acquisition): number {
  const durations = acquisition.durations.filter((d) => Number.isFinite(d));
  return durations.length ? Math.min(...durations) : 0;
}

/** One sample, s (the records share their sampling: sigpipe refuses a profile otherwise). */
export function sampleOf(acquisition: Acquisition): number {
  return 1 / (acquisition.sampling_frequencies[0] || 1);
}

/** Where the records' data end once the muting moves them by the trigger, the first to end:
 * each record's length less the trigger typed (empty, 0). As sigpipe checks a preset (its
 * resolving.py). */
export function dataEnd(acquisition: Acquisition, t0: number | null): number {
  const shift = t0 !== null && Number.isFinite(t0) ? t0 : 0;
  const ends = acquisition.durations.map((duration) => duration - shift);
  const finite = ends.filter((end) => Number.isFinite(end));
  return finite.length ? Math.min(...finite) : 0;
}

/** Each value once, from the least, as the pages list the files' lengths and triggers. */
export function distinct(values: readonly number[]): number[] {
  return [...new Set(values.filter((v) => Number.isFinite(v)).map((v) => +v.toFixed(4)))].sort((a, b) => a - b);
}
