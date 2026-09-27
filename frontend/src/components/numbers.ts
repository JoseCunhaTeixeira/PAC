import { createContext, useContext, useEffect, useId, useState, type ChangeEvent } from "react";

// The numbers typed in a page's fields (kit's NumberInput and NumberField): typed freely, and
// checked. A field left empty (that may not be) or beyond its bounds is wrong: it says why, and
// the page's run bar waits until no field is.

/** The page's wrong fields (kit's `Page` keeps them): how many, and how a field says it is. */
export const WrongNumbers = createContext<{
  count: number;
  mark: (id: string, wrong: boolean) => void;
} | null>(null);

/** How many of the page's fields the run waits for. */
export function useWrongCount(): number {
  return useContext(WrongNumbers)?.count ?? 0;
}

/** What is wrong with `n`, in a few words; null when nothing is. */
export function issueOf(
  n: number,
  min?: number,
  max?: number,
  optional?: string,
  unit?: string,
): string | null {
  const u = unit ? ` ${unit}` : "";
  if (Number.isNaN(n)) return optional === undefined ? "Required" : null;
  if (min !== undefined && n < min) return `At least ${written(min)}${u}`;
  if (max !== undefined && n > max) return `At most ${written(max)}${u}`;
  return null;
}

/** A field's bounds, in its issues' words, for its hover: every number field says them. */
export function boundsOf(min?: number, max?: number, unit?: string): string | undefined {
  const u = unit ? ` ${unit}` : "";
  if (min !== undefined && max !== undefined) return `From ${written(min)} to ${written(max)}${u}`;
  if (min !== undefined) return `At least ${written(min)}${u}`;
  if (max !== undefined) return `At most ${written(max)}${u}`;
  return undefined;
}

/** A hover's lines: what the field means, then its bounds. */
export function tipOf(...lines: (string | undefined)[]): string | undefined {
  return lines.filter(Boolean).join("\n") || undefined;
}

const written = (n: number) => n.toLocaleString("en-US", { maximumFractionDigits: 6 });

// A field's rule that another field moves (a pair's upper value above its lower one), for
// NumberField's and NumberInput's `check`. The lower value unset (NaN): no rule.

/** More than `low`. */
export const above = (low: number) => (n: number) =>
  Number.isFinite(low) && n <= low ? `More than ${written(low)}` : null;

/** Less than `high`. */
export const below = (high: number) => (n: number) =>
  Number.isFinite(high) && n >= high ? `Less than ${written(high)}` : null;

/** At most `high`. */
export const upTo = (high: number) => (n: number) =>
  Number.isFinite(high) && n > high ? `At most ${written(high)}` : null;

const same = (a: number, b: number) => a === b || (Number.isNaN(a) && Number.isNaN(b));

/** A number field's state: its input's props, and its issue when it shows. The field may be
 * emptied and rewritten. A number within the bounds is taken as it is typed; emptied, the value
 * is NaN; beyond a bound, the value stays and the number stays as typed, wrong. An issue shows
 * once the field is left, and as it is typed in a field entered wrong. */
export function useNumberDraft({
  value,
  onChange,
  min,
  max,
  optional,
  check,
  unit,
}: {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  /** May be left empty (NaN): the word shown then, such as "auto". */
  optional?: string;
  /** A rule another field moves (`above`, `below`, `upTo`): its number is taken, wrong. */
  check?: (n: number) => string | null;
  /** The value's unit, in its issues' words. */
  unit?: string;
}) {
  // The text typed and the value it goes with: set aside once the value changes elsewhere.
  const [draft, setDraft] = useState<{ text: string; value: number } | null>(null);
  const [focused, setFocused] = useState(false);
  // Whether the field was wrong when entered: its issue then shows as it is typed.
  const [flagged, setFlagged] = useState(false);
  const typed = draft !== null && same(draft.value, value) ? draft.text : null;
  const n = typed === null ? value : typed === "" ? Number.NaN : Number(typed);
  const issue = issueOf(n, min, max, optional, unit) ?? (Number.isFinite(n) ? (check?.(n) ?? null) : null);
  const shown = issue !== null && (!focused || flagged) ? issue : null;

  const mark = useContext(WrongNumbers)?.mark;
  const id = useId();
  const wrong = issue !== null;
  useEffect(() => {
    if (!wrong || !mark) return;
    mark(id, true);
    return () => mark(id, false);
  }, [mark, id, wrong]);

  // A number typed beyond a bound that has moved since (the workers' when more windows are
  // selected), within it now: taken.
  const within = (m: number) => (min === undefined || m >= min) && (max === undefined || m <= max);
  const due = typed !== null && Number.isFinite(n) && within(n) && !same(n, value);
  useEffect(() => {
    if (due) onChange(n);
  });

  return {
    issue: shown,
    input: {
      type: "number",
      min,
      max,
      placeholder: optional,
      value: typed ?? (Number.isFinite(value) ? value : ""),
      className: shown ? "invalid" : undefined,
      "aria-invalid": shown ? true : undefined,
      onChange: (event: ChangeEvent<HTMLInputElement>) => {
        const text = event.target.value;
        const m = text === "" ? Number.NaN : Number(text);
        const taken = Number.isNaN(m) || within(m);
        if (taken) onChange(m);
        setDraft({ text, value: taken ? m : value });
      },
      onFocus: () => {
        setFocused(true);
        setFlagged(issue !== null);
      },
      onBlur: () => {
        setFocused(false);
        setFlagged(false);
        // A number beyond the bounds stays as typed; any other shows as the value.
        if (!(Number.isFinite(n) && !within(n))) setDraft(null);
      },
    },
  } as const;
}
