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

/** A number's bounds: inclusive (`min`, `max`) or strict (`gt`, `lt`: most often a pair's
 * other value, which moves); unset or NaN, none. */
export interface Bounds {
  min?: number;
  max?: number;
  gt?: number;
  lt?: number;
}

const finite = (n: number | undefined): n is number => n !== undefined && Number.isFinite(n);

/** Each side's tighter bound: its value, and whether the value itself is out. */
function sidesOf({ min, max, gt, lt }: Bounds) {
  const low = finite(gt) && !(finite(min) && min > gt) ? { at: gt, strict: true } : finite(min) ? { at: min, strict: false } : null;
  const high = finite(lt) && !(finite(max) && max < lt) ? { at: lt, strict: true } : finite(max) ? { at: max, strict: false } : null;
  return { low, high };
}

/** Whether `n` keeps to `bounds`. */
export function within(n: number, bounds: Bounds): boolean {
  const { low, high } = sidesOf(bounds);
  return (!low || (low.strict ? n > low.at : n >= low.at)) && (!high || (high.strict ? n < high.at : n <= high.at));
}

/** A field's bounds in symbols, a side a line, as its hover says them ("≥ 0 s", then "≤ 2.999
 * s"; "> 300 m/s") and its issue the side it breaks. */
export function boundsOf(bounds: Bounds, unit?: string): string | undefined {
  const u = unit ? ` ${unit}` : "";
  const { low, high } = sidesOf(bounds);
  const said = [
    low && `${low.strict ? ">" : "≥"} ${written(low.at)}${u}`,
    high && `${high.strict ? "<" : "≤"} ${written(high.at)}${u}`,
  ].filter(Boolean);
  return said.length ? said.join("\n") : undefined;
}

/** What is wrong with `n`, in a few words; null when nothing is: a bound broken, said as its
 * line of the hover ("≤ 2.999 s"). */
export function issueOf(
  n: number,
  bounds: Bounds,
  { optional, unit, whole = false }: { optional?: string; unit?: string; whole?: boolean },
): string | null {
  if (Number.isNaN(n)) return optional === undefined ? "Required" : null;
  if (whole && !Number.isInteger(n)) return "A whole number";
  const { low, high } = sidesOf(bounds);
  if (low && !within(n, low.strict ? { gt: low.at } : { min: low.at })) {
    return boundsOf(low.strict ? { gt: low.at } : { min: low.at }, unit) ?? null;
  }
  if (high && !within(n, high.strict ? { lt: high.at } : { max: high.at })) {
    return boundsOf(high.strict ? { lt: high.at } : { max: high.at }, unit) ?? null;
  }
  return null;
}

/** A hover's lines: what the field means, then its bounds. */
export function tipOf(...lines: (string | undefined)[]): string | undefined {
  return lines.filter(Boolean).join("\n") || undefined;
}

/** A number as a hover or a field writes it: no trailing zeros, six decimals at most. */
export const written = (n: number) => n.toLocaleString("en-US", { maximumFractionDigits: 6 });

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
  gt,
  lt,
  optional,
  unit,
  whole = false,
}: {
  value: number;
  onChange: (value: number) => void;
  /** Its bounds (see Bounds): a pair's other value, a strict one, that moves. */
  min?: number;
  max?: number;
  gt?: number;
  lt?: number;
  /** May be left empty (NaN): the word shown then, such as "auto". */
  optional?: string;
  /** The value's unit, in its issues' words. */
  unit?: string;
  /** A count (samples, steps, an order): a fraction is wrong. */
  whole?: boolean;
}) {
  // The text typed and the value it goes with: set aside once the value changes elsewhere.
  const [draft, setDraft] = useState<{ text: string; value: number } | null>(null);
  const [focused, setFocused] = useState(false);
  // Whether the field was wrong when entered: its issue then shows as it is typed.
  const [flagged, setFlagged] = useState(false);
  const typed = draft !== null && same(draft.value, value) ? draft.text : null;
  const n = typed === null ? value : typed === "" ? Number.NaN : Number(typed);
  const bounds = { min, max, gt, lt };
  const issue = issueOf(n, bounds, { optional, unit, whole });
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
  const due = typed !== null && Number.isFinite(n) && within(n, bounds) && !same(n, value);
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
        const taken = Number.isNaN(m) || within(m, bounds);
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
        if (!(Number.isFinite(n) && !within(n, bounds))) setDraft(null);
      },
    },
  } as const;
}
