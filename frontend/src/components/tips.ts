/** What a plot says under the pointer, the same on every plot and page: the named thing there in
 * bold (a curve, a window, a shot, a receiver, a parameter; none over a bare position), its
 * values on one plain line ("x; y; z", with their units), then a bullet for each thing worth
 * adding. */
export interface Tip {
  title?: string;
  values?: string;
  notes?: readonly string[];
}

/** A value with the unit its axis names ("Vs (m/s)": "350 m/s"; "Uncertainty (%)": "12 %"). */
export function withUnit(label: string, value: string): string {
  const unit = /\(([^)]*)\)\s*$/.exec(label)?.[1];
  return unit ? `${value} ${unit}` : value;
}
