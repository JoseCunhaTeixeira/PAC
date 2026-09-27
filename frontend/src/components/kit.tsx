import { useCallback, useMemo, useState, type InputHTMLAttributes, type ReactNode } from "react";
import { AlertCircleIcon, AlertIcon, CheckIcon, InfoIcon } from "./icons";
import { boundsOf, tipOf, useNumberDraft, WrongNumbers } from "./numbers";
import { PageArt, type ArtKind } from "./PageArt";
import "./kit.css";

// PAC's page kit: a page and its header, cards, fields, a segmented control, callouts and stat
// tiles. Every page is built from them, so that they look and read alike.

export function Page({
  icon,
  title,
  subtitle,
  actions,
  art,
  children,
}: {
  icon?: ReactNode;
  title: string;
  subtitle?: ReactNode;
  /** Under the header: what the page shows (a profile, a run). */
  actions?: ReactNode;
  /** The header's small artwork. */
  art?: ArtKind;
  children?: ReactNode;
}) {
  // The page's wrong number fields: its run bar waits until none is left.
  const [wrong, setWrong] = useState<ReadonlySet<string>>(new Set());
  const mark = useCallback(
    (id: string, is: boolean) =>
      setWrong((ids) => {
        if (ids.has(id) === is) return ids;
        const next = new Set(ids);
        if (is) next.add(id);
        else next.delete(id);
        return next;
      }),
    [],
  );
  const numbers = useMemo(() => ({ count: wrong.size, mark }), [wrong, mark]);
  return (
    <WrongNumbers.Provider value={numbers}>
    <div className="page">
      <header className="page-hero">
        <div className="page-hero-text">
          {icon && <div className="page-icon">{icon}</div>}
          <div>
            <h1>{title}</h1>
            {subtitle && <p className="page-sub">{subtitle}</p>}
          </div>
        </div>
        {art && (
          <div className="page-hero-art">
            <PageArt kind={art} />
          </div>
        )}
      </header>
      {actions && <div className="page-toolbar">{actions}</div>}
      {children}
    </div>
    </WrongNumbers.Provider>
  );
}

export function Card({
  title,
  icon,
  hint,
  aside,
  step,
  className = "",
  children,
}: {
  title?: ReactNode;
  icon?: ReactNode;
  /** A line under the title: what the card sets or shows. */
  hint?: ReactNode;
  /** On the head's right: a control, a count. */
  aside?: ReactNode;
  /** The card's number in a workflow. */
  step?: number;
  className?: string;
  children?: ReactNode;
}) {
  return (
    <section className={`card ${className}`.trim()}>
      {(title || aside) && (
        <div className="card-head">
          <div className="card-title" data-tip={typeof hint === "string" ? hint : undefined}>
            {step !== undefined ? <span className="card-step">{step}</span> : icon && <span className="card-icon">{icon}</span>}
            <div>
              {title && <h2>{title}</h2>}
              {hint && typeof hint !== "string" && <p className="card-hint">{hint}</p>}
            </div>
          </div>
          {aside && <div className="card-aside">{aside}</div>}
        </div>
      )}
      {children}
    </section>
  );
}

/** Fields side by side, as many as fit. */
export function Fields({ children, min = 170 }: { children: ReactNode; min?: number }) {
  return (
    <div className="fields" style={{ gridTemplateColumns: `repeat(auto-fill, minmax(${min}px, 1fr))` }}>
      {children}
    </div>
  );
}

/** A number typed freely, in a table or a line (NumberField in a card): a field wrong (left
 * empty, beyond its bounds) turns red and says why on hover; the run bar waits for it (see
 * numbers.ts). */
export function NumberInput({
  value,
  onChange,
  min,
  max,
  optional,
  check,
  unit,
  onFocus,
  onBlur,
  ...rest
}: {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  /** May be left empty (NaN): the word shown then, such as "auto". */
  optional?: string;
  /** A rule another field moves (numbers.ts's `above`, `below`, `upTo`). */
  check?: (n: number) => string | null;
  /** The value's unit, for its hover's bounds and its issues. */
  unit?: string;
} & Omit<InputHTMLAttributes<HTMLInputElement>, "value" | "onChange" | "min" | "max" | "type">) {
  const { input, issue } = useNumberDraft({ value, onChange, min, max, optional, check, unit });
  // What it means (its data-tip), then its bounds.
  const tip = tipOf((rest as { "data-tip"?: string })["data-tip"], boundsOf(min, max, unit));
  return (
    <input
      {...rest}
      {...input}
      className={[rest.className, input.className].filter(Boolean).join(" ") || undefined}
      placeholder={optional ?? rest.placeholder}
      data-tip={issue ?? tip}
      data-tip-tone={issue ? "danger" : undefined}
      onFocus={(e) => {
        input.onFocus();
        onFocus?.(e);
      }}
      onBlur={(e) => {
        input.onBlur();
        onBlur?.(e);
      }}
    />
  );
}

export function NumberField({
  label,
  unit,
  value,
  onChange,
  min,
  max,
  step = 1,
  hint,
  title,
  disabled = false,
  optional,
  check,
}: {
  label: ReactNode;
  unit?: string;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  /** A short read-out under the field (the length in metres). */
  hint?: ReactNode;
  /** What the field means, on hover. */
  title?: string;
  disabled?: boolean;
  /** May be left empty (NaN): the word shown then, such as "auto". */
  optional?: string;
  /** A rule another field moves (numbers.ts's `above`, `below`, `upTo`). */
  check?: (n: number) => string | null;
}) {
  // Wrong (left empty, beyond its bounds, against its rule): red, and why under it, in place of
  // its hint.
  const { input, issue } = useNumberDraft({ value, onChange, min, max, optional, check, unit });
  // What the field means, then its bounds; right by what is hovered: over its name, under its
  // box.
  const tip = tipOf(title, boundsOf(min, max, unit));
  return (
    <label className="field">
      <span className="field-label" data-tip={tip} data-tip-place="above">
        {label}
      </span>
      <span className="field-control" data-tip={tip}>
        <input {...input} step={step} disabled={disabled} />
        {unit && <span className="field-unit">{unit}</span>}
      </span>
      {issue && (
        <span className="field-error" role="alert">
          <AlertCircleIcon size={13} />
          {issue}
        </span>
      )}
      {hint && !issue && <span className="field-hint">{hint}</span>}
    </label>
  );
}

export function SelectField({
  label,
  value,
  onChange,
  children,
  hint,
  icon,
  disabled = false,
}: {
  label?: ReactNode;
  value: string;
  onChange: (value: string) => void;
  children: ReactNode;
  hint?: ReactNode;
  icon?: ReactNode;
  disabled?: boolean;
}) {
  return (
    <label className="field">
      {label && <span className="field-label">{label}</span>}
      <span className={"field-control select" + (icon ? " with-icon" : "")}>
        {icon && <span className="field-icon">{icon}</span>}
        <select value={value} onChange={(e) => onChange(e.target.value)} disabled={disabled}>
          {children}
        </select>
      </span>
      {hint && <span className="field-hint">{hint}</span>}
    </label>
  );
}

export interface SegmentOption<T extends string> {
  value: T;
  label: ReactNode;
  title?: string;
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
  size = "md",
}: {
  value: T;
  options: SegmentOption<T>[];
  onChange: (value: T) => void;
  label?: string;
  size?: "sm" | "md";
}) {
  return (
    <div className={`segmented ${size}`} role="group" aria-label={label}>
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          data-tip={option.title}
          aria-pressed={value === option.value}
          className={value === option.value ? "active" : ""}
          onClick={() => onChange(option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

type Tone = "info" | "warn" | "error" | "success";

const TONE_ICONS: Record<Tone, ReactNode> = {
  info: <InfoIcon size={17} />,
  warn: <AlertIcon size={17} />,
  error: <AlertIcon size={17} />,
  success: <CheckIcon size={17} />,
};

export function Callout({ tone = "info", title, children }: { tone?: Tone; title?: ReactNode; children?: ReactNode }) {
  return (
    <div className={`callout ${tone}`} role={tone === "error" ? "alert" : undefined}>
      <span className="callout-icon">{TONE_ICONS[tone]}</span>
      <div>
        {title && <strong>{title}</strong>}
        {children && <div className="callout-body">{children}</div>}
      </div>
    </div>
  );
}

export function Stats({ children }: { children: ReactNode }) {
  return <div className="stats">{children}</div>;
}

export function Stat({ label, value, sub, icon }: { label: ReactNode; value: ReactNode; sub?: ReactNode; icon?: ReactNode }) {
  return (
    <div className="stat">
      <div className="stat-label">
        {icon}
        {label}
      </div>
      <div className="stat-value">{value}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  );
}

export function Empty({ icon, title, children }: { icon?: ReactNode; title: ReactNode; children?: ReactNode }) {
  return (
    <div className="empty">
      {icon && <div className="empty-icon">{icon}</div>}
      <strong>{title}</strong>
      {children && <div className="empty-body">{children}</div>}
    </div>
  );
}

export function Badge({ tone = "neutral", children, title }: { tone?: "neutral" | "accent" | "ok" | "warn" | "bad"; children: ReactNode; title?: string }) {
  return (
    <span className={`badge ${tone}`} data-tip={title}>
      {children}
    </span>
  );
}
