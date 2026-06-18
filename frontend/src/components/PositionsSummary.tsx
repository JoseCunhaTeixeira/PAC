import { useState } from "react";

export function PositionsSummary({ label, positions }: { label: string; positions: number[] }) {
  const [expanded, setExpanded] = useState(false);

  if (positions.length === 0) {
    return <p>{label} [m]: —</p>;
  }

  const sorted = [...positions].sort((a, b) => a - b);
  const min = sorted[0];
  const max = sorted[sorted.length - 1];
  const step = sorted[1] - sorted[0];
  const uniform =
    sorted.length > 1 &&
    sorted.every((p, i) => i === 0 || Math.abs(p - sorted[i - 1] - step) < 1e-6);

  const summary =
    sorted.length === 1
      ? `${min} m`
      : uniform
        ? `${min} – ${max} m · ${positions.length} positions · ${step.toFixed(2)} m spacing`
        : `${min} – ${max} m · ${positions.length} positions (irregular spacing)`;

  return (
    <p>
      {label} [m]: {expanded ? positions.join(", ") : summary}{" "}
      <button
        onClick={() => setExpanded((e) => !e)}
        style={{
          background: "none",
          boxShadow: "none",
          color: "var(--accent)",
          padding: "0 4px",
          fontSize: "0.85em",
          fontWeight: 600,
          textDecoration: "underline",
        }}
      >
        {expanded ? "Hide" : "Show all"}
      </button>
    </p>
  );
}
