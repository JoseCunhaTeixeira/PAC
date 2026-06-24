export function HoverTooltip({
  x,
  y,
  lines,
}: {
  x: number;
  y: number;
  lines: string[];
}) {
  return (
    <div
      style={{
        position: "absolute",
        left: x + 10,
        top: y + 10,
        background: "rgba(0,0,0,0.8)",
        color: "white",
        padding: "3px 6px",
        borderRadius: 3,
        fontSize: 11,
        lineHeight: 1.4,
        pointerEvents: "none",
        whiteSpace: "nowrap",
        zIndex: 10,
      }}
    >
      {lines.map((line, i) => (
        <div key={i}>{line}</div>
      ))}
    </div>
  );
}
