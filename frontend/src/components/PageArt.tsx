// Each page's small artwork, in its header band: the home page's hero, in miniature, for what
// the page does. Shots are stars, receivers inverted triangles, as in every plot.

export type ArtKind =
  | "active"
  | "passive"
  | "passive-active"
  | "picking"
  | "inversion"
  | "petro"
  | "visualization"
  | "assistant";

const GROUND = 74;
const RECEIVERS = Array.from({ length: 11 }, (_, i) => 150 + i * 22);

function star(cx: number, cy: number, outer: number): string {
  const points: string[] = [];
  for (let k = 0; k < 10; k++) {
    const r = k % 2 === 0 ? outer : outer * 0.42;
    const a = -Math.PI / 2 + (k * Math.PI) / 5;
    points.push(`${(cx + r * Math.cos(a)).toFixed(1)},${(cy + r * Math.sin(a)).toFixed(1)}`);
  }
  return points.join(" ");
}

function Line({ pulse = 0 }: { pulse?: number }) {
  return (
    <>
      <path d={`M-600 ${GROUND} H400`} className="art-ground" />
      {RECEIVERS.map((x) => (
        <polygon
          key={x}
          className={pulse ? "art-receiver lit" : "art-receiver"}
          style={pulse ? { animationDelay: `${((x / 400) * pulse).toFixed(2)}s`, animationDuration: `${pulse}s` } : undefined}
          points={`${x - 4},${GROUND - 9} ${x + 4},${GROUND - 9} ${x},${GROUND - 2}`}
        />
      ))}
    </>
  );
}

function Shot({ x }: { x: number }) {
  return (
    <>
      <circle cx={x} cy={GROUND - 8} r={11} className="art-glow" />
      <polygon points={star(x, GROUND - 8, 7)} className="art-star" />
    </>
  );
}

function Fronts({ x, id }: { x: number; id: string }) {
  return (
    <>
      <defs>
        <clipPath id={id}>
          <rect x={-600} y={GROUND} width={1000} height={60} />
        </clipPath>
      </defs>
      <g clipPath={`url(#${id})`}>
        {[0, 1, 2].map((k) => (
          <circle
            key={k}
            className="art-front"
            cx={x}
            cy={GROUND}
            r={40}
            style={{ animationDelay: `${k * 1.2}s`, transformOrigin: `${x}px ${GROUND}px` }}
          />
        ))}
      </g>
    </>
  );
}

function Strata() {
  return (
    <>
      <rect x={-600} y={GROUND} width={1000} height={16} className="art-layer one" />
      <rect x={-600} y={GROUND + 16} width={1000} height={18} className="art-layer two" />
      <rect x={-600} y={GROUND + 34} width={1000} height={30} className="art-layer three" />
    </>
  );
}

function wave(x0: number, length: number, amplitude: number, y: number): string {
  const points: string[] = [];
  for (let i = 0; i <= 40; i++) {
    const x = x0 + (i / 40) * length;
    const envelope = Math.sin((Math.PI * i) / 40);
    points.push(`${x.toFixed(1)},${(y + amplitude * envelope * Math.sin(i * 0.9)).toFixed(1)}`);
  }
  return `M${points.join(" L")}`;
}

export function PageArt({ kind }: { kind: ArtKind }) {
  return (
    <svg className="page-art" viewBox="0 0 400 120" preserveAspectRatio="xMaxYMid meet" aria-hidden="true">
      {kind === "active" && (
        <>
          <Strata />
          <Fronts x={70} id="art-active" />
          <Line pulse={3} />
          <path d={`M0 ${GROUND} H400`} className="art-pulse" />
          <Shot x={70} />
        </>
      )}
      {kind === "passive" && (
        <>
          <Strata />
          <Line />
          {[0, 1, 2].map((k) => (
            <path
              key={k}
              d={wave(-70, 70, 6, GROUND - 22 - k * 9)}
              className={`art-noise ${k % 2 ? "back" : ""}`}
              style={{ animationDelay: `${k * 1.3}s`, animationDuration: `${4.4 + k * 0.9}s` }}
            />
          ))}
        </>
      )}
      {kind === "passive-active" && (
        <>
          <Strata />
          <Fronts x={60} id="art-pa" />
          <Line />
          {[0, 1].map((k) => (
            <path
              key={k}
              d={wave(-70, 70, 6, GROUND - 26 - k * 8)}
              className="art-noise"
              style={{ animationDelay: `${k * 1.6}s`, animationDuration: "4.2s" }}
            />
          ))}
          <Shot x={60} />
        </>
      )}
      {kind === "picking" && (
        <>
          <path d="M120 18 C190 30 250 58 380 70 L380 92 C250 80 190 52 120 40 Z" className="art-ridge wide" />
          <path d="M120 24 C190 36 250 62 380 75 L380 86 C250 74 190 47 120 34 Z" className="art-ridge" />
          <path id="art-pick-path" d="M120 29 C190 41 250 67 380 80" className="art-pick" />
          <circle r={4} className="art-cursor">
            <animateMotion dur="4s" repeatCount="indefinite" rotate="auto">
              <mpath href="#art-pick-path" />
            </animateMotion>
          </circle>
          {[150, 205, 260, 315, 365].map((x, i) => (
            <circle key={x} cx={x} cy={29 + (x - 120) * 0.2 + (i === 2 ? 6 : 0)} r={2.2} className="art-dot" style={{ animationDelay: `${i * 0.8}s` }} />
          ))}
        </>
      )}
      {kind === "inversion" && (
        <>
          {[0, 1, 2, 3, 4].map((k) => {
            const jitter = [0, 8, -6, 12, -10][k];
            return (
              <path
                key={k}
                d={`M${210 + jitter} 12 V${38 + jitter / 2} H${250 - jitter} V${66 - jitter / 3} H${292 + jitter} V108`}
                className="art-sample"
                style={{ animationDelay: `${k * 0.7}s` }}
              />
            );
          })}
          <path d="M212 12 V40 H250 V66 H292 V108" className="art-profile" />
          <path d="M150 12 V108" className="art-axis" />
          {[12, 40, 66, 108].map((y) => (
            <path key={y} d={`M146 ${y} H154`} className="art-axis" />
          ))}
          <path d="M330 30 C345 20 360 44 375 34" className="art-chain" />
          <path d="M330 60 C345 50 360 74 375 64" className="art-chain late" />
        </>
      )}
      {kind === "petro" && (
        <>
          <defs>
            <pattern id="art-sand" width="8" height="8" patternUnits="userSpaceOnUse">
              <circle cx="2" cy="2" r="1" className="art-grain" />
              <circle cx="6" cy="6" r="1" className="art-grain" />
            </pattern>
            <pattern id="art-clay" width="12" height="6" patternUnits="userSpaceOnUse">
              <path d="M0 3 H7" className="art-hatch" />
            </pattern>
            <pattern id="art-gravel" width="14" height="12" patternUnits="userSpaceOnUse">
              <circle cx="4" cy="4" r="2.4" className="art-grain" />
              <circle cx="11" cy="9" r="1.8" className="art-grain" />
            </pattern>
          </defs>
          <rect x={130} y={20} width={250} height={26} fill="url(#art-sand)" className="art-soil one" />
          <rect x={130} y={46} width={250} height={30} fill="url(#art-clay)" className="art-soil two" />
          <rect x={130} y={76} width={250} height={34} fill="url(#art-gravel)" className="art-soil three" />
          <rect x={130} y={20} width={18} height={90} className="art-scan" />
        </>
      )}
      {kind === "visualization" && (
        <>
          {Array.from({ length: 12 }, (_, i) => (
            <rect
              key={i}
              x={130 + i * 21}
              y={30 + ((i * 7) % 13)}
              width={17}
              height={70 - ((i * 7) % 13)}
              rx={3}
              className="art-column"
              style={{ animationDelay: `${i * 0.25}s` }}
            />
          ))}
          <path d="M126 104 H384" className="art-axis" />
          <path d="M130 58 C180 40 230 76 280 52 S360 44 384 60" className="art-trend" />
        </>
      )}
      {kind === "assistant" && (
        <>
          {[
            [300, 50, 16],
            [352, 30, 9],
            [250, 84, 8],
            [360, 88, 11],
          ].map(([x, y, r], i) => (
            <path
              key={i}
              d={`M${x} ${y - r} Q${x} ${y} ${x + r} ${y} Q${x} ${y} ${x} ${y + r} Q${x} ${y} ${x - r} ${y} Q${x} ${y} ${x} ${y - r} Z`}
              className="art-sparkle"
              style={{ animationDelay: `${i * 0.6}s`, transformOrigin: `${x}px ${y}px` }}
            />
          ))}
          <path d={wave(150, 110, 7, 70)} className="art-trace" />
        </>
      )}
    </svg>
  );
}
