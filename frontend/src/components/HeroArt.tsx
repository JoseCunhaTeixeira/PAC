// The home page's artwork: a shot (a star) on layered ground, its wavefronts spreading through
// the layers, a surface wave running along the receivers (inverted triangles), and what PAC
// makes of it: a dispersion curve, and a Vs profile stepping up with depth.

const SOURCE_X = 96;
const PULSE_S = 3.2; // the surface wave's run along the line (home.css's hero-pulse)
const SURFACE = (x: number) => 150 + 5 * Math.sin(x / 70);
const RECEIVERS = Array.from({ length: 11 }, (_, i) => 176 + i * 36);
const LAYERS = [
  { top: 0, fill: "rgba(142, 178, 232, 0.14)", vs: "Vs 180 m/s" },
  { top: 62, fill: "rgba(118, 138, 240, 0.2)", vs: "Vs 320 m/s" },
  { top: 128, fill: "rgba(95, 104, 226, 0.27)", vs: "Vs 510 m/s" },
  { top: 196, fill: "rgba(74, 80, 196, 0.36)", vs: "Vs 780 m/s" },
];

function boundary(offset: number, wobble: number): string {
  const points: string[] = [];
  for (let x = 0; x <= 640; x += 16) {
    const y = SURFACE(x) + offset + wobble * Math.sin(x / 95 + offset / 40);
    points.push(`${x},${y.toFixed(1)}`);
  }
  return points.join(" L");
}

function star(cx: number, cy: number, outer: number, inner: number): string {
  const points: string[] = [];
  for (let k = 0; k < 10; k++) {
    const r = k % 2 === 0 ? outer : inner;
    const a = -Math.PI / 2 + (k * Math.PI) / 5;
    points.push(`${(cx + r * Math.cos(a)).toFixed(1)},${(cy + r * Math.sin(a)).toFixed(1)}`);
  }
  return points.join(" ");
}

export function HeroArt() {
  const ground = `M${boundary(0, 0)} L640,400 L0,400 Z`;
  const surface = `M${boundary(0, 0)}`;
  const sourceY = SURFACE(SOURCE_X);
  // A dispersion curve: slow at high frequency, fast at low.
  const curve = Array.from({ length: 24 }, (_, i) => {
    const f = i / 23;
    const x = 418 + f * 176;
    const y = 30 + 70 * (1 - Math.exp(-3.2 * f));
    return [x, y] as const;
  });
  return (
    <svg className="hero-art" viewBox="0 0 640 400" role="img" aria-label="A shot's surface waves along a line of receivers over layered ground">
      <defs>
        <clipPath id="hero-ground">
          <path d={ground} />
        </clipPath>
        <radialGradient id="hero-glow" cx="0.5" cy="0.5" r="0.5">
          <stop offset="0" stopColor="#f5c14b" stopOpacity="0.9" />
          <stop offset="1" stopColor="#f5c14b" stopOpacity="0" />
        </radialGradient>
        <linearGradient id="hero-pulse" x1="0" x2="1">
          <stop offset="0" stopColor="#7ee0ee" stopOpacity="0" />
          <stop offset="0.7" stopColor="#7ee0ee" stopOpacity="0.9" />
          <stop offset="1" stopColor="#ffffff" />
        </linearGradient>
        <linearGradient id="hero-curve" x1="0" x2="1">
          <stop offset="0" stopColor="#8f9bff" />
          <stop offset="1" stopColor="#7ee0ee" />
        </linearGradient>
      </defs>

      {/* The ground, layer by layer, and each layer's Vs. */}
      {LAYERS.map((layer, i) => {
        const next = LAYERS[i + 1]?.top;
        const top = `M${boundary(layer.top, i === 0 ? 0 : 7)}`;
        const bottom =
          next === undefined
            ? "L640,400 L0,400 Z"
            : `L${boundary(next, 7).split(" L").reverse().join(" L")} Z`;
        return (
          <g key={layer.vs}>
            <path d={`${top} ${bottom}`} fill={layer.fill} />
            {i > 0 && <path d={top} fill="none" stroke="rgba(255,255,255,0.14)" strokeWidth={1} />}
            <text x={548 + 19 * i} y={SURFACE(548) + layer.top + 34} textAnchor="end" className="hero-art-label">
              {layer.vs}
            </text>
          </g>
        );
      })}

      {/* The shot's wavefronts, spreading through the ground. */}
      <g clipPath="url(#hero-ground)">
        {[0, 1, 2, 3].map((k) => (
          <circle
            key={k}
            className="hero-front"
            cx={SOURCE_X}
            cy={sourceY}
            r={120}
            style={{ animationDelay: `${k * 1.1}s`, transformOrigin: `${SOURCE_X}px ${sourceY}px` }}
          />
        ))}
      </g>

      {/* The Vs profile: faster with depth. */}
      <path
        className="hero-profile"
        d={`M556,${SURFACE(600)} V${SURFACE(600) + 62} H574 V${SURFACE(600) + 128} H592 V${SURFACE(600) + 196} H614 V400`}
      />

      {/* The surface, and a surface wave running along it. */}
      <path d={surface} fill="none" stroke="rgba(255,255,255,0.55)" strokeWidth={1.4} />
      <path className="hero-pulse" d={surface} fill="none" stroke="url(#hero-pulse)" strokeWidth={3} strokeLinecap="round" />

      {/* Receivers along the line, and the shot. */}
      {RECEIVERS.map((x) => {
        const y = SURFACE(x);
        return (
          <polygon
            key={x}
            className="hero-receiver"
            style={{ animationDelay: `${((x / 650) * PULSE_S).toFixed(2)}s` }}
            points={`${x - 6},${y - 13} ${x + 6},${y - 13} ${x},${y - 3}`}
          />
        );
      })}
      <circle cx={SOURCE_X} cy={sourceY - 12} r={22} fill="url(#hero-glow)" opacity={0.55} />
      <polygon points={star(SOURCE_X, sourceY - 12, 11, 4.6)} fill="#f5c14b" stroke="#fff3c9" strokeWidth={0.8} />

      {/* What comes out: a dispersion curve. */}
      <g className="hero-chart">
        <rect x={404} y={14} width={206} height={104} rx={12} fill="rgba(8, 20, 34, 0.55)" stroke="rgba(255,255,255,0.12)" />
        <path d="M418 108 H598 M418 108 V24" stroke="rgba(255,255,255,0.3)" strokeWidth={1} fill="none" />
        <path
          d={`M${curve.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" L")}`}
          fill="none"
          stroke="url(#hero-curve)"
          strokeWidth={2.6}
          strokeLinecap="round"
          className="hero-curve"
        />
        {curve
          .filter((_, i) => i % 4 === 0)
          .map(([x, y]) => (
            <circle key={x} cx={x} cy={y} r={2.6} fill="#ffffff" opacity={0.9} />
          ))}
        <text x={598} y={124} className="hero-art-axis" textAnchor="end">
          frequency
        </text>
        <text x={412} y={20} className="hero-art-axis">
          velocity
        </text>
      </g>
    </svg>
  );
}
