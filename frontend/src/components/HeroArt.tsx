// The home page's artwork: a shot (a star) on layered ground, its wavefronts spreading through
// the layers, lighting the receivers (inverted triangles) as it reaches them, and what PAC
// makes of it: a dispersion curve, and a Vs profile stepping up with depth.

const SOURCE_X = 96;
// The shot, once a CYCLE s: its front spreads through the ground at SPEED px/s, a receiver turning
// blue as the front reaches it; the dispersion curve draws as the front crosses the receivers,
// each point as the line reaches it, the Vs profile just behind; all fade before the next shot.
const CYCLE = 4.4;
const SPEED = 180;
const REACH = 640 - SOURCE_X; // the front's radius, and the wave's run, to the drawing's edge
const SURFACE = (x: number) => 150 + 5 * Math.sin(x / 70);
const FADED = 0.985; // the share of the cycle what a shot gave has faded at
const PROFILE_X = [556, 574, 592, 614]; // the Vs profile at each layer, faster with depth
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

/** Whether the viewer asked for less motion: the SMIL animations then left out (home.css stills
 * the CSS ones). */
function still(): boolean {
  return typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
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
  const moving = !still();
  const grow = (REACH / SPEED / CYCLE).toFixed(4); // the front's share of the cycle
  // The share of the cycle the front reaches the first receiver, and the last, at.
  const first = (RECEIVERS[0] - SOURCE_X) / SPEED / CYCLE;
  const last = (RECEIVERS[RECEIVERS.length - 1] - SOURCE_X) / SPEED / CYCLE;
  const lag = 0.08; // the profile behind the curve
  // The Vs profile, a step down each layer, and when (a share of the cycle) its line reaches each
  // layer, the layer's Vs shown then.
  const top = SURFACE(600);
  const profile = `M${PROFILE_X[0]},${top} ${LAYERS.slice(1)
    .map((layer, i) => `V${top + layer.top} H${PROFILE_X[i + 1]}`)
    .join(" ")} V400`;
  const lengths = LAYERS.map((layer, i) => layer.top + (PROFILE_X[i] - PROFILE_X[0]));
  const total = 400 - top + (PROFILE_X[PROFILE_X.length - 1] - PROFILE_X[0]);
  const reached = lengths.map((length) => first + lag + (length / total) * (last - first));
  const hold = 0.93; // everything shown until then, faded by FADED
  /** A line drawn from `from` to `to` (shares of the cycle), held, faded, hidden again. */
  const drawn = (from: number, to: number) =>
    moving && (
      <>
        <animate
          attributeName="stroke-dashoffset"
          values="1;1;0;0;1"
          keyTimes={`0;${from.toFixed(4)};${to.toFixed(4)};${FADED};1`}
          dur={`${CYCLE}s`}
          repeatCount="indefinite"
        />
        <animate attributeName="opacity" values="1;1;0;0" keyTimes={`0;${hold};${FADED};1`} dur={`${CYCLE}s`} repeatCount="indefinite" />
      </>
    );
  // A dispersion curve: slow at high frequency, fast at low.
  const curve = Array.from({ length: 24 }, (_, i) => {
    const f = i / 23;
    const x = 418 + f * 176;
    const y = 30 + 70 * (1 - Math.exp(-3.2 * f));
    return [x, y] as const;
  });
  return (
    <svg className="hero-art" viewBox="0 0 640 400" role="img" aria-label="A shot's waves reaching a line of receivers over layered ground">
      <defs>
        <clipPath id="hero-ground">
          <path d={ground} />
        </clipPath>
        <radialGradient id="hero-glow" cx="0.5" cy="0.5" r="0.5">
          <stop offset="0" stopColor="#f5c14b" stopOpacity="0.9" />
          <stop offset="1" stopColor="#f5c14b" stopOpacity="0" />
        </radialGradient>
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
            <text
              x={548 + 19 * i}
              y={SURFACE(548) + layer.top + 34}
              textAnchor="end"
              className="hero-art-label"
              opacity={moving ? 0 : undefined}
            >
              {layer.vs}
              {moving && (
                <animate
                  attributeName="opacity"
                  values="0;0;1;1;0;0"
                  keyTimes={`0;${reached[i].toFixed(4)};${(reached[i] + 0.03).toFixed(4)};${hold};${FADED};1`}
                  dur={`${CYCLE}s`}
                  repeatCount="indefinite"
                />
              )}
            </text>
          </g>
        );
      })}

      {/* The shot's front, spreading through the ground. */}
      <g clipPath="url(#hero-ground)">
        <circle className="hero-front" cx={SOURCE_X} cy={sourceY} r={moving ? 0 : REACH / 2} opacity={moving ? 0 : 0.35}>
          {moving && (
            <>
              <animate attributeName="r" values={`0;${REACH};${REACH}`} keyTimes={`0;${grow};1`} dur={`${CYCLE}s`} repeatCount="indefinite" />
              <animate
                attributeName="opacity"
                values="0.95;0.5;0;0"
                keyTimes={`0;${(0.8 * Number(grow)).toFixed(4)};${grow};1`}
                dur={`${CYCLE}s`}
                repeatCount="indefinite"
              />
            </>
          )}
        </circle>
      </g>

      {/* The Vs profile: faster with depth. */}
      <path className="hero-profile" pathLength={1} d={profile}>
        {drawn(first + lag, last + lag)}
      </path>

      {/* The surface. */}
      <path d={surface} fill="none" stroke="rgba(255,255,255,0.55)" strokeWidth={1.4} />

      {/* Receivers along the line, and the shot. */}
      {RECEIVERS.map((x) => {
        const y = SURFACE(x);
        return (
          <polygon
            key={x}
            className="hero-receiver"
            style={{ animationDelay: `${((x - SOURCE_X) / SPEED).toFixed(2)}s`, animationDuration: `${CYCLE}s` }}
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
          pathLength={1}
        >
          {drawn(first, last)}
        </path>
        {curve.map(([x, y], i) => {
          if (i % 4 !== 0) return null;
          // Shown as the line reaches it.
          const at = first + (i / (curve.length - 1)) * (last - first);
          return (
            <circle key={x} cx={x} cy={y} r={2.6} fill="#ffffff" className="hero-dot">
              {moving && (
                <animate
                  attributeName="opacity"
                  values="0;0;0.9;0.9;0;0"
                  keyTimes={`0;${at.toFixed(4)};${(at + 0.02).toFixed(4)};${hold};${FADED};1`}
                  dur={`${CYCLE}s`}
                  repeatCount="indefinite"
                />
              )}
            </circle>
          );
        })}
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
