// Each page's small artwork, in its header band: the home page's hero, in miniature, for what
// the page does. Shots are stars, receivers inverted triangles, as in every plot. The passive
// pages tell how a line becomes its own source: the noise (or a shot) recorded, then correlated
// at the first receiver, which turns into a virtual shot (a dashed star) and fires.

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
const FIRST = RECEIVERS[0];
const RIGHT = 400; // the drawing's right edge, past which the fronts leave the array
// Every front in the soil moves at this speed, px/s: a receiver lights as one reaches it.
const SPEED = 110;
// The processing pages' cycles, s: the active shot's, and the passive pages' story.
const ACTIVE = 4;
const STORY = 9;
// When the virtual shot fires: on the passive page after the noise, on the passive-active page
// after its shot (kit.css's art-shot-fade glides that shot into the first receiver before).
const VIRTUAL = { passive: 6, "passive-active": 4.8 };
const SHOT_AT = 0.3; // the passive-active shot fires
// The noise: fronts from sources out of the drawing, on either side and below, reaching the
// array `begin` s into the story.
const NOISE = [
  { x: -60, y: 80, begin: 0 },
  { x: 500, y: 110, begin: 0.5 },
  { x: 210, y: 330, begin: 1.1 },
  { x: -40, y: 150, begin: 1.7 },
  { x: 340, y: 320, begin: 2.3 },
  { x: 520, y: 70, begin: 2.9 },
];
const SOIL = { left: FIRST - 40, right: RIGHT + 10, top: GROUND, bottom: GROUND + 60 };

function star(cx: number, cy: number, outer: number): string {
  const points: string[] = [];
  for (let k = 0; k < 10; k++) {
    const r = k % 2 === 0 ? outer : outer * 0.42;
    const a = -Math.PI / 2 + (k * Math.PI) / 5;
    points.push(`${(cx + r * Math.cos(a)).toFixed(1)},${(cy + r * Math.sin(a)).toFixed(1)}`);
  }
  return points.join(" ");
}

/** Whether the viewer asked for less motion: SMIL's animations then left out (kit.css stills
 * the CSS ones). */
function still(): boolean {
  return typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/** SMIL timing of a move lasting `travel` s once a `cycle` s from `begin` s, held after: where
 * in the cycle it ends (`end`, for keyTimes), and the animation's own attributes (`clock`). */
function timing(travel: number, cycle: number, begin: number) {
  return {
    end: Math.min(travel / cycle, 0.999).toFixed(4),
    clock: { dur: `${cycle}s`, begin: `${begin.toFixed(2)}s`, repeatCount: "indefinite" as const },
  };
}

/** A receiver at `x`: an inverted triangle on the ground. */
function receiver(x: number): string {
  return `${x - 4},${GROUND - 9} ${x + 4},${GROUND - 9} ${x},${GROUND - 2}`;
}

/** The ground and the receivers, each flashing once a `cycle` s at the times `flashes` give it
 * (one flash a front, as it reaches the receiver). */
function Line({ flashes = [], cycle = 0 }: { flashes?: ((x: number) => number)[]; cycle?: number }) {
  return (
    <>
      <path d={`M-600 ${GROUND} H${RIGHT}`} className="art-ground" />
      {RECEIVERS.map((x) => (
        <g key={x}>
          <polygon className="art-receiver" points={receiver(x)} />
          {flashes.map((at, k) => (
            <polygon
              key={k}
              className="art-flash"
              points={receiver(x)}
              style={{ animationDelay: `${at(x).toFixed(2)}s`, animationDuration: `${cycle}s` }}
            />
          ))}
        </g>
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

/** The soil, for fronts to travel in and nowhere else. */
function Soil({ id }: { id: string }) {
  return (
    <defs>
      <clipPath id={id}>
        <rect x={-600} y={GROUND} width={1000} height={60} />
      </clipPath>
    </defs>
  );
}

/** A shot's front in the soil: a circle from `x` growing at SPEED, once a `cycle` s from `begin`
 * s, until it has crossed the array (fading as it leaves). */
function Front({ x, id, begin = 0, cycle }: { x: number; id: string; begin?: number; cycle: number }) {
  const radius = RIGHT + 10 - x;
  const time = timing(radius / SPEED, cycle, begin);
  return (
    <>
      <Soil id={id} />
      <circle className="art-wave" cx={x} cy={GROUND} r={0} opacity={0} clipPath={`url(#${id})`}>
        {!still() && (
          <>
            <animate attributeName="r" values={`0;${radius};${radius}`} keyTimes={`0;${time.end};1`} {...time.clock} />
            <animate
              attributeName="opacity"
              values="0.9;0.55;0;0"
              keyTimes={`0;${(0.8 * Number(time.end)).toFixed(4)};${time.end};1`}
              {...time.clock}
            />
          </>
        )}
      </circle>
    </>
  );
}

/** The radii a noise front from `source` enters the soil with (its nearest point) and leaves it
 * with, the array crossed (its farthest). */
function noiseRadii(source: (typeof NOISE)[number]): [number, number] {
  const nearestX = Math.min(Math.max(source.x, SOIL.left), SOIL.right);
  const nearestY = Math.min(Math.max(source.y, SOIL.top), SOIL.bottom);
  const farthest = Math.max(
    ...[SOIL.left, SOIL.right].flatMap((x) => [SOIL.top, SOIL.bottom].map((y) => Math.hypot(x - source.x, y - source.y))),
  );
  return [Math.hypot(nearestX - source.x, nearestY - source.y), farthest];
}

/** When a noise front reaches the receiver at `x`. */
function noiseAt(source: (typeof NOISE)[number], x: number): number {
  const [from] = noiseRadii(source);
  return source.begin + (Math.hypot(x - source.x, GROUND - source.y) - from) / SPEED;
}

/** The ambient noise: fronts from sources out of the drawing, on either side and below, crossing
 * the soil and the array at SPEED. */
function Noise({ id }: { id: string }) {
  return (
    <>
      <Soil id={id} />
      <g clipPath={`url(#${id})`}>
        {NOISE.map((source, k) => {
          const [from, to] = noiseRadii(source);
          const time = timing((to - from) / SPEED, STORY, source.begin);
          return (
            <circle
              key={k}
              className={k % 2 ? "art-wave back" : "art-wave"}
              cx={source.x}
              cy={source.y}
              r={from}
              opacity={0}
            >
              {!still() && (
                <>
                  <animate attributeName="r" values={`${from};${to};${to}`} keyTimes={`0;${time.end};1`} {...time.clock} />
                  <animate
                    attributeName="opacity"
                    values="0;0.85;0.85;0;0"
                    keyTimes={`0;${(0.08 * Number(time.end)).toFixed(4)};${(0.88 * Number(time.end)).toFixed(4)};${time.end};1`}
                    {...time.clock}
                  />
                </>
              )}
            </circle>
          );
        })}
      </g>
    </>
  );
}

/** The first receiver as a virtual shot: itself turned gold, a source's colour, a halo round
 * it, shown about when it fires (`at`, s). */
function VirtualShot({ at }: { at: number }) {
  return (
    <g className="art-virtual" style={{ animationDelay: `${(at - 0.45).toFixed(2)}s` }}>
      <circle cx={FIRST} cy={GROUND - 5.5} r={11} className="art-glow" />
      <polygon points={receiver(FIRST)} className="art-receiver virtual" />
    </g>
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
          <Front x={70} id="art-active" cycle={ACTIVE} />
          <Line flashes={[(x) => (x - 70) / SPEED]} cycle={ACTIVE} />
          <Shot x={70} />
        </>
      )}
      {kind === "passive" && (
        <>
          <Strata />
          <Noise id="art-noise" />
          <Front x={FIRST} id="art-virtual-p" begin={VIRTUAL.passive} cycle={STORY} />
          <Line
            flashes={[
              ...NOISE.map((source) => (x: number) => noiseAt(source, x)),
              (x) => VIRTUAL.passive + (x - FIRST) / SPEED,
            ]}
            cycle={STORY}
          />
          <VirtualShot at={VIRTUAL.passive} />
        </>
      )}
      {kind === "passive-active" && (
        <>
          <Strata />
          {/* The shot, recorded; gone, it returns as the first receiver, which fires. */}
          <Front x={60} id="art-shot-pa" begin={SHOT_AT} cycle={STORY} />
          <Front x={FIRST} id="art-virtual-pa" begin={VIRTUAL["passive-active"]} cycle={STORY} />
          <Line
            flashes={[
              (x) => SHOT_AT + (x - 60) / SPEED,
              (x) => VIRTUAL["passive-active"] + (x - FIRST) / SPEED,
            ]}
            cycle={STORY}
          />
          <g className="art-shot-fade">
            <Shot x={60} />
          </g>
          <VirtualShot at={VIRTUAL["passive-active"]} />
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
          <path d="M198 12 V40 H236 V66 H276 V108 H308 V66 H264 V40 H226 V12 Z" className="art-spread" />
          <path d="M212 12 V40 H250 V66 H292 V108" className="art-profile" />
          <path d="M150 12 V108" className="art-axis" />
          {[12, 40, 66, 108].map((y) => (
            <path key={y} d={`M146 ${y} H154`} className="art-axis" />
          ))}
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
              y={16}
              width={17}
              height={70 - ((i * 7) % 13)}
              rx={3}
              className="art-column"
              style={{ animationDelay: `${i * 0.25}s` }}
            />
          ))}
          <path d="M126 16 H384" className="art-axis" />
          <path d="M130 62 C180 80 230 44 280 68 S360 76 384 60" className="art-trend" />
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
