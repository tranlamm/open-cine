import React from 'react';
import {getLength} from '@remotion/paths';
import rough from 'roughjs';
import type {Drawable, Options} from 'roughjs/bin/core';
import type {RoughGenerator} from 'roughjs/bin/generator';
import {Easing, interpolate, random} from 'remotion';
import type {FxItem, Look, ShapeKey, TrackKey} from './schema';

// The pen: shapes are sketched by rough.js with a seed (so a frame always draws the same lines),
// split into strokes and drawn on in order, each stroke at its own share of the drawing time,
// then painted in the item's look (pen, marker, neon, fire).

export const clamp = {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'} as const;

/** Props every item gets: its item, its index, its length in frames and the frame size. */
export type Ctx = {i: number; len: number; fps: number; W: number; H: number};

export type Pt = [number, number];
export type Stroke = {d: string; len: number};

const gen = rough.generator();

/** One sub-path per pen stroke (rough.js draws most lines twice: two strokes; a hatch fill is
 *  one stroke per line). */
const strokesOf = (dr: Drawable): Stroke[] => {
	const out: Stroke[] = [];
	for (const set of dr.sets) {
		if (set.type === 'fillPath') continue; // solid fills are painted by the item itself
		let d = '';
		for (const {op, data} of set.ops) {
			const n = data.map((v) => Math.round(v * 10) / 10).join(' ');
			if (op === 'move') {
				if (d) out.push({d, len: 0});
				d = `M ${n}`;
			} else {
				d += op === 'lineTo' ? ` L ${n}` : ` C ${n}`;
			}
		}
		if (d) out.push({d, len: 0});
	}
	return out.filter((s) => s.d.includes(' L ') || s.d.includes(' C ')).map((s) => ({...s, len: getLength(s.d)}));
};

/** Sketch shapes with rough.js; `seed` must be ≥ 1 to draw the same lines every render. */
export const sketch = (draw: (g: RoughGenerator, o: Options) => Drawable | Drawable[], o: Options): Stroke[] =>
	[draw(gen, {...o, seed: Math.max(1, Math.round(o.seed ?? 1))})].flat().flatMap(strokesOf);

/** Which redraw of the line this frame shows: it "boils" `boil` times a second, like
 *  hand-drawn animation drawn on threes. */
export const boilStep = (it: FxItem, c: Ctx, frame: number) => (it.boil > 0 ? Math.floor((frame * it.boil) / c.fps) : 0);

/** rough.js options of an item: its roughness and stroke width, a seed that changes with each
 *  boil; `marker` adds loose second passes, `neon`/`fire` straighten the tube. */
export const penOptions = (it: FxItem, c: Ctx, frame: number, extra: Options = {}): Options => {
	const o: Options = {
		roughness: it.rough,
		bowing: it.rough * 0.8,
		strokeWidth: it.stroke * c.W,
		seed: it.seed * 1009 + c.i * 37 + boilStep(it, c, frame) + 1,
		...extra,
	};
	const r = o.roughness ?? it.rough;
	if (it.look === 'marker') return {...o, roughness: r * 1.3 + 0.5, bowing: (o.bowing ?? 1) + 1, disableMultiStroke: false};
	if (it.look === 'neon' || it.look === 'fire') return {...o, roughness: r * 0.35};
	return o;
};

/** Draw progress 0 → 1 (eased like a hand) and opacity, with the item's exit. */
export const timing = (it: FxItem, frame: number, {len, fps}: Ctx, drawS = it.draw_s) => {
	const draw = interpolate(frame, [0, Math.max(1, drawS * fps)], [0, 1], {...clamp, easing: Easing.inOut(Easing.quad)});
	const n = it.exit === 'cut' ? 0 : Math.max(2, Math.round((it.exit === 'erase' ? Math.min(0.35, drawS) : 0.2) * fps));
	const out = n ? interpolate(frame, [len - n, len - 1], [1, 0], clamp) : 1;
	return {draw: it.exit === 'erase' ? Math.min(draw, out) : draw, opacity: it.exit === 'fade' ? out : 1};
};

const keysOf = (track: unknown): TrackKey[] =>
	(Array.isArray(track) && typeof track[0] === 'object' ? (track as TrackKey[]) : []);

/** Where a tracked item is now: its `track` keys (linear, held outside), else its own x/y;
 *  `s` = its scale on a thing that comes closer or moves away. */
export const place = (it: {x: number; y: number; track?: unknown}, frame: number, fps: number): {x: number; y: number; s: number} => {
	const keys = keysOf(it.track);
	if (!keys.length) return {x: it.x, y: it.y, s: 1};
	const t = frame / fps;
	const at = (k: TrackKey) => ({x: k.x, y: k.y, s: k.s ?? 1});
	if (t <= keys[0].t) return at(keys[0]);
	for (let j = 1; j < keys.length; j++) {
		if (t < keys[j].t) {
			const [a, b] = [at(keys[j - 1]), at(keys[j])];
			const k = (t - keys[j - 1].t) / Math.max(1e-6, keys[j].t - keys[j - 1].t);
			return {x: a.x + (b.x - a.x) * k, y: a.y + (b.y - a.y) * k, s: a.s + (b.s - a.s) * k};
		}
	}
	return at(keys[keys.length - 1]);
};

/** Where a tracked thing is at `t` s, in pixels: a Catmull-Rom curve through its keys (held
 *  outside), so a trail bends the way the thing swung. */
export const along = (track: unknown, t: number, W: number, H: number): Pt | null => {
	const keys = keysOf(track);
	if (!keys.length) return null;
	const last = keys.length - 1;
	if (t <= keys[0].t) return [keys[0].x * W, keys[0].y * H];
	if (t >= keys[last].t) return [keys[last].x * W, keys[last].y * H];
	let j = 0;
	while (j < last - 1 && keys[j + 1].t <= t) j++;
	const [p0, p1, p2, p3] = [keys[Math.max(0, j - 1)], keys[j], keys[j + 1], keys[Math.min(last, j + 2)]];
	const u = (t - p1.t) / Math.max(1e-6, p2.t - p1.t);
	const cr = (a: number, b: number, c: number, d: number) =>
		0.5 * (2 * b + (c - a) * u + (2 * a - 5 * b + 4 * c - d) * u * u + (3 * b - a - 3 * c + d) * u * u * u);
	return [cr(p0.x, p1.x, p2.x, p3.x) * W, cr(p0.y, p1.y, p2.y, p3.y) * H];
};

/** The thing's contour now, in pixels: its keys blended point by point (held outside). */
export const shapeAt = (keys: ShapeKey[], frame: number, fps: number, W: number, H: number): Pt[] => {
	const t = frame / fps;
	const j = keys.findIndex((k) => k.t > t);
	const [a, b] = j === -1 ? [keys[keys.length - 1], keys[keys.length - 1]] : j === 0 ? [keys[0], keys[0]] : [keys[j - 1], keys[j]];
	const k = a === b ? 0 : (t - a.t) / Math.max(1e-6, b.t - a.t);
	if (a.pts.length !== b.pts.length) return (k < 0.5 ? a : b).pts.map(([x, y]) => [x * W, y * H]);
	return a.pts.map(([x, y], i) => [(x + (b.pts[i][0] - x) * k) * W, (y + (b.pts[i][1] - y) * k) * H]);
};

const num = ([x, y]: Pt) => `${x.toFixed(1)} ${y.toFixed(1)}`;

/** A smooth SVG path through `pts` (Catmull-Rom as cubic Béziers). */
export const smoothPath = (pts: Pt[], closed: boolean): string => {
	const n = pts.length;
	if (n < 2) return '';
	const at = (i: number) => (closed ? pts[(i + n) % n] : pts[Math.max(0, Math.min(n - 1, i))]);
	let d = `M ${num(pts[0])}`;
	for (let i = 0; i < (closed ? n : n - 1); i++) {
		const [p0, p1, p2, p3] = [at(i - 1), at(i), at(i + 1), at(i + 2)];
		const c1: Pt = [p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6];
		const c2: Pt = [p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6];
		d += ` C ${num(c1)} ${num(c2)} ${num(p2)}`;
	}
	return closed ? `${d} Z` : d;
};

/** Small seeded wobble (fraction of `amp`) so hand-built shapes are never perfectly regular. */
export const wobble = (seed: string, amp: number) => (random(seed) * 2 - 1) * amp;

/** `color` mixed `k` of the way to white (a neon tube's hot core). */
export const tint = (color: string, k: number): string => {
	const m = /^#?([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(color.trim());
	if (!m) return '#FFFFFF';
	const hex = m[1].length === 3 ? [...m[1]].map((ch) => ch + ch).join('') : m[1];
	const rgb = [0, 2, 4].map((i) => parseInt(hex.slice(i, i + 2), 16));
	return `rgb(${rgb.map((v) => Math.round(v + (255 - v) * k)).join(',')})`;
};

/** What an item paints with this frame: its look, colour, line width (px) and boil step. */
export type Pen = {look: Look; color: string; width: number; shadow: boolean; W: number; H: number; id: string; step: number};

export const penOf = (it: FxItem, c: Ctx, frame: number, width = it.stroke * c.W): Pen => ({
	look: it.look, color: it.color, width, shadow: it.shadow, W: c.W, H: c.H, id: `fx${c.i}`, step: boilStep(it, c, frame),
});

/** The layers a look paints, bottom first: `paint` turns a stroke's own colour into this
 *  layer's, `scale` × the line width, `blur` in px. One blurred halo per look: blur is most of a
 *  frame's render time (soft alpha is costly to encode). */
type Layer = {paint: (c: string) => string; scale: number; opacity: number; blur: number};
const same = (c: string) => c;
const fixed = (c: string) => () => c;
const layersOf = ({look, width: w, shadow}: Pen): Layer[] => {
	if (look === 'neon') {
		return [{paint: same, scale: 2.6, opacity: 0.7, blur: w * 1.4}, {paint: same, scale: 1.3, opacity: 1, blur: 0},
			{paint: (c) => tint(c, 0.75), scale: 0.5, opacity: 1, blur: 0}];
	}
	if (look === 'fire') {
		return [{paint: fixed('#FF3D00'), scale: 2.4, opacity: 0.75, blur: w * 1.2}, {paint: fixed('#FF9A00'), scale: 1.2, opacity: 1, blur: 0},
			{paint: fixed('#FFF2B3'), scale: 0.45, opacity: 1, blur: 0}];
	}
	const ink = {paint: same, scale: 1, opacity: 1, blur: 0};
	return shadow ? [{paint: fixed('rgba(0,0,0,0.45)'), scale: 1.8, opacity: 1, blur: 0}, ink] : [ink];
};

/** Paints `draw(color, scale, paint)` once per layer of the pen's look: blurred halos under a
 *  sharp core for neon and fire (fire also licked by turbulence that changes every boil).
 *  `color` is the pen's colour in this layer, `paint(c)` any other colour in it. */
export const Paint: React.FC<{
	pen: Pen; opacity?: number; defs?: React.ReactNode; inset?: number;
	draw: (color: string, scale: number, paint: (c: string) => string) => React.ReactNode;
}> = ({pen, opacity = 1, defs, inset, draw}) => {
	const {W, H, id, width: w} = pen;
	const region = {filterUnits: 'userSpaceOnUse' as const, x: -0.25 * W, y: -0.25 * H, width: 1.5 * W, height: 1.5 * H};
	const layers = layersOf(pen);
	return (
		<svg width={W} height={H} style={{position: 'absolute', inset: 0, opacity, overflow: 'visible',
			clipPath: inset ? `inset(${inset.toFixed(1)}px)` : undefined}}>
			<defs>
				{defs}
				{layers.map((l, k) => (l.blur > 0
					? <filter key={k} id={`${id}-blur${k}`} {...region}><feGaussianBlur stdDeviation={l.blur} /></filter>
					: null))}
				{pen.look === 'fire' ? (
					<filter id={`${id}-fire`} {...region}>
						<feTurbulence type="fractalNoise" baseFrequency={`${1 / (6 * w)} ${1 / (2.5 * w)}`} numOctaves={2} seed={pen.step + 1} result="noise" />
						<feDisplacementMap in="SourceGraphic" in2="noise" scale={w * 2.4} xChannelSelector="R" yChannelSelector="G" />
					</filter>
				) : null}
			</defs>
			<g filter={pen.look === 'fire' ? `url(#${id}-fire)` : undefined}>
				{layers.map((l, k) => (
					<g key={k} opacity={l.opacity} filter={l.blur > 0 ? `url(#${id}-blur${k})` : undefined}>{draw(l.paint(pen.color), l.scale, l.paint)}</g>
				))}
			</g>
		</svg>
	);
};

export const lineStyle = (color: string, width: number): React.SVGAttributes<SVGElement> => ({
	fill: 'none', stroke: color, strokeWidth: width, strokeLinecap: 'round', strokeLinejoin: 'round',
});

/** The strokes drawn so far: the first `progress` share of their total length. */
export const drawn = (strokes: Stroke[], progress: number) => {
	const total = strokes.reduce((s, k) => s + k.len, 0) || 1;
	let acc = 0;
	return strokes.flatMap((s) => {
		const a = acc / total;
		acc += s.len;
		const p = Math.min(1, Math.max(0, (progress - a) / Math.max(1e-6, acc / total - a)));
		return p > 0 ? [{d: s.d, p}] : [];
	});
};

/** SVG paths of drawn strokes; a stroke still being drawn shows its first `p`. */
export const Strokes: React.FC<{shown: {d: string; p: number}[]}> = ({shown}) => (
	<>
		{shown.map(({d, p}, k) => (p >= 1
			? <path key={k} d={d} />
			: <path key={k} d={d} pathLength={1} strokeDasharray="1 1" strokeDashoffset={1 - p} />))}
	</>
);

/** Strokes drawn on in order and painted in the pen's look; `inset` (px) hides what runs along
 *  the frame's edge. */
export const Ink: React.FC<{strokes: Stroke[]; progress: number; pen: Pen; opacity?: number; inset?: number}> = (
	{strokes, progress, pen, opacity, inset},
) => {
	const shown = drawn(strokes, progress);
	return <Paint pen={pen} opacity={opacity} inset={inset} draw={(color, scale) => <g {...lineStyle(color, pen.width * scale)}><Strokes shown={shown} /></g>} />;
};

/** Inset (px) that hides a contour where the thing runs off the frame: there the mask's edge is
 *  the frame's, not the thing's. */
export const offFrame = (pen: Pen) => Math.max(pen.width * 1.5, 0.012 * pen.W);

/** A bright stroke `length` of the way round the closed path `d`, its head at `phase` (0–1),
 *  a longer, fainter tail behind it and a faint line `base` all the way round. */
export const Runner: React.FC<{d: string; phase: number; length: number; pen: Pen; opacity: number; base?: number; inset?: number}> = (
	{d, phase, length, pen, opacity, base = 0, inset},
) => (
	<Paint pen={pen} opacity={opacity} inset={inset} draw={(color, scale) => {
		const w = pen.width * scale;
		const dash = (len: number, lead: number, width: number, alpha: number) => (
			<path d={d} pathLength={1} strokeDasharray={`${len} ${1 - len}`} strokeDashoffset={-(phase - lead)}
				{...lineStyle(color, width)} opacity={alpha} />
		);
		return (
			<>
				{base > 0 ? <path d={d} {...lineStyle(color, w * 0.5)} opacity={base} /> : null}
				{dash(length * 1.8, length * 0.8, w * 0.8, 0.35)}
				{dash(length, 0, w, 1)}
			</>
		);
	}} />
);
