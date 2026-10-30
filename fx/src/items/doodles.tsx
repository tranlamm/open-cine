import React from 'react';
import type {Drawable, Options} from 'roughjs/bin/core';
import {Easing, interpolate, spring, useCurrentFrame} from 'remotion';
import {clamp, drawn, Ink, lineStyle, Paint, penOf, penOptions, place, sketch, Strokes, timing, type Ctx, type Pt, type Stroke} from '../draw';
import type {DoodleKind, Motion, Of} from '../schema';

// Doodles in a unit box (−0.5…0.5, y down), drawn stroke by stroke in the listed order;
// `closed` parts take the doodle's `fill`.
type Part = {curve?: Pt[]; line?: Pt[]; dot?: Pt; closed?: boolean};

const TAU = Math.PI * 2;
const ring = (n: number, f: (k: number) => Pt): Pt[] => Array.from({length: n + 1}, (_, k) => f(k / n));
const circle = (cx: number, cy: number, r: number, n = 20): Pt[] =>
	ring(n, (f) => [cx + Math.cos(f * TAU) * r, cy + Math.sin(f * TAU) * r]);
const ellipse = (cx: number, cy: number, rx: number, ry: number, turn = 0, n = 24): Pt[] => ring(n, (f) => {
	const [x, y] = [Math.cos(f * TAU) * rx, Math.sin(f * TAU) * ry];
	return [cx + x * Math.cos(turn) - y * Math.sin(turn), cy + x * Math.sin(turn) + y * Math.cos(turn)];
});
/** An arc from `a0` to `a1` degrees (y down: 90 is the bottom). */
const arc = (cx: number, cy: number, r: number, a0: number, a1: number, n = 16): Pt[] => ring(n, (f) => {
	const a = ((a0 + (a1 - a0) * f) * Math.PI) / 180;
	return [cx + Math.cos(a) * r, cy + Math.sin(a) * r];
});
const flip = (pts: Pt[]): Pt[] => pts.map(([x, y]) => [-x, y]);

const sparkle = (cx: number, cy: number, s: number): Pt[] => [
	[0, -0.5], [0.08, -0.08], [0.5, 0], [0.08, 0.08], [0, 0.5], [-0.08, 0.08], [-0.5, 0], [-0.08, -0.08], [0, -0.5],
].map(([x, y]) => [cx + x * s, cy + y * s]);

const heart = (dy = 0, up = false): Pt[] => ring(28, (f) => {
	const t = f * TAU;
	const y = -(13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t)) / 34 - 0.04;
	return [(16 * Math.sin(t) ** 3) / 34, (up ? -y : y) + dy];
});

// a handlebar moustache, right half: its top swept out into a curl, its underside back in
const lip: Pt[] = [[0, -0.04], [0.14, -0.15], [0.3, -0.1], [0.4, 0.01], [0.48, -0.06], [0.47, -0.19], [0.39, -0.21], [0.37, -0.13]];
const lipUnder: Pt[] = [[0.4, 0.01], [0.34, 0.07], [0.2, 0.07], [0.07, 0.03], [0, 0.03]];

const SHAPES: Record<DoodleKind, Part[]> = {
	sparkle: [{curve: sparkle(-0.08, 0.06, 0.82)}, {curve: sparkle(0.36, -0.34, 0.3)}],
	star: [{closed: true, line: ring(10, (f) => {
		const a = -Math.PI / 2 + f * TAU;
		const r = Math.round(f * 10) % 2 ? 0.2 : 0.5;
		return [Math.cos(a) * r, Math.sin(a) * r];
	})}],
	heart: [{closed: true, curve: heart()}],
	burst: Array.from({length: 8}, (_, k) => {
		const a = (k / 8) * TAU + 0.2;
		return {line: [[Math.cos(a) * 0.3, Math.sin(a) * 0.3], [Math.cos(a) * 0.5, Math.sin(a) * 0.5]] as Pt[]};
	}),
	swirl: [{curve: ring(30, (f) => {
		const a = f * Math.PI * 5;
		const r = 0.04 + 0.44 * f;
		return [Math.cos(a) * r, Math.sin(a) * r];
	})}],
	squiggle: [{curve: ring(12, (f) => [f - 0.5, Math.sin(f * Math.PI * 4) * 0.12])}],
	lightning: [{closed: true, line: [[0.12, -0.5], [-0.24, 0.06], [0.02, 0.06], [-0.12, 0.5], [0.26, -0.08], [0, -0.08], [0.12, -0.5]]}],
	crown: [
		{closed: true, line: [[-0.5, 0.3], [-0.46, -0.3], [-0.22, 0.02], [0, -0.44], [0.22, 0.02], [0.46, -0.3], [0.5, 0.3], [-0.5, 0.3]]},
		{line: [[-0.47, 0.16], [0.47, 0.16]]},
	],
	diamond: [
		{closed: true, line: [[-0.28, -0.36], [0.28, -0.36], [0.5, -0.1], [0, 0.48], [-0.5, -0.1], [-0.28, -0.36]]},
		{line: [[-0.5, -0.1], [0.5, -0.1]]},
		{line: [[-0.1, -0.36], [-0.18, -0.1], [0, 0.48], [0.18, -0.1], [0.1, -0.36]]},
	],
	music: [
		{closed: true, curve: ellipse(-0.2, 0.32, 0.17, 0.12, -0.3, 16)},
		{line: [[-0.04, 0.3], [-0.04, -0.46]]},
		{curve: [[-0.04, -0.46], [0.2, -0.32], [0.32, -0.12], [0.2, 0.04]]},
	],
	exclaim: [{line: [[0, -0.5], [0.02, 0.18]]}, {dot: [0.01, 0.4]}],
	question: [{curve: [[-0.22, -0.2], [-0.16, -0.44], [0.1, -0.5], [0.26, -0.32], [0.16, -0.1], [0, 0.02], [0, 0.16]]}, {dot: [0, 0.4]}],
	sun: [
		{closed: true, curve: circle(0, 0, 0.22)},
		...Array.from({length: 8}, (_, k): Part => {
			const a = (k / 8) * TAU + 0.3;
			return {line: [[Math.cos(a) * 0.32, Math.sin(a) * 0.32], [Math.cos(a) * 0.48, Math.sin(a) * 0.48]]};
		}),
	],
	smiley: [{curve: circle(0, 0, 0.45, 24)}, {dot: [-0.15, -0.12]}, {dot: [0.15, -0.12]}, {curve: arc(0, 0.02, 0.26, 25, 155, 10)}],
	moon: [{closed: true, curve: [...arc(0, 0, 0.45, 60, 300, 18), ...arc(0.22, 0, 0.39, 270, 90, 14)]}],
	skull: [
		{curve: [...arc(0, -0.05, 0.4, 160, 380, 18), [0.3, 0.2], [0.22, 0.42], [-0.22, 0.42], [-0.3, 0.2], [-0.376, 0.087]]},
		{closed: true, curve: circle(-0.15, 0.02, 0.1, 12)},
		{closed: true, curve: circle(0.15, 0.02, 0.1, 12)},
		{line: [[0, 0.13], [-0.04, 0.22], [0.04, 0.22], [0, 0.13]]},
		{line: [[-0.1, 0.3], [-0.1, 0.42]]}, {line: [[0, 0.3], [0, 0.42]]}, {line: [[0.1, 0.3], [0.1, 0.42]]},
	],
	flame: [
		{closed: true, curve: [[0, -0.5], [0.1, -0.28], [0.28, -0.08], [0.32, 0.18], [0.2, 0.4], [0, 0.48], [-0.2, 0.4],
			[-0.32, 0.18], [-0.24, -0.04], [-0.12, 0.06], [-0.12, -0.2], [0, -0.5]]},
		{curve: [[0, 0.02], [0.12, 0.2], [0.08, 0.38], [-0.08, 0.38], [-0.12, 0.22], [0, 0.02]]},
	],
	eye: [
		{curve: [[-0.5, 0], [-0.25, -0.22], [0.25, -0.22], [0.5, 0], [0.25, 0.2], [-0.25, 0.2], [-0.5, 0]]},
		{closed: true, curve: circle(0, 0, 0.17, 14)},
		{dot: [0, 0]},
		{line: [[-0.3, -0.17], [-0.38, -0.34]]}, {line: [[-0.1, -0.22], [-0.12, -0.4]]},
		{line: [[0.1, -0.22], [0.12, -0.4]]}, {line: [[0.3, -0.17], [0.38, -0.34]]},
	],
	drop: [{closed: true, curve: [[0, -0.5], [0.16, -0.2], [0.3, 0.1], [0.28, 0.3], [0.12, 0.46], [-0.12, 0.46], [-0.28, 0.3],
		[-0.3, 0.1], [-0.16, -0.2], [0, -0.5]]}],
	planet: [{closed: true, curve: circle(0, 0, 0.27)}, {curve: ellipse(0, 0, 0.5, 0.13, -0.35)}],
	flower: [
		...Array.from({length: 5}, (_, k): Part => {
			const a = -Math.PI / 2 + (k / 5) * TAU;
			return {curve: ellipse(Math.cos(a) * 0.25, Math.sin(a) * 0.25, 0.21, 0.12, a, 16)};
		}),
		{closed: true, curve: circle(0, 0, 0.08, 10)},
	],
	spade: [
		{closed: true, curve: heart(-0.1, true).map(([x, y]): Pt => [x * 0.95, y * 0.9])},
		{closed: true, line: [[0, 0.12], [-0.13, 0.48], [0.13, 0.48], [0, 0.12]]},
	],
	rainbow: [0.48, 0.36, 0.24].map((r): Part => ({curve: arc(0, 0.25, r, 180, 360, 16)})),
	dollar: [
		{curve: [[0.24, -0.26], [0.06, -0.36], [-0.18, -0.3], [-0.22, -0.1], [0, 0], [0.22, 0.1], [0.2, 0.3], [-0.04, 0.38], [-0.26, 0.28]]},
		{line: [[0.02, -0.5], [-0.02, 0.5]]},
	],
	asterisk: [90, 30, 150].map((d): Part => {
		const a = (d * Math.PI) / 180;
		return {line: [[Math.cos(a) * 0.45, Math.sin(a) * 0.45], [-Math.cos(a) * 0.45, -Math.sin(a) * 0.45]]};
	}),
	triangle: [{closed: true, line: [[0, -0.45], [0.42, 0.35], [-0.42, 0.35], [0, -0.45]]}],
	bubble: [{curve: circle(0, 0, 0.45, 24)}, {curve: arc(0, 0, 0.31, 15, 70, 8)}, {dot: [0.06, 0.31]}],
	dashes: [{line: [[-0.32, -0.05], [-0.46, -0.28]]}, {line: [[0, -0.15], [0, -0.45]]}, {line: [[0.32, -0.05], [0.46, -0.28]]}],
	smoke: [{curve: [[-0.1, 0.5], [0.12, 0.3], [-0.08, 0.12], [0.14, -0.08], [-0.06, -0.28], [0.1, -0.5]]}],
	halo: [{curve: ellipse(0, 0, 0.5, 0.14)}],
	mustache: [{curve: flip(lip)}, {curve: lip}, {curve: flip(lipUnder)}, {curve: lipUnder}],
	cross: [{line: [[-0.4, -0.4], [0.4, 0.4]]}, {line: [[0.4, -0.4], [-0.4, 0.4]]}],
	glasses: [
		{closed: true, curve: circle(-0.26, 0, 0.2, 16)},
		{closed: true, curve: circle(0.26, 0, 0.2, 16)},
		{curve: [[-0.07, -0.03], [0, -0.08], [0.07, -0.03]]},
		{line: [[-0.46, -0.02], [-0.5, -0.12]]}, {line: [[0.46, -0.02], [0.5, -0.12]]},
	],
};

type Fill = Of<'doodle'>['fill'];

/** The strokes of a doodle centred on (cx, cy), `size` px across, turned `rotate` degrees; its
 *  closed parts hatched or scribbled in after their outline when `fill` asks. */
export const doodleStrokes = (kind: DoodleKind, fill: Fill, [cx, cy]: Pt, size: number, rotate: number,
	o: Options, width: number): Stroke[] => {
	const [cos, sin] = [Math.cos((rotate * Math.PI) / 180), Math.sin((rotate * Math.PI) / 180)];
	const px = ([u, v]: Pt): Pt => [cx + (u * cos - v * sin) * size, cy + (u * sin + v * cos) * size];
	return sketch((g, o) => SHAPES[kind].flatMap((part): Drawable[] => {
		if (part.dot) {
			const [x, y] = px(part.dot);
			return [g.circle(x, y, Math.max(2, 0.07 * size), {...o, roughness: Math.min(o.roughness ?? 1, 0.8)})];
		}
		const pts = (part.curve ?? part.line ?? []).map(px);
		const out = [part.curve ? g.curve(pts, o) : g.linearPath(pts, o)];
		if (part.closed && fill !== 'none') {
			out.push(g.polygon(pts, {...o, stroke: 'none', fill: 'ink', fillStyle: fill === 'hatch' ? 'hachure' : 'zigzag',
				hachureGap: Math.max(2, 0.07 * size), fillWeight: width * 0.6}));
		}
		return out;
	}), o);
};

/** A doodle's scale, extra turn (degrees) and lift (of the width) `f` frames after it starts:
 *  all but `still` spring in, then `bob` floats, `spin` turns, `pulse` beats twice a second and
 *  `wiggle` rocks; `k` puts many doodles out of step. */
export const moving = (motion: Motion, f: number, fps: number, draw: number, k = 0): {s: number; rot: number; lift: number} => {
	if (motion === 'still') return {s: 0.85 + 0.15 * draw, rot: 0, lift: 0};
	const t = f / fps;
	const ph = k * 1.7;
	const s = 0.35 + 0.65 * spring({frame: f, fps, config: {damping: 9, stiffness: 170, mass: 0.6}});
	if (motion === 'bob') return {s, rot: 4 * Math.sin(TAU * 0.6 * t + ph), lift: 0.008 * Math.sin(TAU * 0.9 * t + ph)};
	if (motion === 'spin') return {s, rot: (k % 2 ? -1 : 1) * 110 * t, lift: 0};
	if (motion === 'pulse') return {s: s * (1 + 0.14 * Math.max(0, Math.sin(TAU * 2 * t + ph)) ** 2), rot: 0, lift: 0};
	if (motion === 'wiggle') return {s, rot: 10 * Math.sin(TAU * 3 * t + ph), lift: 0};
	return {s, rot: 0, lift: 0};
};

/** A small doodle drawn on as it springs in and moves; on a thing it rides with it, growing and
 *  shrinking as the thing does. */
export const Doodle: React.FC<{it: Of<'doodle'>} & Ctx> = ({it, ...c}) => {
	const frame = useCurrentFrame();
	const {W, H} = c;
	const p = place(it, frame, c.fps);
	const {draw, opacity} = timing(it, frame, c);
	const m = moving(it.motion, frame, c.fps, draw);
	const pen = penOf(it, c, frame);
	const o = penOptions(it, c, frame, {disableMultiStroke: true});
	const strokes = doodleStrokes(it.kind, it.fill, [p.x * W, p.y * H - m.lift * W], it.size * W * p.s * m.s,
		it.rotate + m.rot, o, pen.width);
	return <Ink strokes={strokes} progress={draw} pen={pen} opacity={opacity} />;
};

/** Many doodles strewn round the thing, each drawn on and springing in `delay` after the item
 *  starts, then moving; one paint for all (one glow for a neon scatter). */
export const Scatter: React.FC<{it: Of<'scatter'>} & Ctx> = ({it, ...c}) => {
	const frame = useCurrentFrame();
	const {W, H, fps} = c;
	const {opacity} = timing(it, frame, c);
	const pen = penOf(it, c, frame);
	const o = penOptions(it, c, frame, {disableMultiStroke: true});
	const marks = it.marks.flatMap((m, k) => {
		const f = frame - Math.round(m.delay * fps);
		if (f < 0) return [];
		const draw = interpolate(f, [0, Math.max(1, it.draw_s * fps)], [0, 1], {...clamp, easing: Easing.inOut(Easing.quad)});
		const mv = moving(it.motion, f, fps, draw, k);
		const strokes = doodleStrokes(m.kind, it.fill, [m.x * W, m.y * H - mv.lift * W], m.size * W * mv.s, m.rotate + mv.rot,
			{...o, seed: (o.seed ?? 1) + k * 101}, pen.width);
		return [{color: m.color, shown: drawn(strokes, draw)}];
	});
	if (!marks.length) return null;
	return (
		<Paint pen={pen} opacity={opacity} draw={(_, scale, paint) => marks.map(({color, shown}, k) => (
			<g key={k} {...lineStyle(paint(color), pen.width * scale)}><Strokes shown={shown} /></g>
		))} />
	);
};
