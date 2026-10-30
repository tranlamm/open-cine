import React from 'react';
import {useCurrentFrame} from 'remotion';
import {lineStyle, Paint, penOf, shapeAt, smoothPath, timing, wobble, type Ctx, type Pt} from '../draw';
import type {Of} from '../schema';

const COIL = 1.6; // between coils, in ring heights

/** Where a ring goes round the thing now: across its contour at `level` of its height, the run
 *  inside it under its centre (else the widest run), as a centre and half-width in pixels. */
export const slice = (pts: Pt[], level: number): {x: number; y: number; half: number} | null => {
	const ys = pts.map(([, y]) => y);
	const [top, bottom] = [Math.min(...ys), Math.max(...ys)];
	const y = top + (bottom - top) * level;
	const xs: number[] = [];
	pts.forEach(([x0, y0], k) => {
		const [x1, y1] = pts[(k + 1) % pts.length];
		if ((y0 <= y) !== (y1 <= y)) xs.push(x0 + ((y - y0) / (y1 - y0)) * (x1 - x0));
	});
	xs.sort((a, b) => a - b);
	const cx = pts.reduce((s, [x]) => s + x, 0) / pts.length;
	let best: [number, number] | null = null;
	for (let k = 0; k + 1 < xs.length; k += 2) {
		if (xs[k] <= cx && cx <= xs[k + 1]) {
			best = [xs[k], xs[k + 1]];
			break;
		}
		if (!best || xs[k + 1] - xs[k] > best[1] - best[0]) best = [xs[k], xs[k + 1]];
	}
	return best ? {x: (best[0] + best[1]) / 2, y, half: (best[1] - best[0]) / 2} : null;
};

/** Rings wrapped round the thing (a waist, a wrist, a bottle), drawn on as they spin; the half
 *  going round the back is hidden where the thing is (its contour masks it). */
export const Ring: React.FC<{it: Of<'ring'>} & Ctx> = ({it, ...c}) => {
	const frame = useCurrentFrame();
	if (!it.keys.length) return null; // the engine fills the keys
	const {W, H, fps} = c;
	const body = shapeAt(it.keys, frame, fps, W, H);
	const at = slice(body, it.level);
	if (!at) return null;
	const pen = penOf(it, c, frame);
	const {draw, opacity} = timing(it, frame, c);
	const rx = at.half * it.spread + pen.width;
	const ry = rx * it.tilt;
	const a0 = Math.PI * (0.15 + 2 * it.spin * (frame / fps)); // starts at the front, low on the right
	const sweep = Math.PI * 2 * it.turns * it.arc * draw;
	const n = Math.max(8, Math.ceil(64 * it.turns * it.arc * draw));
	const s = `${it.seed}-${c.i}-${pen.step}`;
	const front: Pt[][] = [];
	const back: Pt[][] = [];
	let run: Pt[] = [];
	let side: boolean | null = null;
	for (let k = 0; k <= n; k++) {
		const u = k / n;
		const a = a0 + sweep * u;
		const r = 1 + wobble(`${s}r${k}`, 0.025 * Math.min(1, it.rough));
		const drop = it.turns > 1 ? (u * draw - 0.5) * (it.turns - 1) * COIL * ry * 2 : 0;
		const p: Pt = [at.x + Math.cos(a) * rx * r, at.y + Math.sin(a) * ry * r + drop];
		const inFront = Math.sin(a) > 0; // the lower half of the ellipse is the near side
		if (side !== null && inFront !== side) {
			(side ? front : back).push(run);
			run = [run[run.length - 1]];
		}
		run.push(p);
		side = inFront;
	}
	if (run.length > 1) (side ? front : back).push(run);
	const behind = `${pen.id}-behind`;
	const defs = (
		<mask id={behind} maskUnits="userSpaceOnUse" x={-W} y={-H} width={3 * W} height={3 * H}>
			<rect x={-W} y={-H} width={3 * W} height={3 * H} fill="white" />
			<path d={smoothPath(body, true)} fill="black" />
		</mask>
	);
	return (
		<Paint pen={pen} opacity={opacity} defs={defs} draw={(color, scale) => (
			<g {...lineStyle(color, pen.width * scale)}>
				<g mask={`url(#${behind})`}>{back.map((r, k) => <path key={k} d={smoothPath(r, false)} />)}</g>
				{front.map((r, k) => <path key={k} d={smoothPath(r, false)} />)}
			</g>
		)} />
	);
};
