import React from 'react';
import {interpolate, useCurrentFrame} from 'remotion';
import {along, clamp, Paint, penOf, timing, type Ctx, type Pt} from '../draw';
import type {Of} from '../schema';

const SAMPLES_PER_S = 60;
const STILL = [0.02, 0.08]; // path length (of the width) below which the trail is gone, above which it is full

const xy = ([x, y]: Pt) => `${x.toFixed(1)} ${y.toFixed(1)}`;

/** A filled stroke along `pts`, tapering from nothing at the tail to `half` at the round head. */
export const taper = (pts: Pt[], half: number): string => {
	const n = pts.length;
	const left: Pt[] = [];
	const right: Pt[] = [];
	pts.forEach(([x, y], k) => {
		const [a, b] = [pts[Math.max(0, k - 1)], pts[Math.min(n - 1, k + 1)]];
		const len = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
		const [nx, ny] = [-(b[1] - a[1]) / len, (b[0] - a[0]) / len];
		const w = half * (k / (n - 1)) ** 0.8;
		left.push([x + nx * w, y + ny * w]);
		right.push([x - nx * w, y - ny * w]);
	});
	const h = half.toFixed(1);
	return `M ${left.map(xy).join(' L ')} A ${h} ${h} 0 0 0 ${xy(right[n - 1])} L ${right.reverse().map(xy).join(' L ')} Z`;
};

/** A streak behind a moving thing (its tracked path, smoothed): where it was over the last
 *  `tail_s`, tapering off; it fades away while the thing holds still. */
export const Trail: React.FC<{it: Of<'trail'>} & Ctx> = ({it, ...c}) => {
	const frame = useCurrentFrame();
	const {W, H, fps} = c;
	const t = frame / fps;
	const n = Math.max(2, Math.round(it.tail_s * SAMPLES_PER_S));
	const pts: Pt[] = [];
	for (let k = 0; k <= n; k++) {
		const p = along(it.track, Math.max(0, t - it.tail_s * (1 - k / n)), W, H);
		if (!p) return null; // the engine fills the track
		const last = pts[pts.length - 1];
		if (!last || Math.hypot(p[0] - last[0], p[1] - last[1]) > 0.5) pts.push(p);
	}
	const length = pts.slice(1).reduce((s, p, k) => s + Math.hypot(p[0] - pts[k][0], p[1] - pts[k][1]), 0);
	const moving = interpolate(length, [STILL[0] * W, STILL[1] * W], [0, 1], clamp);
	if (pts.length < 3 || moving <= 0) return null;
	const pen = penOf(it, c, frame);
	const {draw, opacity} = timing(it, frame, c);
	const half = (pen.width / 2) * (0.5 + 0.5 * draw);
	return <Paint pen={pen} opacity={opacity * moving} draw={(color, scale) => <path d={taper(pts, half * scale)} fill={color} />} />;
};
