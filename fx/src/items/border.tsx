import React from 'react';
import {interpolate, useCurrentFrame} from 'remotion';
import {clamp, Ink, offFrame, penOf, penOptions, Runner, shapeAt, sketch, smoothPath, timing, wobble, type Ctx, type Pt} from '../draw';
import type {Of} from '../schema';

/** Hand-drawn border round the whole frame: drawn on twice (`draw`), or a stroke with a fading
 *  tail running round it `laps` times a second (`run`). */
export const Frame: React.FC<{it: Of<'frame'>} & Ctx> = ({it, ...c}) => {
	const frame = useCurrentFrame();
	const {W, H, fps} = c;
	const m = it.inset * W;
	const pen = penOf(it, c, frame);
	const o = penOptions(it, c, frame, {disableMultiStroke: true});
	const {draw, opacity} = timing(it, frame, c);
	if (it.style === 'draw') {
		const strokes = sketch((g, o) => [g.rectangle(m, m, W - 2 * m, H - 2 * m, o),
			g.rectangle(m, m, W - 2 * m, H - 2 * m, {...o, seed: (o.seed ?? 1) + 500})], o);
		return <Ink strokes={strokes} progress={draw} pen={pen} opacity={opacity} />;
	}
	// one closed, slightly wobbly path round the frame, clockwise from the top left
	const corners: Pt[] = [[m, m], [W - m, m], [W - m, H - m], [m, H - m]];
	const pts: Pt[] = corners.flatMap(([x0, y0], k) => {
		const [x1, y1] = corners[(k + 1) % 4];
		const n = Math.max(2, Math.round(Math.hypot(x1 - x0, y1 - y0) / (0.08 * W)));
		return Array.from({length: n}, (_, j): Pt => {
			const f = j / n;
			const jit = j ? wobble(`${o.seed}-${k}-${j}`, it.rough * 0.0025 * W) : 0;
			return [x0 + (x1 - x0) * f + (y1 === y0 ? 0 : jit), y0 + (y1 - y0) * f + (y1 === y0 ? jit : 0)];
		});
	});
	const d = `M ${pts.map(([x, y]) => `${x.toFixed(1)} ${y.toFixed(1)}`).join(' L ')} Z`;
	const fadeIn = interpolate(frame, [0, Math.max(1, 0.15 * fps)], [0, 1], clamp);
	return <Runner d={d} phase={((frame / fps) * it.laps) % 1} length={it.length} pen={pen} opacity={opacity * fadeIn} />;
};

/** A line hugging the thing's silhouette, moving and bending with it (the engine's contour keys,
 *  blended point by point): drawn on round it from its top (`draw`), or a bright stroke running
 *  round it over a faint line (`run`). Where the thing runs off the frame, no line. */
export const Outline: React.FC<{it: Of<'outline'>} & Ctx> = ({it, ...c}) => {
	const frame = useCurrentFrame();
	if (!it.keys.length) return null; // the engine fills the keys
	const pts = shapeAt(it.keys, frame, c.fps, c.W, c.H);
	const pen = penOf(it, c, frame);
	const {draw, opacity} = timing(it, frame, c);
	if (it.style === 'run') {
		return <Runner d={smoothPath(pts, true)} phase={((frame / c.fps) * it.laps) % 1} length={it.length} pen={pen}
			opacity={opacity * Math.min(1, draw * 2)} base={0.3 * draw} inset={offFrame(pen)} />;
	}
	const o = penOptions(it, c, frame, {disableMultiStroke: true, roughness: it.rough * 0.6});
	const strokes = sketch((g, o) => g.curve([...pts, pts[0], pts[1]], o), o);
	return <Ink strokes={strokes} progress={draw} pen={pen} opacity={opacity} inset={offFrame(pen)} />;
};
