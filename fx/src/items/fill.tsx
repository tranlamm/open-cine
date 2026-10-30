import React from 'react';
import {useCurrentFrame} from 'remotion';
import {Ink, offFrame, penOf, penOptions, shapeAt, sketch, smoothPath, timing, type Ctx} from '../draw';
import type {Of} from '../schema';

const FILL = {scribble: 'zigzag', hatch: 'hachure', cross: 'cross-hatch'} as const;
const EDGE_SHARE = 0.3; // of the drawing time: the edge goes round first, then the paint goes in

/** The thing painted in as it moves (its contour keys, blended): hatch lines or a back-and-forth
 *  marker fill drawn on stroke after stroke, or a see-through colour wiped on from the top. */
export const Fill: React.FC<{it: Of<'fill'>} & Ctx> = ({it, ...c}) => {
	const frame = useCurrentFrame();
	if (!it.keys.length) return null; // the engine fills the keys
	const {W, H} = c;
	const pts = shapeAt(it.keys, frame, c.fps, W, H);
	const pen = penOf(it, c, frame);
	const {draw, opacity} = timing(it, frame, c);
	const paintP = it.edge ? Math.max(0, (draw - EDGE_SHARE / 2) / (1 - EDGE_SHARE / 2)) : draw;
	const o = penOptions(it, c, frame, {disableMultiStroke: true});
	let paint: React.ReactNode;
	if (it.pattern === 'solid') {
		const ys = pts.map(([, y]) => y);
		const [top, bottom] = [Math.min(...ys), Math.max(...ys)];
		const glow = it.look === 'neon' || it.look === 'fire';
		const region = {filterUnits: 'userSpaceOnUse' as const, x: -0.25 * W, y: -0.25 * H, width: 1.5 * W, height: 1.5 * H};
		const d = smoothPath(pts, true);
		paint = (
			<svg width={W} height={H} style={{position: 'absolute', inset: 0, opacity: opacity * it.alpha, overflow: 'visible'}}>
				<defs>
					<clipPath id={`${pen.id}-wipe`}><rect x={-W} y={-H} width={3 * W} height={H + top + (bottom - top) * paintP} /></clipPath>
					{glow ? <filter id={`${pen.id}-glow`} {...region}><feGaussianBlur stdDeviation={pen.width * 2.5} /></filter> : null}
				</defs>
				<g clipPath={`url(#${pen.id}-wipe)`}>
					{glow ? <path d={d} fill={it.color} filter={`url(#${pen.id}-glow)`} /> : null}
					<path d={d} fill={it.color} />
				</g>
			</svg>
		);
	} else {
		const fillStyle = FILL[it.pattern];
		const strokes = sketch((g, o) => g.polygon(pts, {
			...o, stroke: 'none', fill: it.color, fillStyle, hachureGap: it.gap * W,
			hachureAngle: it.angle, fillWeight: pen.width,
		}), o);
		paint = <Ink strokes={strokes} progress={paintP} pen={{...pen, shadow: false}} opacity={opacity * it.alpha} />;
	}
	if (!it.edge) return <>{paint}</>;
	const edge = sketch((g, o) => g.curve([...pts, pts[0], pts[1]], o), {...o, roughness: (o.roughness ?? 1) * 0.6});
	const edgePen = {...pen, id: `${pen.id}e`, width: pen.width * 1.4};
	return (
		<>
			{paint}
			<Ink strokes={edge} progress={Math.min(1, draw / EDGE_SHARE)} pen={edgePen} opacity={opacity} inset={offFrame(edgePen)} />
		</>
	);
};
