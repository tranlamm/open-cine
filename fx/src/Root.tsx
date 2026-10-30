import React from 'react';
import {Composition} from 'remotion';
import {Scribble} from './Scribble';
import {fxSchema} from './schema';

// For the Studio only: a blob drifting right and a point swinging round, keyed 10× a second as
// the engine keys a thing's contour and path.
const blob = Array.from({length: 31}, (_, k) => ({t: k / 10, pts: Array.from({length: 48}, (_, j): [number, number] => {
	const a = (j / 48) * Math.PI * 2 - Math.PI / 2;
	return [0.32 + 0.01 * k + Math.cos(a) * 0.18, 0.55 + Math.sin(a) * 0.22];
})}));
const swing = Array.from({length: 31}, (_, k) => ({t: k / 10, x: 0.5 + 0.3 * Math.cos(k / 3), y: 0.25 + 0.1 * Math.sin(k / 3)}));

// render.mjs passes `meta` (the video's size, fps, frame count) and `items` (the resolved file).
export const RemotionRoot: React.FC = () => (
	<Composition
		id="Scribble"
		component={Scribble}
		schema={fxSchema}
		width={1080}
		height={1920}
		fps={30}
		durationInFrames={90}
		defaultProps={fxSchema.parse({meta: {width: 1080, height: 1920, fps: 30, frames: 90}, items: [
			{type: 'fill', at_s: 0.2, dur_s: 2.6, keys: blob},
			{type: 'outline', at_s: 0.2, dur_s: 2.6, keys: blob, look: 'neon', style: 'run', color: '#00E5FF'},
			{type: 'ring', at_s: 0.6, dur_s: 2.2, keys: blob, level: 0.6},
			{type: 'trail', at_s: 0, dur_s: 3, track: swing, look: 'fire'},
			{type: 'doodle', at_s: 1, kind: 'crown', spot: 'top', x: 0.45, y: 0.3, size: 0.2, look: 'marker', color: '#FFD400'},
		]})}
		calculateMetadata={({props}) => {
			const p = fxSchema.parse(props); // fills item defaults
			return {props: p, width: p.meta.width, height: p.meta.height, fps: p.meta.fps, durationInFrames: p.meta.frames};
		}}
	/>
);
