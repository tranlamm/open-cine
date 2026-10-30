import React from 'react';
import {AbsoluteFill, Sequence, useVideoConfig} from 'remotion';
import type {Ctx} from './draw';
import {Frame, Outline} from './items/border';
import {Doodle, Scatter} from './items/doodles';
import {Fill} from './items/fill';
import {Ring} from './items/ring';
import {Trail} from './items/trail';
import type {FxItem, FxProps} from './schema';
import {span} from './span';

const ITEMS: {[K in FxItem['type']]: React.FC<{it: Extract<FxItem, {type: K}>} & Ctx>} = {
	outline: Outline, fill: Fill, ring: Ring, trail: Trail, doodle: Doodle, scatter: Scatter, frame: Frame,
};

/** The scribble layer alone, on a transparent background: the engine lays it over the video. */
export const Scribble: React.FC<FxProps> = ({items}) => {
	const {fps, width: W, height: H, durationInFrames} = useVideoConfig();
	return (
		<AbsoluteFill>
			{items.map((it, i) => {
				const [from, len] = span(it, fps, durationInFrames);
				const Comp = ITEMS[it.type] as React.FC<{it: FxItem} & Ctx>;
				return (
					<Sequence key={i} from={from} durationInFrames={len} name={`${it.type} #${i}`}>
						<Comp it={it} i={i} len={len} fps={fps} W={W} H={H} />
					</Sequence>
				);
			})}
		</AbsoluteFill>
	);
};
