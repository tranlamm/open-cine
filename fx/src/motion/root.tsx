import React from 'react';
import {AbsoluteFill, Composition} from 'remotion';
import {EMPTY, MotionProvider, PassProvider, type MotionData} from './kit';

/** A layer file: its component, and the timeline spans (s) where it draws anything. */
export type Layer = {default: React.FC; spans?: Array<[number, number]>};
type Meta = {width: number; height: number; fps: number; frames: number};
type Props = {meta: Meta; data: MotionData; spans: Array<[number, number]> | null; pass: string | null};

/** The root render.mjs registers for a layer: one transparent "Motion" composition sized like
 * the video; `spans` is returned in its props so only those frames are captured. `pass`: null =
 * front pass, a matte name = only that matte's <Behind>. */
export const motionRoot = (layer: Layer): React.FC => {
	// a behind pass hides everything; its <Behind> shows itself again (visibility is inherited)
	const Draw: React.FC<Props> = ({data, pass}) => (
		<AbsoluteFill style={{visibility: pass ? 'hidden' : undefined}}>
			<MotionProvider value={{...EMPTY, ...data}}>
				<PassProvider value={pass ?? null}>
					<layer.default />
				</PassProvider>
			</MotionProvider>
		</AbsoluteFill>
	);
	return () => (
		<Composition
			id="Motion"
			component={Draw}
			width={1080}
			height={1920}
			fps={30}
			durationInFrames={90}
			defaultProps={{meta: {width: 1080, height: 1920, fps: 30, frames: 90}, data: EMPTY, spans: null, pass: null} as Props}
			calculateMetadata={({props}) => ({
				props: {...props, spans: layer.spans ?? null},
				width: props.meta.width, height: props.meta.height, fps: props.meta.fps, durationInFrames: props.meta.frames,
			})}
		/>
	);
};
