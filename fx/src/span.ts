import type {FxItem} from './schema';

/** First frame and length of an item, inside a timeline of `frames` frames. */
export const span = (it: Pick<FxItem, 'at_s' | 'dur_s'>, fps: number, frames: number): [number, number] => {
	const from = Math.min(frames - 1, Math.max(0, Math.round(it.at_s * fps)));
	return [from, Math.max(1, Math.min(frames - from, Math.round(it.dur_s * fps)))];
};
