// Props of the `Scribble` composition: hand-drawn and neon strokes drawn on the things in one
// rendered video, rendered as a transparent layer. Times are timeline seconds (same base as
// EditPlan `sfx.at_s`); x/y are fractions of the frame (0–1, from the top left); sizes and strokes
// are fractions of the frame width, so one file fits the 720×1280 preview and the 1080×1920 final.
// The engine (`opencine.scribble`) fills `at_s` from anchors, shape `keys` and `track` from the video.
import {z} from 'zod';

const unit = z.number().min(0).max(1);
const at = (x: number, y: number) => ({x: unit.default(x), y: unit.default(y)});

/** Where a moving item is, `t` seconds after its start (blended between keys, held outside);
 *  `s` scales a doodle riding on a thing that comes closer or moves away. */
export const trackKey = z.strictObject({
	t: z.number().min(0), x: unit, y: unit, s: z.number().positive().max(4).optional(),
});

/** The thing's contour `t` seconds after the item's start: the engine lines the points up from
 *  key to key, so they blend point by point. */
export const shapeKey = z.strictObject({t: z.number().min(0), pts: z.array(z.tuple([unit, unit])).min(3)});

/** How the line is painted: `pen` a solid line over a dark shadow, `marker` loose overlapping
 *  passes, `neon` a glowing tube (a light core in a blurred halo), `fire` a flickering flame. */
export const LOOKS = ['pen', 'marker', 'neon', 'fire'] as const;

/** Pen settings every item shares. */
const pen = (dur: number, color: string, stroke: number, draw: number, look: (typeof LOOKS)[number]) => ({
	at_s: z.number().min(0),
	dur_s: z.number().positive().default(dur),
	look: z.enum(LOOKS).default(look),
	color: z.string().default(color),
	stroke: z.number().positive().max(0.08).default(stroke),
	draw_s: z.number().positive().default(draw),
	rough: z.number().min(0).max(3).default(1.2),
	boil: z.number().min(0).max(24).default(8),
	exit: z.enum(['fade', 'erase', 'cut']).default('fade'),
	shadow: z.boolean().default(true),
	seed: z.number().int().min(0).default(1),
});
const shape = {keys: z.array(shapeKey).default([])};
// `true` / [x, y] (a point on what to follow) ask the engine to track; it writes the keys
const track = {track: z.union([z.array(trackKey), z.tuple([unit, unit]), z.literal(true)]).optional()};

/** A line hugging the thing's silhouette `offset` out from it: drawn on round it (`draw`), or a
 *  bright stroke `length` of the way round running `laps` times a second over a faint line (`run`). */
export const outlineFx = z.strictObject({
	type: z.literal('outline'), ...pen(1.6, '#FFFFFF', 0.012, 0.6, 'pen'), ...at(0.5, 0.5), ...shape,
	offset: z.number().min(0).max(0.1).default(0.012),
	style: z.enum(['draw', 'run']).default('draw'),
	length: z.number().positive().max(1).default(0.3),
	laps: z.number().positive().max(4).default(0.5),
});

/** The thing painted in, stroke after stroke: `scribble` (a back-and-forth marker fill), `hatch`,
 *  `cross` (cross-hatch) or `solid` (a see-through colour wiped on); `edge` also draws its outline. */
export const fillFx = z.strictObject({
	type: z.literal('fill'), ...pen(1.6, '#FF2BD6', 0.007, 0.8, 'marker'), ...at(0.5, 0.5), ...shape,
	pattern: z.enum(['scribble', 'hatch', 'cross', 'solid']).default('scribble'),
	gap: z.number().min(0.004).max(0.1).default(0.022),
	angle: z.number().default(-41),
	alpha: z.number().min(0.05).max(1).default(0.85),
	edge: z.boolean().default(true),
});

/** Rings wrapped round the thing at `level` (0 = its top, 1 = its bottom), `spread`× as wide as it
 *  is there, `tilt` = height / width; the half behind it hides behind it. `turns` coils, an open
 *  `arc` of each turn shown, spinning `spin` turns a second. */
export const ringFx = z.strictObject({
	type: z.literal('ring'), ...pen(1.6, '#00E5FF', 0.009, 0.5, 'neon'), ...at(0.5, 0.5), ...shape,
	level: unit.default(0.5),
	spread: z.number().min(0.6).max(3).default(1.2),
	tilt: z.number().min(0.05).max(1).default(0.28),
	turns: z.number().min(1).max(4).default(1),
	arc: z.number().min(0.2).max(1).default(0.85),
	spin: z.number().min(-4).max(4).default(0.6),
});

/** A streak behind a moving thing: where it was over the last `tail_s`, tapering to nothing;
 *  it shows while the thing moves. */
export const trailFx = z.strictObject({
	type: z.literal('trail'), ...pen(1.6, '#FF2BD6', 0.02, 0.2, 'neon'), ...at(0.5, 0.5), ...track,
	tail_s: z.number().min(0.05).max(2).default(0.4),
});

export const DOODLES = ['sparkle', 'star', 'heart', 'burst', 'swirl', 'squiggle', 'lightning', 'crown',
	'diamond', 'music', 'exclaim', 'question', 'sun', 'smiley', 'moon', 'skull', 'flame', 'eye', 'drop',
	'planet', 'flower', 'spade', 'rainbow', 'dollar', 'asterisk', 'triangle', 'bubble', 'dashes', 'smoke',
	'halo', 'mustache', 'cross', 'glasses'] as const;

/** What a scatter deals out unless told: everything but the parts made for a face or a thing. */
export const ART = DOODLES.filter((k) => !['mustache', 'cross', 'glasses', 'halo', 'eye', 'dashes', 'smoke'].includes(k));

/** How a doodle moves once drawn: `pop` springs in, `bob` floats, `spin` turns, `pulse` beats
 *  twice a second, `wiggle` rocks, `still` stays (all but `still` spring in). */
export const MOTIONS = ['pop', 'bob', 'spin', 'pulse', 'wiggle', 'still'] as const;

/** `fill` paints a doodle's closed parts (a heart, a skull's eyes) in hatching or scribble. */
const art = {fill: z.enum(['none', 'hatch', 'scribble']).default('none')};

/** A small doodle drawn on: on a thing it rides with it (`spot`: off its corner, on its top, on it;
 *  the engine turns `u`, `v` on the thing's box into x, y). */
export const doodleFx = z.strictObject({
	type: z.literal('doodle'), ...pen(1.4, '#FFFFFF', 0.009, 0.35, 'pen'), ...at(0.5, 0.5), ...track, ...art,
	motion: z.enum(MOTIONS).default('pop'),
	kind: z.enum(DOODLES).default('sparkle'),
	spot: z.enum(['side', 'top', 'on']).default('side'),
	size: z.number().positive().max(1).default(0.14),
	rotate: z.number().default(0),
});

/** One of a scatter's doodles (the engine places them). */
export const markKey = z.strictObject({
	kind: z.enum(DOODLES), x: unit, y: unit, size: z.number().positive().max(1), rotate: z.number(),
	color: z.string(), delay: z.number().min(0),
});

/** Many doodles strewn round the thing (clear of it) or over `area`, drawn on one after another
 *  `stagger_s` apart, kinds and `colors` dealt round. */
export const scatterFx = z.strictObject({
	type: z.literal('scatter'), ...pen(2, '#FFFFFF', 0.007, 0.3, 'marker'), ...at(0.5, 0.5), ...art,
	motion: z.enum(MOTIONS).default('bob'),
	kinds: z.array(z.enum(DOODLES)).min(1).default([...ART]),
	colors: z.array(z.string()).default([]),
	count: z.number().int().min(1).max(40).default(10),
	size: z.number().positive().max(0.5).default(0.09),
	area: z.tuple([unit, unit, unit, unit]).default([0.05, 0.06, 0.95, 0.78]),
	stagger_s: z.number().min(0).max(1).default(0.06),
	marks: z.array(markKey).default([]),
});

/** A border round the whole frame: drawn on twice (`draw`), or a stroke running round it (`run`). */
export const frameFx = z.strictObject({
	type: z.literal('frame'), ...pen(2, '#FFFFFF', 0.01, 1, 'pen'),
	style: z.enum(['draw', 'run']).default('draw'),
	inset: z.number().min(0).max(0.2).default(0.035),
	length: z.number().positive().max(1).default(0.3),
	laps: z.number().positive().max(4).default(0.6),
});

export const fxItem = z.discriminatedUnion('type', [outlineFx, fillFx, ringFx, trailFx, doodleFx, scatterFx, frameFx]);

export const fxSchema = z.object({
	meta: z.object({
		width: z.number().int().positive(), height: z.number().int().positive(),
		fps: z.number().positive(), frames: z.number().int().positive(),
	}),
	items: z.array(fxItem),
});

export type FxProps = z.infer<typeof fxSchema>;
export type FxItem = z.infer<typeof fxItem>;
export type Of<T extends FxItem['type']> = Extract<FxItem, {type: T}>;
export type TrackKey = z.infer<typeof trackKey>;
export type ShapeKey = z.infer<typeof shapeKey>;
export type Look = (typeof LOOKS)[number];
export type DoodleKind = (typeof DOODLES)[number];
export type Motion = (typeof MOTIONS)[number];
