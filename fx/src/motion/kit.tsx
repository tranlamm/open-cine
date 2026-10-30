/**
 * opencine-motion: building blocks for the motion layer, the Director's animation code drawn over
 * a rendered video. A layer file imports from 'opencine-motion' and default-exports a component.
 *
 * Units: times are timeline seconds (as in the EditPlan's `*_s` fields and review_sheet), x/y are
 * 0–1 of the frame, sizes are fractions of the frame height.
 */
import React, {createContext, useContext} from 'react';
import {
	AbsoluteFill,
	Easing,
	cancelRender,
	continueRender,
	delayRender,
	interpolate,
	random,
	spring,
	staticFile,
	useCurrentFrame,
	useVideoConfig,
} from 'remotion';

// ---- fonts (assets/fonts, the bundle's public dir) ----------------------
export const FONT = {display: 'Anton', body: 'Montserrat'} as const;
const FACES = [
	{family: FONT.display, file: 'Anton-Regular.ttf', weight: '400'},
	{family: FONT.body, file: 'Montserrat-Bold.ttf', weight: '700'},
	{family: FONT.body, file: 'Montserrat-ExtraBold.ttf', weight: '800'},
];
if (typeof document !== 'undefined') {
	const wait = delayRender('opencine-motion fonts');
	Promise.all(FACES.map(async (f) => {
		const face = new FontFace(f.family, `url(${staticFile(f.file)})`, {weight: f.weight});
		document.fonts.add(await face.load());
	})).then(() => continueRender(wait), (e) => cancelRender(e));
}

// ---- what the engine passes: the video's beats, cuts, words, named moments, safe margins ----
export type Word = {text: string; start: number; end: number};
export type MotionData = {
	beats: number[]; // music beats on the timeline (s)
	strong: number[]; // the strong ones among them
	cuts: number[]; // where each shot starts (s)
	words: Word[]; // voice-over words (s)
	marks: Record<string, number>; // the Director's named moments ("S03@2.1", "say:crispy") resolved (s)
	safe: {top: number; bottom: number; side: number}; // safe margins, fractions of H, H, W
	mattes: string[]; // the subjects <Behind> can put things behind (the tool's `behind`)
};
export const EMPTY: MotionData = {beats: [], strong: [], cuts: [0], words: [], marks: {},
	safe: {top: 0.15, bottom: 0.35, side: 0.11}, mattes: []};
const Data = createContext<MotionData>(EMPTY);
export const MotionProvider = Data.Provider;
// the pass being drawn: null = front (everything but <Behind>), a matte name = only that <Behind>
const Pass = createContext<string | null>(null);
export const PassProvider = Pass.Provider;

/** The data plus now: `t` (s), `fps`, `width`, `height`, `frames`, `mark(name)`. */
export const useMotion = () => {
	const d = useContext(Data);
	const frame = useCurrentFrame();
	const {fps, width, height, durationInFrames} = useVideoConfig();
	const mark = (name: string): number => {
		if (!(name in d.marks)) throw new Error(`mark "${name}" is not in marks (${Object.keys(d.marks).join(', ') || 'none'})`);
		return d.marks[name];
	};
	return {...d, t: frame / fps, fps, width, height, frames: durationInFrames, mark};
};

// ---- time helpers ----------------------------------------------------------------------------
const clamp01 = (v: number) => Math.min(1, Math.max(0, v));
/** 1 on a beat, decaying to 0 over `decay` s: scale things by `1 + k * beatPulse(t, beats)`. */
export const beatPulse = (t: number, beats: number[], decay = 0.18): number => {
	let last = -Infinity;
	for (const b of beats) {
		if (b > t) break;
		last = b;
	}
	const dt = t - last;
	return dt >= 0 && dt < decay ? Math.pow(1 - dt / decay, 2) : 0;
};
/** The first beat at or after `t` (or `t` when none). */
export const nextBeat = (t: number, beats: number[]): number => beats.find((b) => b >= t - 1e-6) ?? t;
/** Beats in [a, b). */
export const beatsIn = (a: number, b: number, beats: number[]): number[] => beats.filter((x) => x >= a && x < b);

/** Children only while at ≤ t < at + dur. */
export const Show: React.FC<{at: number; dur: number; children: React.ReactNode}> = ({at, dur, children}) => {
	const {t} = useMotion();
	return t >= at && t < at + dur ? <>{children}</> : null;
};

/** Children drawn behind the subject of `matte`: rendered in their own pass and cut out where the
 * subject covers them, frame by frame. The matte's window must cover the time they show. */
export const Behind: React.FC<{matte: string; children: React.ReactNode}> = ({matte, children}) => {
	const {mattes} = useContext(Data);
	const pass = useContext(Pass);
	if (!mattes.includes(matte)) throw new Error(`<Behind matte="${matte}">: no such matte (behind: ${mattes.join(', ') || 'none'})`);
	return <AbsoluteFill style={{visibility: pass === matte ? 'visible' : 'hidden'}}>{children}</AbsoluteFill>;
};

// ---- placement -------------------------------------------------------------------------------
export type Band = 'top' | 'center' | 'bottom';
export type Place = {x?: number; y?: number; band?: Band};
/** Absolute style putting a block at x, y (its centre) or on a band of the safe area. */
const placeStyle = (p: Place, safe: MotionData['safe'], W: number, H: number): React.CSSProperties => {
	const x = (p.x ?? 0.5) * W;
	if (p.y !== undefined) return {position: 'absolute', left: x, top: p.y * H, transform: 'translate(-50%, -50%)'};
	const band = p.band ?? 'center';
	if (band === 'top') return {position: 'absolute', left: x, top: (safe.top + 0.01) * H, transform: 'translate(-50%, 0)'};
	if (band === 'bottom') return {position: 'absolute', left: x, top: (1 - safe.bottom - 0.01) * H, transform: 'translate(-50%, -100%)'};
	return {position: 'absolute', left: x, top: 0.5 * H, transform: 'translate(-50%, -50%)'};
};

// ---- kinetic text ----------------------------------------------------------------------------
export type Enter = 'pop' | 'slam' | 'stagger' | 'wave' | 'rise' | 'blur' | 'flip' | 'mask' | 'type'
	| 'scramble' | 'glitch' | 'zoom' | 'none';
export type Exit = 'fade' | 'pop' | 'blur' | 'drop' | 'glitch' | 'none';

export type TextProps = Place & {
	text: string; // '\n' breaks lines
	at: number; // timeline seconds it starts entering
	dur: number; // seconds it shows, exit included
	enter?: Enter;
	exit?: Exit;
	by?: 'char' | 'word'; // the unit that staggers (default: char for ≤ 14 letters, else word)
	stagger?: number; // s between units (default 0.04 char, 0.08 word)
	font?: 'display' | 'body'; // display = Anton (tall, condensed), body = Montserrat ExtraBold
	size?: number; // fraction of H (default display 0.085, body 0.055)
	upper?: boolean; // default: true for display
	color?: string;
	gradient?: [string, string]; // top → bottom fill instead of color
	accent?: string; // colour of the emphasis words
	emphasis?: string[]; // words or phrases (case-insensitive) in the accent colour, a little bigger
	mark?: string; // highlighter colour swept behind the emphasis words after they land
	stroke?: number; // outline width, fraction of the font size (default 0.07; 0 = none)
	strokeColor?: string;
	glow?: string; // glow colour
	shadow?: boolean; // soft drop shadow (default true)
	box?: string; // pill background colour behind the whole text
	rotate?: number; // degrees
	maxWidth?: number; // fraction of W (default the safe width)
	align?: 'left' | 'center' | 'right';
	pulse?: 'beats' | 'strong'; // a bump on each (strong) beat while it shows
	lineHeight?: number;
};

const SEG = typeof Intl !== 'undefined' && 'Segmenter' in Intl ? new Intl.Segmenter('en', {granularity: 'grapheme'}) : null;
const graphemes = (s: string): string[] => (SEG ? [...SEG.segment(s)].map((g) => g.segment) : Array.from(s));
const bare = (s: string) => s.toLocaleLowerCase('en').replace(/[^\p{L}\p{N}%$₫]+/gu, '');
const ENTER_S = 0.32;
const EXIT_S = 0.22;
const GLITCH_S = 0.42;
const SCRAMBLE_S = 0.3;
const SCRAMBLE = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#%&@$';
const BLOCK: Enter[] = ['slam', 'zoom']; // whole-text entrances

/** Which words of `words` an emphasis phrase covers (phrases match consecutive words). */
const emphasised = (words: string[], phrases: string[]): boolean[] => {
	const flags = words.map(() => false);
	const keys = words.map(bare);
	for (const ph of phrases) {
		const want = ph.split(/\s+/).map(bare).filter(Boolean);
		for (let i = 0; want.length && i + want.length <= keys.length; i++) {
			if (want.every((w, k) => keys[i + k] === w)) want.forEach((_, k) => (flags[i + k] = true));
		}
	}
	return flags;
};

type Unit = {chars: string[]; line: number; word: number; emph: boolean; index: number};

export const Text: React.FC<TextProps> = (props) => {
	const {t, fps, width: W, height: H, safe, beats, strong} = useMotion();
	const {text, at, dur, enter = 'pop', exit = 'fade', font = 'display', align = 'center'} = props;
	if (t < at || t >= at + dur) return null;
	const display = font === 'display';
	const upper = props.upper ?? display;
	const px = (props.size ?? (display ? 0.085 : 0.055)) * H;
	const color = props.color ?? '#FFFFFF';
	const accent = props.accent ?? '#FFD400';
	const strokeW = (props.stroke ?? 0.07) * px;
	const strokeColor = props.strokeColor ?? '#000000';
	const body = (upper ? text.toLocaleUpperCase('en') : text).normalize('NFC');
	const lines = body.split('\n').map((l) => l.split(/\s+/).filter(Boolean));
	const flat = lines.flat();
	const emph = emphasised(flat, props.emphasis ?? []);
	const letters = flat.join('').length;
	const by = props.by ?? (letters <= 14 ? 'char' : 'word');
	const stagger = props.stagger ?? (by === 'char' ? 0.04 : 0.08);

	// units: every word, split into letters when they stagger by char
	let w = 0;
	let index = 0;
	const units: Unit[][][] = lines.map((ws, line) => ws.map((word) => {
		const e = emph[w];
		const u: Unit[] = by === 'char'
			? graphemes(word).map((c) => ({chars: [c], line, word: w, emph: e, index: index++}))
			: [{chars: graphemes(word), line, word: w, emph: e, index: index++}];
		w++;
		return u;
	}));
	const block = BLOCK.includes(enter);
	const lt = t - at; // local time
	const out = clamp01((t - (at + dur - EXIT_S)) / EXIT_S); // 0 → 1 over the exit

	// the whole text: block entrances, exit, pulse, rotation
	let scale = 1;
	let tx = 0;
	let ty = 0;
	let opacity = 1;
	let blurPx = 0;
	if (enter === 'slam') {
		scale = interpolate(lt, [0, 0.16], [2.8, 1], {extrapolateRight: 'clamp', easing: Easing.in(Easing.cubic)});
		opacity = clamp01(lt / 0.06);
		blurPx = interpolate(lt, [0, 0.16], [px * 0.12, 0], {extrapolateRight: 'clamp'});
		const shake = lt > 0.16 ? Math.max(0, 1 - (lt - 0.16) / 0.3) : 0;
		tx = (random(`sx${Math.floor(lt * fps)}`) - 0.5) * px * 0.25 * shake;
		ty = (random(`sy${Math.floor(lt * fps)}`) - 0.5) * px * 0.25 * shake;
	} else if (enter === 'zoom') {
		const p = spring({frame: lt * fps, fps, config: {damping: 11, stiffness: 160, mass: 0.7}});
		scale = 0.2 + 0.8 * p;
		opacity = clamp01(lt / 0.1);
	}
	if (exit === 'fade') opacity *= 1 - out;
	if (exit === 'pop') scale *= 1 - Easing.in(Easing.back(2))(out) * 0.999;
	if (exit === 'blur') {
		blurPx += out * px * 0.3;
		opacity *= 1 - out;
		scale *= 1 + 0.15 * out;
	}
	if (exit === 'drop') {
		ty += Easing.in(Easing.quad)(out) * px * 1.2;
		opacity *= 1 - out;
	}
	if (props.pulse) scale *= 1 + 0.07 * beatPulse(t, props.pulse === 'strong' ? strong : beats);

	const glitchOut = exit === 'glitch' ? out : 0;
	// painted on each glyph: a gradient clipped to the text of a parent misses transformed children
	const paint = (emphasised: boolean): React.CSSProperties => (props.gradient
		? {backgroundImage: `linear-gradient(180deg, ${props.gradient[0]}, ${props.gradient[1]})`,
			WebkitBackgroundClip: 'text', backgroundClip: 'text', color: 'transparent'}
		: {color: emphasised ? accent : color});
	const filters = [
		props.shadow === false ? '' : `drop-shadow(0 ${px * 0.05}px ${px * 0.12}px rgba(0,0,0,0.55))`,
		props.glow ? `drop-shadow(0 0 ${px * 0.12}px ${props.glow}) drop-shadow(0 0 ${px * 0.35}px ${props.glow})` : '',
		blurPx > 0.3 ? `blur(${blurPx}px)` : '',
	].filter(Boolean).join(' ');

	const unitStyle = (u: Unit): {style: React.CSSProperties; chars: string[]; ghost: number} => {
		const ut = lt - (block ? 0 : (enter === 'mask' ? u.line : u.index) * stagger);
		const q = clamp01(ut / ENTER_S);
		const e = Easing.out(Easing.cubic)(q);
		const p = spring({frame: Math.max(0, ut) * fps, fps, config: {damping: 13, stiffness: 190, mass: 0.6}});
		let o = ut < 0 && !block ? 0 : 1;
		let transform = '';
		let filter = '';
		let chars = u.chars;
		let ghost = glitchOut; // RGB split strength
		switch (block ? 'none' : enter) {
			case 'pop':
				transform = `scale(${p})`;
				o *= clamp01(q * 3);
				break;
			case 'stagger':
				transform = `translateY(${(1 - p) * 0.55}em) rotate(${(1 - p) * 10}deg)`;
				o *= clamp01(q * 2);
				break;
			case 'wave': {
				const bob = Math.sin(2 * Math.PI * (t * 1.4 - u.index * 0.11)) * 0.05;
				transform = `translateY(${(1 - p) * 0.55 + bob * p}em)`;
				o *= clamp01(q * 2);
				break;
			}
			case 'rise':
				transform = `translateY(${(1 - e) * 0.6}em)`;
				o *= e;
				break;
			case 'blur':
				filter = `blur(${(1 - e) * px * 0.22}px)`;
				transform = `scale(${1.25 - 0.25 * e})`;
				o *= e;
				break;
			case 'flip':
				transform = `perspective(${px * 6}px) rotateX(${(1 - p) * -95}deg)`;
				o *= clamp01(q * 2.5);
				break;
			case 'mask':
				transform = `translateY(${(1 - e) * 110}%)`;
				break;
			case 'type':
				break;
			case 'scramble':
				if (ut < SCRAMBLE_S) {
					const tick = Math.floor((ut * fps) / 2);
					chars = chars.map((c, k) => (/\s/.test(c) ? c : SCRAMBLE[Math.floor(random(`s${u.index}-${k}-${tick}`) * SCRAMBLE.length)]));
				}
				break;
			case 'glitch':
				if (ut < GLITCH_S) {
					const f = Math.floor(ut * fps);
					const k = 1 - ut / GLITCH_S;
					ghost = Math.max(ghost, k);
					transform = `translateX(${(random(`gx${u.index}-${f}`) - 0.5) * px * 0.35 * k}px)`;
					if (random(`go${u.index}-${f}`) < 0.18 * k) o = 0;
				}
				break;
		}
		if (glitchOut > 0) {
			const f = Math.floor(lt * fps);
			transform += ` translateX(${(random(`ex${u.index}-${f}`) - 0.5) * px * 0.4 * glitchOut}px)`;
			if (random(`eo${u.index}-${f}`) < 0.3 * glitchOut) o = 0;
		}
		return {style: {display: 'inline-block', opacity: o, transform: transform || undefined, filter: filter || undefined},
			chars, ghost};
	};

	const renderUnit = (u: Unit) => {
		const {style, chars, ghost} = unitStyle(u);
		const s = chars.join('');
		const d = ghost * px * 0.09;
		return (
			<span key={`u${u.index}`} style={{...style, position: 'relative'}}>
				{ghost > 0.02 ? (
					<>
						<span style={{position: 'absolute', left: d, top: 0, color: '#00E5FF', opacity: 0.85, mixBlendMode: 'screen', WebkitTextStroke: 0}}>{s}</span>
						<span style={{position: 'absolute', left: -d, top: 0, color: '#FF2BD6', opacity: 0.85, mixBlendMode: 'screen', WebkitTextStroke: 0}}>{s}</span>
					</>
				) : null}
				{props.gradient && strokeW > 0 ? ( // a clipped gradient sits under the stroke: outline a copy below it
					<span style={{position: 'absolute', left: 0, top: 0, color: strokeColor}}>{s}</span>
				) : null}
				<span style={{position: 'relative', ...paint(u.emph), ...(props.gradient ? {WebkitTextStroke: 0} : {})}}>{s}</span>
			</span>
		);
	};

	// highlighter behind an emphasis word, swept in once the word has landed
	const markOf = (word: number, first: Unit) => {
		if (!props.mark || !emph[word]) return null;
		const landed = lt - (block ? 0.2 : first.index * stagger + ENTER_S * 0.6);
		const k = Easing.out(Easing.cubic)(clamp01(landed / 0.25));
		return (
			<span style={{position: 'absolute', left: '-0.08em', right: '-0.08em', top: '18%', bottom: '4%', background: props.mark,
				transform: `scaleX(${k}) rotate(-1.5deg)`, transformOrigin: 'left center', borderRadius: '0.1em', zIndex: -1}} />
		);
	};

	const caretOn = enter === 'type' && lt < index * stagger + 0.4 && Math.floor(t * 2.5) % 2 === 0;
	const shown = enter === 'type' ? Math.min(index, Math.floor(lt / stagger) + 1) : index;
	return (
		<AbsoluteFill style={{pointerEvents: 'none'}}>
			<div style={{...placeStyle(props, safe, W, H), width: (props.maxWidth ?? 1 - 2 * safe.side) * W, textAlign: align}}>
				<div style={{display: 'inline-block', transform: `translate(${tx}px, ${ty}px) scale(${scale}) rotate(${props.rotate ?? 0}deg)`,
					opacity, filter: filters || undefined, background: props.box, borderRadius: px * 0.22,
					padding: props.box ? `${px * 0.14}px ${px * 0.38}px` : undefined}}>
					{units.map((ws, line) => (
						<div key={line} style={{overflow: enter === 'mask' ? 'hidden' : undefined, lineHeight: props.lineHeight ?? (display ? 1.08 : 1.2),
							fontFamily: display ? FONT.display : FONT.body, fontWeight: display ? 400 : 800, fontSize: px,
							letterSpacing: display ? '0.01em' : '-0.01em', whiteSpace: 'nowrap',
							WebkitTextStroke: strokeW > 0 ? `${strokeW * 2}px ${strokeColor}` : undefined, paintOrder: 'stroke fill', color}}>
							{ws.map((us, k) => {
								const word = us[0].word;
								const visible = us.filter((u) => u.index < shown);
								if (enter === 'type' && !visible.length) return null;
								return (
									<React.Fragment key={word}>
										{k ? ' ' : null}
										<span style={{position: 'relative', display: 'inline-block', zIndex: 0,
											fontSize: emph[word] ? '1.12em' : undefined}}>
											{markOf(word, us[0])}
											{(enter === 'type' ? visible : us).map(renderUnit)}
										</span>
									</React.Fragment>
								);
							})}
							{caretOn && line === units.length - 1 ? <span style={{opacity: 0.9}}>|</span> : null}
						</div>
					))}
				</div>
			</div>
		</AbsoluteFill>
	);
};

// ---- numbers ---------------------------------------------------------------------------------
export type CounterProps = Omit<TextProps, 'text' | 'by' | 'enter'> & {
	to: number;
	from?: number;
	count?: number; // seconds the count runs (default 0.8)
	decimals?: number;
	prefix?: string;
	suffix?: string;
	enter?: 'zoom' | 'pop' | 'slam' | 'none';
};
/** Number with thousands separators: 1,250,000 / 4.5. */
export const fmtNumber = (v: number, decimals = 0): string => {
	const [i, f] = Math.abs(v).toFixed(decimals).split('.');
	return `${v < 0 ? '-' : ''}${i.replace(/\B(?=(\d{3})+(?!\d))/g, ',')}${f ? `.${f}` : ''}`;
};
/** A number counting up to `to` (a price, a stat), landing with a bump. */
export const Counter: React.FC<CounterProps> = ({to, from = 0, count = 0.8, decimals = 0, prefix = '', suffix = '', enter = 'zoom', ...rest}) => {
	const {t} = useMotion();
	const k = Easing.out(Easing.cubic)(clamp01((t - rest.at) / count));
	const land = t - rest.at - count;
	const bump = land >= 0 && land < 0.2 ? 1 + 0.12 * Math.sin((land / 0.2) * Math.PI) : 1;
	return (
		<div style={{position: 'absolute', inset: 0, transform: `scale(${bump})`, transformOrigin: `${(rest.x ?? 0.5) * 100}% ${(rest.y ?? 0.5) * 100}%`}}>
			<Text {...rest} enter={enter} by="word" text={`${prefix}${fmtNumber(from + (to - from) * k, decimals)}${suffix}`} />
		</div>
	);
};

// ---- effects ---------------------------------------------------------------------------------
/** A full-frame flash on a hit (a beat, a stomp, a cut). */
export const Flash: React.FC<{at: number; dur?: number; color?: string; strength?: number}> = ({at, dur = 0.14, color = '#FFFFFF', strength = 0.85}) => {
	const {t} = useMotion();
	if (t < at || t >= at + dur) return null;
	return <AbsoluteFill style={{background: color, opacity: strength * Math.pow(1 - (t - at) / dur, 2)}} />;
};

/** A ring expanding from a point: an impact, a landing. */
export const Shock: React.FC<{at: number; x?: number; y?: number; size?: number; color?: string; dur?: number}> = ({at, x = 0.5, y = 0.5, size = 0.35, color = '#FFFFFF', dur = 0.45}) => {
	const {t, width: W, height: H} = useMotion();
	if (t < at || t >= at + dur) return null;
	const k = Easing.out(Easing.cubic)((t - at) / dur);
	const r = size * H * k;
	return (
		<AbsoluteFill>
			<div style={{position: 'absolute', left: x * W - r, top: y * H - r, width: 2 * r, height: 2 * r, borderRadius: '50%',
				border: `${Math.max(1, size * H * 0.06 * (1 - k))}px solid ${color}`, opacity: 1 - k, boxShadow: `0 0 ${size * H * 0.05}px ${color}`}} />
		</AbsoluteFill>
	);
};

/** Manga speed lines radiating from a point, flickering: energy, a rush, an impact. */
export const SpeedLines: React.FC<{at: number; dur: number; x?: number; y?: number; count?: number; color?: string; inner?: number}> = ({at, dur, x = 0.5, y = 0.5, count = 36, color = '#FFFFFF', inner = 0.28}) => {
	const {t, fps, width: W, height: H} = useMotion();
	if (t < at || t >= at + dur) return null;
	const tick = Math.floor(((t - at) * fps) / 2);
	const env = Math.min(1, (t - at) / 0.08, (at + dur - t) / 0.12);
	const reach = Math.hypot(W, H);
	const lines = Array.from({length: count}, (_, i) => {
		const a = (i / count) * Math.PI * 2 + (random(`la${i}-${tick}`) - 0.5) * 0.15;
		const r0 = inner * H * (0.8 + 0.5 * random(`lr${i}-${tick}`));
		const wd = H * 0.006 * (0.5 + random(`lw${i}-${tick}`));
		const cx = x * W;
		const cy = y * H;
		return `M${cx + Math.cos(a - wd / r0) * r0},${cy + Math.sin(a - wd / r0) * r0} L${cx + Math.cos(a) * reach},${cy + Math.sin(a) * reach} L${cx + Math.cos(a + wd / r0) * r0},${cy + Math.sin(a + wd / r0) * r0} Z`;
	});
	return (
		<AbsoluteFill style={{opacity: 0.75 * env}}>
			<svg width={W} height={H}><path d={lines.join(' ')} fill={color} /></svg>
		</AbsoluteFill>
	);
};

/** Particles bursting from a point: sparks, confetti or emoji. */
export const Burst: React.FC<{at: number; x?: number; y?: number; kind?: 'spark' | 'confetti' | 'emoji'; emoji?: string;
	count?: number; colors?: string[]; size?: number; spread?: number; dur?: number; seed?: string}> = ({
	at, x = 0.5, y = 0.5, kind = 'spark', emoji = '🔥', count = 16, colors = ['#FFD400', '#FFFFFF', '#FF4D6D', '#00E5FF'],
	size = 0.025, spread = 0.3, dur = 0.8, seed = 'b'}) => {
	const {t, width: W, height: H} = useMotion();
	if (t < at || t >= at + dur) return null;
	const lt = t - at;
	const k = lt / dur;
	return (
		<AbsoluteFill>
			{Array.from({length: count}, (_, i) => {
				const a = (i / count) * Math.PI * 2 + (random(`${seed}a${i}`) - 0.5) * 0.6;
				const v = spread * H * (0.55 + 0.6 * random(`${seed}v${i}`));
				const d = v * Easing.out(Easing.cubic)(Math.min(1, lt / (dur * 0.7)));
				const px = x * W + Math.cos(a) * d;
				const py = y * H + Math.sin(a) * d + (kind === 'spark' ? 0 : 0.5 * H * 0.9 * lt * lt);
				const s = size * H * (0.6 + 0.8 * random(`${seed}s${i}`)) * (kind === 'spark' ? 1 - k : 1);
				const c = colors[i % colors.length];
				const spin = (random(`${seed}r${i}`) - 0.5) * 720 * lt;
				const o = 1 - Math.max(0, (k - 0.6) / 0.4);
				const style: React.CSSProperties = {position: 'absolute', left: px, top: py, transform: `translate(-50%, -50%) rotate(${spin}deg)`, opacity: o};
				if (kind === 'emoji') return <div key={i} style={{...style, fontSize: s * 1.6}}>{emoji}</div>;
				if (kind === 'confetti') return <div key={i} style={{...style, width: s, height: s * 0.5, background: c}} />;
				return <div key={i} style={{...style, width: s * 0.28, height: s * 1.6, background: c, borderRadius: s, boxShadow: `0 0 ${s * 0.6}px ${c}`}} />;
			})}
		</AbsoluteFill>
	);
};
