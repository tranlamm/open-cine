// A transparent layer over a rendered video, drawn by Remotion: the scribble items, or a motion
// layer (the Director's TSX on src/motion/kit.tsx, imported as 'opencine-motion').
//   node fx/render.mjs <job.json>
// job.json = {"meta": {width, height, fps, frames}, "out_dir": "...", "stills": [t, …] | "auto"?, "scale": 1?,
//             and either "items": [...] (scribble: src/schema.ts, times already resolved)
//             or "layer": "<file.tsx>", "data": {beats, strong, cuts, words, marks, safe, mattes} (motion: kit.tsx MotionData),
//                "behind": {"<matte>": [frame, …]}? (a pass per matte, only its <Behind> drawn, into <out_dir>/behind-<matte>/)}
// Without `stills`: one PNG per frame where something shows (an item's span; a layer's `spans`,
// else every frame), <out_dir>/f%06d.png (absolute frame numbers), listed as runs of consecutive
// frames for the engine to overlay with FFmpeg.
// With `stills`: one PNG per timeline second, <out_dir>/still-NN.png, at `scale`; "auto" (a layer):
// one per span, once its elements are in.
// With `"normalize": true`: only check the scribble items and print them with their defaults filled.
// Last stdout line is JSON: {"ok": true, "runs"|"stills": …, "spans"?} or {"ok": false, "error": "..."}.
import {bundle} from '@remotion/bundler';
import {ensureBrowser, openBrowser, renderFrames, renderStill, selectComposition} from '@remotion/renderer';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {z} from 'zod';
import {fxSchema} from './src/schema.ts';
import {span} from './src/span.ts';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FONTS = path.join(HERE, '..', 'assets', 'fonts'); // a layer's public dir: staticFile('Anton-Regular.ttf')
const KIT = path.join(HERE, 'src', 'motion', 'kit.tsx');
const ENTRY = `import {registerRoot} from 'remotion';
import {motionRoot} from '../../src/motion/root';
import * as layer from './Layer';
registerRoot(motionRoot(layer));
`;

class Fail extends Error {}
const fail = (error) => {
	throw new Fail(error);
};

/** Frames as runs [first, last] of consecutive frames. */
const runsOf = (on) => {
	const frames = [...on].sort((a, b) => a - b);
	const runs = [];
	for (const f of frames) {
		const last = runs.at(-1);
		if (last && f === last[1] + 1) last[1] = f;
		else runs.push([f, f]);
	}
	return {frames, runs};
};

/** Frames where at least one scribble item shows. */
const itemFrames = (items, meta) => {
	const on = new Set();
	for (const it of items) {
		const [from, len] = span(it, meta.fps, meta.frames);
		for (let f = from; f < from + len; f++) on.add(f);
	}
	return on;
};

/** A layer's `spans` checked and clamped to the video, or null (it draws all along). */
const spansOf = (spans, meta) => {
	if (spans === null || spans === undefined) return null;
	const end = meta.frames / meta.fps;
	if (!Array.isArray(spans) || !spans.every((s) => Array.isArray(s) && s.length === 2 && s.every(Number.isFinite) && s[0] < s[1]))
		fail('spans: export an array of [start_s, end_s] pairs, start < end');
	return spans.map(([a, b]) => [Math.max(0, a), Math.min(end, b)]).filter(([a, b]) => a < b);
};

const spanFrames = (spans, meta) => {
	const on = new Set();
	for (const [a, b] of spans ?? [[0, meta.frames / meta.fps]]) {
		for (let f = Math.floor(a * meta.fps); f < Math.min(meta.frames, Math.ceil(b * meta.fps)); f++) on.add(f);
	}
	return on;
};

/** "auto" stills of a layer: one per span, ~1 s in (or 60 % of a shorter span); every 2 s without spans. */
const autoStills = (spans, meta) => {
	const end = meta.frames / meta.fps;
	if (spans?.length) return spans.map(([a, b]) => Math.min(a + 1, a + (b - a) * 0.6));
	return Array.from({length: Math.min(12, Math.max(1, Math.floor(end / 2)))}, (_, k) => Math.min(end - 1 / meta.fps, 1 + 2 * k));
};

const layerBundle = async (layer, work) => {
	fs.copyFileSync(layer, path.join(work, 'Layer.tsx'));
	fs.writeFileSync(path.join(work, 'index.ts'), ENTRY);
	try {
		return await bundle({entryPoint: path.join(work, 'index.ts'), publicDir: FONTS,
			webpackOverride: (c) => ({...c, resolve: {...c.resolve, alias: {...(c.resolve?.alias ?? {}), 'opencine-motion': KIT}}})});
	} catch (e) {
		fail(`the layer does not build: ${String(e.message ?? e).replaceAll(work, '<layer>').slice(0, 2000)}`);
	}
};

const main = async () => {
	const jobPath = process.argv[2] ?? fail('usage: node fx/render.mjs <job.json>');
	let job;
	try {
		job = JSON.parse(fs.readFileSync(path.resolve(jobPath), 'utf8'));
	} catch (e) {
		fail(`cannot read ${jobPath}: ${e.message}`);
	}
	const layer = typeof job.layer === 'string' ? path.resolve(job.layer) : null;
	if (layer && !fs.existsSync(layer)) fail(`no layer file ${layer}`);
	const check = fxSchema.safeParse({meta: job.meta, items: layer ? [] : job.items ?? []});
	if (!check.success) fail(`invalid ${layer ? 'meta' : 'scribble items'}:\n${z.prettifyError(check.error)}`);
	const {meta, items} = check.data;
	if (job.normalize) return {ok: true, items};
	const outDir = path.resolve(job.out_dir ?? fail('job.out_dir is missing'));
	fs.mkdirSync(outDir, {recursive: true});
	const what = layer ? 'motion' : 'scribble';

	let serveUrl = '';
	let browser;
	let work = '';
	try {
		process.chdir(HERE); // Remotion keeps its Chrome download in <cwd's package>/node_modules/.remotion
		await ensureBrowser();
		if (layer) {
			fs.mkdirSync(path.join(HERE, '.layers'), {recursive: true});
			work = fs.mkdtempSync(path.join(HERE, '.layers', 'l-'));
			serveUrl = await layerBundle(layer, work);
		} else {
			serveUrl = await bundle({entryPoint: path.join(HERE, 'src', 'index.ts')});
		}
		browser = await openBrowser('chrome');
		// a layer's passes: the front (null), then one per matte with only its <Behind> drawn
		const behind = layer ? job.behind ?? {} : {};
		const passes = {};
		for (const pass of [null, ...Object.keys(behind)]) {
			const inputProps = layer ? {meta, data: job.data ?? {}, spans: null, pass} : {meta, items};
			const composition = await selectComposition({serveUrl, id: layer ? 'Motion' : 'Scribble', inputProps, puppeteerInstance: browser});
			passes[pass ?? ''] = {inputProps, composition};
		}
		const {composition, inputProps} = passes[''];
		const spans = layer ? spansOf(composition.props.spans, meta) : null;
		const t0 = Date.now();
		const at = (e) => fail(`the ${what} layer failed while drawing: ${String(e.message ?? e).slice(0, 2000)}`);
		const stillTimes = job.stills === 'auto' ? autoStills(spans, meta) : job.stills;
		if (Array.isArray(stillTimes)) {
			const stills = [];
			const still = (pass, frame, output) => renderStill({...passes[pass], serveUrl, frame, output, imageFormat: 'png',
				scale: Number(job.scale ?? 1), puppeteerInstance: browser, overwrite: true}).catch(at);
			for (const [k, t] of stillTimes.entries()) {
				const frame = Math.min(meta.frames - 1, Math.max(0, Math.round(Number(t) * meta.fps)));
				const output = path.join(outDir, `still-${String(k).padStart(2, '0')}.png`);
				await still('', frame, output);
				const behinds = {};
				for (const [name, frames] of Object.entries(behind)) {
					if (!frames.includes(frame)) continue;
					behinds[name] = path.join(outDir, `still-${String(k).padStart(2, '0')}-${name}.png`);
					await still(name, frame, behinds[name]);
				}
				stills.push({t: Number(t), frame, path: output, behind: behinds});
			}
			return {ok: true, stills, spans, render_s: Math.round((Date.now() - t0) / 100) / 10};
		}
		const draw = async (pass, frames, dir) => {
			let shown = -1;
			fs.mkdirSync(dir, {recursive: true});
			await renderFrames({...passes[pass], serveUrl, frames, imageFormat: 'png', outputDir: null,
				puppeteerInstance: browser, concurrency: job.concurrency ?? null,
				onStart: () => undefined,
				onFrameUpdate: (n) => {
					const pct = Math.floor((n / frames.length) * 10) * 10;
					if (pct !== shown) console.error(`${what}${pass ? ` behind ${pass}` : ''} ${(shown = pct)}%`);
				},
				onFrameBuffer: (buffer, frame) => {
					fs.writeFileSync(path.join(dir, `f${String(frame).padStart(6, '0')}.png`), buffer);
				}}).catch(at);
		};
		const {frames, runs} = runsOf(layer ? spanFrames(spans, meta) : itemFrames(items, meta));
		if (frames.length) await draw('', frames, outDir);
		const behinds = {};
		for (const [name, want] of Object.entries(behind)) {
			const todo = want.filter((f) => f >= 0 && f < meta.frames);
			behinds[name] = {pattern: path.join(outDir, `behind-${name}`, 'f%06d.png'), frames: todo};
			if (todo.length) await draw(name, todo, path.join(outDir, `behind-${name}`));
		}
		return {ok: true, pattern: path.join(outDir, 'f%06d.png'), runs, spans, behind: behinds,
			frames: frames.length, render_s: Math.round((Date.now() - t0) / 100) / 10};
	} finally {
		await browser?.close({silent: true}).catch(() => undefined);
		if (serveUrl) fs.rmSync(serveUrl, {recursive: true, force: true});
		if (work) fs.rmSync(work, {recursive: true, force: true});
	}
};

// Exit once the answer is out: after a layer throws, Remotion keeps retrying frames in the
// background, which would hold the process open until the engine's timeout.
main()
	.catch((e) => ({ok: false, error: e instanceof Fail ? e.message : e.stack ?? String(e)}))
	.then((res) => process.stdout.write(`${JSON.stringify(res)}\n`, () => process.exit(res.ok ? 0 : 1)));
