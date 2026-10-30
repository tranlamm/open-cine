# Scribble: drawing on things in the edit

Contents: When to use · Density · Variety · Timing · Recipes by shot type · Rules · Workflow · Items · Missing styles.

The `opencine` engine builds the edit. The scribble layer (Remotion, in the engine's `fx/`) does **one job**: draw on the rendered video, **following the shape of things**: outlines hugging the silhouette, partial fills, glowing rings wrapped round (back half hidden behind the thing), motion trails, doodles stuck to the thing and doodles strewn around it. Styles: hip-hop, doodle, cyberpunk.

You say **which thing** (`on`, in English) and **when** (`at`). Florence-2 finds the thing; EdgeTAM carries its mask frame by frame; the strokes follow the mask, so they move and change shape with the thing.

What users like: dance videos covered in neon on nearly every shot; portraits full of doodles (crowns, drawn-over eyes, moustaches, a background packed with black doodles); thick white outlines round people with X eyes, lightning bolts, bubbles. In short: **dense, layered, colourful, always moving**.

## When to use

- **Only when the user explicitly asks**: in chat or in the first request.
  - Never add it by template or moment. Never offer it at G1 if the user has not mentioned it.
  - `taste` decides only the style (palette, look, density), never whether to draw.
- Once asked, **dense is the default** (below). The user says "a little", "light", "only a few places": 1–2 strokes on the best shots.
- The user says "drop the scribble": deliver the plain `opencine` render.

## Density: nearly every shot gets strokes

- **Every shot ≥ 0.7 s has strokes.** Shorter shots: one quick stroke (`draw_s` 0.15), or leave it empty as a rest, at most 1 empty shot in a row.
- **3–8 strokes per shot, in layers:**

| Layer | Strokes | Items |
|---|---|---|
| 1. Shape | 1–2 | `outline` (thick white `draw`, or neon `run`), `ring` wrapped round the waist / wrist, a partial `fill` (shirt, jeans, cup) |
| 2. Accents on the thing | 1–3 | `doodle` `on` the thing with `u`, `v` or `spot`: a crown on the head, drawn-over eyes, a moustache, glasses, X eyes, lightning by the shoulder, a flower in the hair, `dashes` beside a contact |
| 3. Motion | 0–2 | `trail` on hands, feet, a swinging or pouring object |
| 4. Background | 0–1 | `scatter`: 6–14 doodles strewn round the subject, keeping clear of it. Good on still shots, close-ups, wides |
| 5. Beat accents | now and then | `frame` neon `run`, `burst` / `sparkle` `pulse` right on the drop |

- **On screen at once**: 3–6 strokes. The subject must stay readable: layer 1 hugs the thing, layer 4 stays off it.
- A 30 s video usually has **40–80 items**. Use `each`, `scatter` and `colors` to write fewer by hand.

## Variety

- **Each shot**: mix ≥ 2 looks and ≥ 2 colours. Vary stroke weight (`stroke` 0.005–0.025) and doodle size (`size` 0.05–0.25), rotate ±25°.
- **Doodle motion**: mix `pop`, `bob`, `spin`, `pulse`, `wiggle`. Closed doodles (heart, star, skull, drop, diamond…) sometimes get `fill` `scribble` to become solid patches, like the black hearts users like.
- **Neighbouring shots** never repeat one combo (same items, look, colours). Change "theme" with the music's sections: a neon section, a black-and-white doodle section, fire on the drop.
- **One palette for the whole video** (3–5 colours; pick one, or follow `taste`):

| Palette | Colours | Good for |
|---|---|---|
| Neon cyberpunk | `#FF2BD6`, `#00E5FF`, `#E6FF00`, `#9D4DFF` + white | dark scenes, dance, night |
| Doodle sketch | black `#111111` + white + yellow `#FFD400` | bright backgrounds, sky, portraits |
| Hip-hop | white + yellow `#FFD400` + red `#FF3B30` + green `#39FF14` | street, fashion, food |

- **Look by background:**
  - `pen`, thick white, on dark footage;
  - `marker`, black or coloured, on bright backgrounds;
  - `neon` on dark scenes;
  - `fire` for hot moments (drop, flaming pan, a big swing).

## Timing

- **On the beat**: get the beats with `beats(pid, name)` when there is music. Put the first stroke of a shot on a beat. Stagger strokes within a shot by 0.1–0.3 s, so the shot "blooms" in layers: shape, then accents, then background.
- **Draw fast**: `draw_s` 0.2–0.4 s (`outline` / `fill` 0.4–0.6 s). Leave with `exit` `cut` on the next beat, or `erase`.
- **Drops and stressed words** (`say:`): the section's biggest stroke, e.g. `ring` with `turns` 2, `outline` `run`, `trail` `fire`.
- `boil` 8–12: the line is redrawn like hand-drawn animation. Neon may use `boil` 0–4 for smoothness.

## Recipes by shot type

| Shot | Suggestion |
|---|---|
| Several dancers, wide | `ring` `each: true` `on` "dancer" round the waist (`colors` alternating); `trail` `neon` / `fire` on the centre dancer's hands; `frame` `run` on the drop |
| One person, full body | `outline` `neon` `run`, or white `pen` with `offset` 0.02; `fill` `solid` `neon` on "jeans" / "jacket"; `trail` on "right hand"; `doodle` `lightning` `spot` `side` |
| Portrait, face close-up | On "face": `eye` at `u` 0.3 and 0.7, `v` 0.42; `mustache` `u` 0.5 `v` 0.74; `glasses` `u` 0.5 `v` 0.42; `cross` (X eyes). `crown` `spot` `top` on "head". Background `scatter`, black or white `marker`, 10–14 doodles |
| Hands at work (cooking, crafts) | `trail` on "hand" / "knife" during swings; `outline` the object in hand; `dashes` / `burst` beside the contact; `sparkle` `pulse` |
| Food or product close-up | thick white `outline` + partial `fill` `hatch`; `crown` `spot` `top`; `flame` (hot), `smoke` (steam), `sparkle`; `scatter` round the plate |
| Wide, transition | `scatter` 10–14 doodles + neon `frame` `run`; `outline` people if any |

## Rules

- **`on`** is 1–3 **English** words for exactly what is visible at the frame where the stroke starts: `"dancer"`, `"man"`, `"face"`, `"head"`, `"red jacket"`, `"jeans"`, `"right hand"`, `"plate"`, `"bottle"`.
  - The engine finds the box **once per shot**, at the first stroke on it, then tracks the mask once until the last stroke. All strokes with the same `on` (same words) in a shot share that result. So **name one thing with exactly one phrase** per shot: cheaper, and the strokes line up.
  - `each: true`: every thing found (up to 6, left to right) gets its own copy of the stroke, e.g. a `ring` round each dancer. `colors: [...]` hands colours to the copies in turn.
  - `doodle` with `u`, `v`: a point in the thing's box (0–1 from the top-left; < 0 or > 1 is outside the box). Without `size`, size = 0.35 × the box width. The doodle rides on the thing and scales with it.
  - **Part of a thing**: name the part (`"jeans"`, `"jacket"`, `"lid"`, `"face"`).
  - **Several things with the same name**: give `x`, `y` near the one you want.
  - **Thing fills the frame (`fills the frame`) or is not seen**: with `x`, `y` (a point **on** the thing) the mask starts from there. Without them the stroke is dropped (reported in `notes`). Food, pan and plate close-ups often fill the frame: give `x`, `y` up front.
  - Florence sometimes boxes the wrong thing (e.g. "pork" lands on the person's shirt). The check sheet shows it: use more specific words or add `x`, `y`.
- **Strokes stop at the cut** of the shot they draw on. A shot shorter than the draw time is drawn faster (`notes`: `drawn in …`). Where the thing runs off the frame edge, there is no outline.
- **Never `fill` a face.** Face doodles (eyes, moustache, glasses) are fine: that is the style.
- The background layer (`scatter`) stays inside `area` [0.05, 0.06, 0.95, 0.78] by default, clear of the captions band and the app buttons at the bottom. A shot with text elsewhere (title, top label): set `area` to avoid it. Doodles never cover text.
- **Without the optional `vision` extra** (`opencine[vision]`): `outline` / `fill` / `ring` are dropped (reported in `notes`); `trail` and `doodle` with `track` follow by optical flow from `x`, `y`; `scatter` cannot avoid the thing.

## Workflow

1. **G1** (the user asked): add a **Scribble:** line to the G1 summary with palette, density and themes per section. Example: "neon cyberpunk, 3–6 strokes on every shot; intro: running neon outline + background doodles; drop: fire ring round the waist + hand trails; outro: crown + white outline". Save it in `Brief.notes`.
2. **S6**, after the R1–R10 self-review and the critic on the plain `opencine` version:
   - **Make a shot table:**
     - read the EditPlan: each item is a shot with timeline seconds;
     - `beats(pid, name)` when there is music;
     - per shot, note the subject (English name) and pick a recipe;
     - write the items per shot (layers 1 → 5), `at` as `S03@…` or a beat.
   - `scribble(pid, items, check=true)`: a check sheet with **one tile per shot**, taken when most strokes are drawn, and a table listing each item, `n strokes` per shot and `notes`.
     - A shot with `0 strokes`: add some. A cluttered shot, or one covering a face: remove some.
     - Off the thing: fix `on` first (more specific words, `x` / `y`), then `u` / `v` / `level` / `offset`.
     - Check sheets count against the self-review budget.
   - Sound (optional): `scribble` / `whoosh` / `pop` SFX in the EditPlan's `sfx` on the big strokes, −14…−10 dB, then `plan_save` → `render`.
   - `scribble(pid, items)` (a job; times are re-resolved against the newest plan) → `path` = `…-preview-scribble.mp4` → **G4**: `review_open(…, media=[path])`, with the palette and section themes in the summary.
3. **S7**: `render(pid, preset="final")` → `scribble(pid, preset="final")` (masks cached from the preview). Give the user the `path` it returns: that is the final.
4. **Feedback only about scribble**: edit the items, then `scribble(pid, items)` (add `check=true` when you change things or places). No new plan. General taste ("love pink neon", "even denser"): `taste_note`.
5. **The user asks for scribble later:**
   - A G4 preview exists: do step 2, then open a new G4 round.
   - A final exists: do step 3 and give the new final path.
   - Nothing rendered yet: note it in `Brief.notes`, do it at S6.
   - No specifics given: make the dense version described above.
6. **Errors:**
   - Item errors (a validation error naming the item, `S03@… is not on screen`, the thing `is not found`): fix as the message says.
   - Missing `fx/node_modules` or any other error: retry once. Still failing: deliver the `opencine` version without scribble and tell the user. Scribble never blocks the video.

**Timing** (on a GTX 1650; expected for a dense 30 s video):
- First check sheet ~1–3 minutes: Florence ~0.5 s per thing per shot; EdgeTAM ~2.2 s per second tracked per thing (15 masks/s with a `trail`, slower).
- Preview render ~1 minute; `neon` / `fire` are ~2× slower per frame than `pen`.
- Masks are cached per plan version, so the final reuses the preview's.

## Items

`scribble(pid, items, version, preset, check, times)`.
- `x`, `y`, `u`, `v` are fractions of the frame or of the thing's box.
- `size`, `stroke`, `offset`, `gap`, `inset` are fractions of the frame width.
- Fields left out take their defaults.

**Common:**
- `at`: timeline seconds, `"S03@2.1"`, `"C01@4.0"` or `"say:word[#n]"`.
- `on`: the thing, in English. Every type except `frame`.
- `each` (every thing found), `colors` (colours handed to the `each` copies).
- `dur_s`, `draw_s` (drawing time), `stroke`.
- `look`: `pen` (solid line + dark shadow) | `marker` (several scribbly passes) | `neon` (bright core + glow) | `fire` (flames, ignores `color`).
- `color`.
- `rough` 1.2 (0 = clean line, 3 = very scribbly); `boil` 8 (redraws per second; 0 = still).
- `exit`: `fade` | `erase` | `cut`.
- `shadow` true; `seed` (change it for another line).

| `type` | Own fields (defaults) |
|---|---|
| `outline` | `style` `draw` \| `run`; `offset` 0.012; `length` 0.3, `laps` 0.5 (with `run`). White, `pen`, 1.6 s |
| `fill` | `pattern` `scribble` \| `hatch` \| `cross` \| `solid`; `gap` 0.022; `angle` −41; `alpha` 0.85; `edge` true. Pink, `marker` |
| `ring` | `level` 0.5 (0 = top of the thing, 1 = bottom); `spread` 1.2; `tilt` 0.28; `turns` 1; `arc` 0.85; `spin` 0.6. Cyan, `neon` |
| `trail` | `tail_s` 0.4; `stroke` 0.02 (head of the trail). Without `on`: `x`, `y` or `track: [x, y]`. Pink, `neon` |
| `doodle` | `kind`; `spot` `side` \| `top` \| `on`, or `u`, `v`; `size` (default 0.14; `top` / `on` 0.8 × the thing's width; `u`,`v` 0.35 ×); `rotate`; `motion` `pop` \| `bob` \| `spin` \| `pulse` \| `wiggle` \| `still`; `fill` `none` \| `hatch` \| `scribble`. With `on` it rides on the thing (`"track": false` to stay put). White, `pen` |
| `scatter` | `count` 10; `kinds` (default: every doodle except face items); `colors`; `size` 0.09 (±35 %); `area` [0.05, 0.06, 0.95, 0.78]; `stagger_s` 0.06; `motion` `bob`; `fill`. With `on` it avoids the thing throughout. White, `marker`, 2 s |
| `frame` | `style` `draw` \| `run`; `inset` 0.035; `length` 0.3; `laps` 0.6 |

**Doodle `kind`:**
- shapes: `sparkle`, `star`, `heart`, `burst`, `swirl`, `squiggle`, `lightning`, `crown`, `diamond`, `music`, `sun`, `smiley`, `moon`, `skull`, `flame`, `drop`, `planet`, `flower`, `spade`, `rainbow`, `dollar`, `asterisk`, `triangle`, `bubble`;
- marks: `exclaim`, `question`, `dashes` (lines radiating from a contact), `smoke` (rising steam), `halo`;
- face items: `eye`, `mustache`, `glasses`, `cross` (X eyes).

One portrait close-up shot (shot 4 starts at `S04`) and one wide dance shot:

```json
[
  {"type": "outline", "at": "S04@0.0", "on": "woman", "offset": 0.02, "stroke": 0.016, "draw_s": 0.4},
  {"type": "doodle", "at": "S04@0.15", "on": "head", "kind": "crown", "spot": "top", "look": "marker", "color": "#FFD400", "motion": "wiggle"},
  {"type": "doodle", "at": "S04@0.25", "on": "face", "kind": "eye", "u": 0.3, "v": 0.42, "motion": "still"},
  {"type": "doodle", "at": "S04@0.25", "on": "face", "kind": "eye", "u": 0.7, "v": 0.42, "motion": "still"},
  {"type": "doodle", "at": "S04@0.35", "on": "face", "kind": "mustache", "u": 0.5, "v": 0.74, "stroke": 0.014},
  {"type": "scatter", "at": "S04@0.3", "on": "woman", "count": 12, "color": "#111111", "fill": "scribble", "size": 0.08},
  {"type": "ring", "at": "S07@0.0", "on": "dancer", "each": true, "level": 0.55, "colors": ["#FF2BD6", "#00E5FF"]},
  {"type": "trail", "at": "S07@0.2", "on": "right hand", "look": "fire", "dur_s": 1.2},
  {"type": "frame", "at": "S07@0.8", "style": "run", "look": "neon", "color": "#E6FF00", "dur_s": 0.8, "exit": "cut"}
]
```

## Missing styles

Extending the engine (in the opencine repository):
- **New doodle**: add the shape to `SHAPES` in `fx/src/items/doodles.tsx` (coordinates −0.5…0.5; `closed` for fillable parts) and to `DOODLES` in `fx/src/schema.ts`.
- **New item type**:
  - write a component in `fx/src/items/`, add its schema, register it in `ITEMS` of `fx/src/Scribble.tsx`;
  - read the `remotion-best-practices` skill first, if you have it.
- **Keep the conventions**: timeline seconds; frame-relative coordinates; contours through `shapeAt`; draw with `sketch` + `Ink` / `Paint` (`fx/src/draw.tsx`).
- **Then check**: run `npx tsc --noEmit` in `fx/` and look at a `check=true` sheet.
