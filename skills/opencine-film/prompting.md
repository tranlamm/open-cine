# Film prompts: MiniMax-H3 and Qwen-Image-2.1

Prompts are written **in English**, except dialogue inside `<d>` and text shown in the picture; translate the user's words, and summarise prompts in the user's language in gate summaries. Main sources: the official prompt guides in the [MiniMax-H3 model repo](https://huggingface.co/MiniMaxAI/MiniMax-H3) (`VIDEO_PROMPT_WRITING_GUIDE_base_en.md`, and `VIDEO_PROMPT_WRITING_GUIDE_ref_en.md` for Ref2VA) and the sample IRs in its model card.

Contents: H3 clips · Clips from reference pictures (Ref2VA) · Size and weak spots · Checklist before each gen_video · Stills (Qwen) · Examples.

## H3 clips: Context-IR

H3 was trained on Context-IR text: the closer a prompt is to an IR (dense, concrete, every detail visible or audible), the better the picture.

```
integrated_multimodal_description: [Shot 1] Live-action, cinematic, <look: format / lens, light source + direction, 3–5 colours, grain>, a <shot size + angle> frames <setting: 2–3 landmarks + position in frame>. <Character: fixed description> <position in frame>, facing <direction>. The camera <one move> [with small|large amplitude] [at slow|fast speed] as <character> <strong verb + distance>; <background motion: dust, fabric, steam> while <synchronised sound>. By the end of the shot <final state>. [Shot 2] At 00:04.000, the camera cuts to a <new size / angle> of the woman from Shot 1, … [Shot 3] At 00:08.000, the camera cuts to … Early in the shot <beat>; midway <beat>; in the final second <final state, held still>.

overall_soundscape: <ambient bed>. <physical sounds in the order they happen>. <wordless human sounds>.

non_diegetic_music: <instruments> at a <tempo>, <rhythm>, rising toward the cut at 00:08.000 and stopping on the final beat. | N/A
```

- **Alignment line** (`For the target video…`): the engine writes it from the frames you give (text-to-video: none). Never write it yourself.
- **Order within each shot**: look (Shot 1 only) → size, angle, lens → subject (fixed description, position in frame, facing) → setting (2–3 landmarks) → light (source, direction, colour) → the camera move tied to when the action starts → the action beats up to the final state → sound inside the action sentence.
- **Density**: ~80–120 words per shot (a 10–12 s clip: 250–400 words; a 5 s I2V: 120–200 words; the whole prompt ≤ 7,000 characters). Add visible / audible detail of the **same** events (materials, light on the face, background motion, sound), never more events. No plot summaries, no empty emotion words ("epic", "beautiful"): emotion shows through posture, expression and light.
- **No negative sentences**: the checkpoint is CFG-distilled with no negative prompt, so "no cars" only adds the word "cars". Describe the state you want (`an empty road`).
- **Action**: each shot is a countable chain with a final state (`takes four long strides to the hull, stops, and presses her palm to the metal`). The size of a person's or object's movement is given by distance, step count and concrete speed; `with small|large amplitude` and `at slow|fast speed` are **camera** words (leave them out for medium). Add secondary motion (hair, fabric, smoke) so the frame never stands still.
- **No timestamps inside a shot**: timestamps only open `[Shot N]`, and each `[Shot N]` is one continuous take. Within a shot use phase words: `Early in the shot…`, `midway…`, `in the final second…`.
- **Several shots**: `[Shot N] At MM:SS.mmm, the camera cuts to …`, timestamps increasing and inside the real length (17n+5 frames at 24 fps: 5 → 5.17 s · 8 → 8.00 · 10 → 10.13 · 12 → 12.25 · 15 → 15.08). Each shot 2.5–8 s (a take over ~8 s usually degrades in its second half: split it into two `[Shot N]` or two clips; this figure is empirical, not official), one shot size, **one camera move** (push, pull, pan, tilt, arc or static; a `tracking` move following the subject can combine with one other move), **one main action**; 3–4 shots for 10–12 s. A cut must bring new information (subject, place, angle, state, time); for a slight change of distance, use a camera move instead. Dissolves / fades inside a clip can be written too (`the shot dissolves to…`).
- **Characters in a clip**: describe fully on first appearance, then `the woman from Shot 1`; ≤ 3 important characters per shot. An object whose state changes between shots must be restated (`the now-empty glass`), or the model keeps the old state.
- **Camera**: Static Shot · Push In / Pull Out (camera body) · Zoom In / Out (lens) · Pan Left/Right · Truck Left/Right · Tilt Up/Down · Pedestal Up/Down · Arc Shot · Tracking Shot · Shake Slightly/Strongly · POV · Roll. Write natural sentences (`The camera pushes in with small amplitude at slow speed as she lifts the letter`), not tags at the end. Orbit → `arcs … with small amplitude` (large circles warp the image); crane → pedestal + tilt; handheld → `shakes slightly`.
- **Sound**: tie each sound to its action sentence (`…slams the hatch with a hollow metallic boom`); `overall_soundscape` 1–4 sentences in order of events; `non_diegetic_music` 1–3 sentences: instruments, tempo, where it rises, where it stops (often on a cut), or `N/A` when the film has a score.
- **Dialogue**: only when the script has lines. The speaker gets an ID `(S1)` + a voice description, the line goes verbatim inside `<d>[English] …</d>`, and the mouth closes when done (`her lips close`). H3 speaks English (it learned mostly English and Chinese): lines in other languages, e.g. Vietnamese, are made with `tts` in the edit, and the character on screen does not speak (`while her lips remain closed`).
- **Text in the picture** (signs…): short, large, in double quotes; titles and credits are made with text / motion in the edit.
- **From a first frame (I2V)**: a short anchor first, `the <subject> shown in <Picture 1>, preserving her face, hair, clothing, colours and the layout of the <place>`, then the action starts in the first second → develops → a held final state; the look comes from the still. First + last: one shot describing the path from the first frame's state to the last frame's. If the two frames are alike, describe a closed loop (peak amplitude in the middle, then back), or the clip will stand still.

## Clips from reference pictures: Ref2VA (`refs`)

`gen_video` with `refs` = a character picture (and a location): every shot of the clip keeps the character, including shots 2–3 of a multi-shot clip (I2V only anchors shot 1). Model `h3-ref` (8 steps, the default with `refs`) · `h3-ref-hq` (16 steps); cost `hf_quota().calls.clip_5s_ref` (≈ 1.4–1.5× a Turbo clip). **Quality not yet measured**: use it only when the user picks it at G1 and `clip_5s_ref` is not in `hf_quota().not_set`.

- At most 3 pictures including the first frame (4 and 5 were tried: the conditioner ran out of GPU memory and spent quota with no clip; the engine refuses a 4th). Pick the most important character and location. The order sets the `<Picture N>` numbers: with `first`, it is `<Picture 1>` and the clip starts exactly on it; the `refs` follow (`<Picture 2>`…); character pictures before location pictures. No `last`.
- Character pictures: stills approved at G2 (the character sheet, or its panels `gen/stills/<id>~<seed>.png#face` + `#body`). Location pictures: wide and **empty of people**, at the scene's time and light. Reference pictures carry no text or labels.
- The prompt is sent verbatim (no alignment line, no rewrite), with exactly the 6 fields of the official reference guide (`VIDEO_PROMPT_WRITING_GUIDE_ref_en.md`), each opening with its field name, separated by one blank line:
  - `subject_definitions`: one line per subject, `<Subject N> is the … in <Picture K>: <the film bible's fixed description>.` A picture that only defines a character or place is named in its Subject line; a first frame gets its own line `<Picture 1> is the first frame of [Shot 1], showing …`. Every picture must be mentioned (the engine refuses before spending quota).
  - `summary`: one paragraph opening with `[reference generation]` (with a first frame: `[reference generation + keyframe completion]`), using only the defined labels.
  - `retention_analysis`: one line per label, `<Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved - …`.
  - `detailed_description`: 1–2 look sentences from the film bible **before** `[Shot 1]`, then the shots as in Context-IR (timestamps, one camera move, action to a final state, sound in the sentence). First appearance of a subject: label + traits + position in frame, then only the label; 350–500 words for the field.
  - `overall_soundscape`, `non_diegetic_music`: as in Context-IR.

## Size and weak spots

The canvas is at most 1344×768 (16:9) or 768×1344 (9:16). Each token covers a 32 px cell and 4 frames, so a frame is only 42×24 tokens: anything smaller than a cell is easily dropped, anything crossing several cells in 1/6 s easily smears.

| Good | Weak → how to avoid it |
|---|---|
| A subject large in frame: close faces, fabric, skin, glass are very sharp | Small, distant, fast objects smear and deform → subject ≥ ¼ of the frame height; a `tracking shot` keeps the subject still in frame; end the shot as the object passes the lens, never as it shrinks into the distance |
| Strong light, backlight, bokeh, smoke, dust, light shafts | Small effects (sparks, rain, small lights) are often dropped → make them big, the subject of their own sentence, with a sound: `A shower of orange sparks bursts across the frame with a sharp crackle` |
| Slow camera moves (push in, arc, tracking) and medium motion | The subject barely moves → strong verb + distance + a counted rhythm + a final state, plus secondary motion |
| Multi-shot cuts on time; characters keep their design across shots | Small faces warp and change identity → faces at medium or close size; fast hand work and fingers → describe each step, avoid finger close-ups; characters rarely blink → write `she blinks once` |

Cost scales roughly with length: a 10 s Turbo clip ≈ 1.7× a 5 s clip (812 vs 484 s reserved), base H3 ≈ 2.2×. A multi-shot clip costs no more than separate shots, but fixing one shot regenerates the whole clip. Real numbers: `hf_quota().calls`.

## Checklist before each gen_video

- [ ] Look sentence at the start of Shot 1, from the film bible; character / scene descriptions copied verbatim from it.
- [ ] Each shot: one size, one camera move, 1–2 strong verbs with distance / rhythm and a final state; no timestamps inside a shot.
- [ ] Subject ≥ ¼ of the frame height; effects big, in their own sentence, with sound.
- [ ] `[Shot N]` timestamps increasing, inside the real length, each shot ≥ 2.5 s, every cut brings new information.
- [ ] ~80–120 words per shot; no negative sentences; no invented `<d>`; no dialogue in languages H3 does not speak.
- [ ] `overall_soundscape` in order of events; `non_diegetic_music: N/A` when the film has a score.
- [ ] `aspect`, `model` (the user's choice) and `seconds` set.
- [ ] `gen_check(pid, "video", slots)` shows no hints (or they are kept on purpose); on a re-roll, `changes.axes` lists one axis.

## Stills: Qwen-Image-2.1

The engine calls Qwen without its official prompt rewriter, so write in the rewriter's own output form: text-to-image is an **observer** describing a finished picture; an edit is an **instruction**.

- **No refs = text-to-image**: one English paragraph in the present tense, 250–500 words. Open with `The image is a wide cinematic film still of <subject>, <setting, palette>.` → move across the frame with position phrases (`In the left third…`, `In the upper-right corner…`, out to the edges) → people: pose, gaze, expression, hair, skin, every garment with colour and material (colours always qualified: `deep navy`) → **one lighting sentence** (source, direction, quality, shadows) → close with one `The overall composition …` sentence. `wide` / `vertical` are fine; no aspect ratios, resolutions, "8K", "masterpiece" or negative sentences. Text nobody needs to read: `plain, unlettered surfaces`, `distant signs blurred and unreadable` (the model likes to invent signs).
- **Edit (with refs)**: the operation first, the change pushed until obvious, then one keep sentence naming what to keep by category, without detail (the more detail, the more it drifts). Several pictures: `<image1>`, `<image2>` in `refs` order, with each picture's role stated; with one picture, "the image" ("Picture N" is H3's naming).
- **Identity comes from the picture**: with a character picture, point at `<image1>` with its role (`defines her face, hair and outfit only, not her pose`), and never describe the face in words; words lower the likeness. The film bible's fixed descriptions are for H3 prompts and text-to-image.
- **Character sheet** (no refs; `aspect` 16:9 even in a 9:16 film, because five figures in a portrait frame are too small): template (a). **Location**: one wide still with no people, at the scene's time and light, written as text-to-image above.
- **First frame of each clip** (`refs` = the approved character sheet and location): the still **is** frame 0 of the clip. Draw the starting pose of the action, the subject ≥ ¼ of the frame height, room left in the direction of gaze / travel, light direction matching the planned shot, sharp, no motion blur: template (b). First + last: the two frames as alike as possible (very different frames → the model cuts); the clip describes the path between them.
- First frame and clip share the `aspect` (16:9: a 2K still is 2688×1536, scaled exactly 2× down to H3's 1344×768 canvas).
- Location pictures used as refs contain no people (the model tends to copy them in). Reference pictures carry no text or labels.
- Character refs for first frames: the 5-panel sheet by default. `<sheet>#face` (the close-up panel) and `<sheet>#body` (the front full-body panel) are cut locally at no quota. Some video models' guides recommend separate pictures (a face close-up + a full body) over a multi-angle sheet, because a sheet can read as several people; for Qwen / H3 this is unmeasured, so keep the full sheet unless a comparison shows otherwise.

```
(a) The image is a wide photorealistic character reference sheet of <fixed description>, on a seamless plain light-grey studio background. From left to right, five evenly spaced panels: a head-and-shoulders close-up of her face with a calm neutral expression, then four full-body views of the same woman standing upright, arms slightly away from her body: front view, three-quarter view, left profile, back view. Every figure has identical proportions, hairstyle, outfit, colours and materials, stands on one shared ground line and fills about four fifths of the panel height. The lighting is soft, even frontal studio light with faint contact shadows. The overall layout is clean, neutral and unlabelled.

(b) Using the woman in <image1> as the identity reference (it defines her face, hair and outfit only, not her pose) and the <place> in <image2> as the location, light and colour grade, generate a cinematic film still of the first moment of the shot: a <shot size> from <camera height>, <distance> in front of her, <lens mm>. She stands on the left third, two thirds of the frame height, turned three-quarters right, weight on her back foot, about to step forward, gazing toward frame right. The right two thirds stay open: <the setting stretching that way>. <Light source + direction> rims her hair and casts a long shadow to the lower right. Deep focus, fine film grain; every surface plain and unlettered.
```

## Examples

**Coastal mountain race: T2V, 12 s, 4 shots.** A shorter version was run: the cuts landed on time, the aerial and the low tracking shot followed the cars well; the two cars shrinking into the distance at the end smeared, so shot 4 ends as the cars pass the lens.

```
integrated_multimodal_description: [Shot 1] Live-action, cinematic, anamorphic 35mm film look, warm golden-hour backlight from a low sun on frame right, long shadows, a palette of amber, rust red, silver and deep sea blue, fine grain. A high wide aerial shot frames a cliffside coastal mountain road that zigzags down the left half of the frame above a glittering sea, a white stone guardrail lining its outer edge. Two vintage endurance race cars, a red one leading a silver one, carve through the switchbacks. The camera tracks the cars from above and pedestals down with large amplitude as they dive into a tight left hairpin, until the two cars fill the lower third of the frame; dust trails from their rear tyres and drifts out over the cliff edge, catching the sunlight. [Shot 2] At 00:03.000, the camera cuts to a close-up through the windscreen of the red car's driver, a man in his forties in an open-face white helmet, round goggles and a rust-coloured scarf, leather gloves gripping a wooden steering wheel. The low sun flares across the curved glass from frame right. The camera shakes slightly with the vibrating car as the driver's eyes flick up to the rear-view mirror, where the silver car's headlights grow larger, then he grits his teeth and hauls the wheel hard to the left as the engine note climbs. [Shot 3] At 00:06.000, the camera cuts to a low tracking shot at bumper height alongside both cars, the silver car on the inside of the bend filling the left half of the frame and the red car the right. The camera tracks with them at fast speed as the silver car pulls level. A large burst of orange flame bangs out of the red car's exhaust with a sharp crack as it downshifts, and both cars slide side by side through the corner, blue-white tyre smoke curling up between them. [Shot 4] At 00:09.000, the camera cuts to a static low-angle shot at the corner exit, the lens a hand's width above the warm asphalt, the road running straight toward the camera. The red car and then the silver car blast past the lens one after the other, each filling the frame for an instant, throwing gravel and a cloud of golden dust that swirls through the backlight. The camera shakes strongly in the rush of air, and in the final second the dust drifts slowly across the empty road.

overall_soundscape: Two high-revving racing engines roar and echo off the cliff walls, rising and falling with each gear change. Tyres squeal through the hairpin, the red car's exhaust cracks with a sharp backfire, and gravel sprays against the guardrail as the cars blast past in a thundering whoosh of wind, followed by settling dust and distant surf.

non_diegetic_music: Low taiko drums at a fast tempo over a pulsing synthesizer bass, brass stabs rising in volume toward the cut at 00:09.000 and stopping abruptly as the cars pass the camera.
```

**Desert: I2V, 5 s, one shot** (first frame = the approved still of Mira before the hull from G2; film bible as in SKILL.md):

```
integrated_multimodal_description: [Shot 1] Live-action, cinematic, anamorphic 35mm film look, low golden backlight, palette of ochre, rust and teal, fine grain, shallow depth of field. The woman shown in <Picture 1> stands on the left third of the frame before the half-buried starship hull, preserving her face, short black hair, sand-coloured hood, rust-red scarf, olive canvas coat, brass-rimmed goggles and the layout of the salt flat. In the first second she pulls the scarf down from her mouth with her right hand and takes two slow steps toward the hull. The camera pushes in with small amplitude at slow speed as she lifts her gaze to a row of large amber lights that switch on one after another along the dark metal, the glow sliding across her cheek, wind tugging at her coat and streaming sand past her boots. In the final second she stops, lips parted, and holds still as the last light burns steady. The whole shot is one continuous take.

overall_soundscape: Steady desert wind hisses over the sand and flaps the canvas coat. Each light switches on with a deep electrical thunk, followed by a low rising hum from inside the hull.

non_diegetic_music: N/A
```

**Desert: Ref2VA, 10 s, 2 shots** (`refs` = [the approved full-body still of Mira, the empty salt-flat still]; not run yet):

```
subject_definitions:
<Subject 1> is the woman in <Picture 1>: a woman in her late twenties with sun-darkened skin, short black hair under a sand-coloured hood, a rust-red scarf, a worn olive canvas coat and brass-rimmed goggles.
<Subject 2> is the desert salt flat in <Picture 2> at sunset: the sun low on frame right, a half-buried starship hull on the left third and a ridge of dunes behind.

summary:
[reference generation] <Subject 1> crosses <Subject 2> to the buried hull and wakes its lights in two shots: a wide approach across the salt, then a low push-in on her face as the lights switch on behind her.

retention_analysis:
<Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved - her face, short black hair, sand-coloured hood, rust-red scarf, olive canvas coat and brass-rimmed goggles are retained.
<Subject 2> (appears in [Shot 1], [Shot 2]): fully_preserved - the salt flat, the half-buried hull on the left third, the dune ridge and the low sun on frame right are retained.

detailed_description:
Live-action, cinematic, anamorphic 35mm film look, low golden backlight, palette of ochre, rust and teal, fine grain.
[Shot 1] A wide shot at chest height frames <Subject 2>: the cracked white crust of the salt flat runs back to a ridge of rust-coloured dunes, the half-buried starship hull lies dark and scarred on the left third, and the low sun burns on frame right, laying long blue shadows across the salt. <Subject 1>, the woman in the sand-coloured hood, rust-red scarf and worn olive canvas coat, walks in from frame right with her shoulder turned to the camera, her long shadow stretching toward the hull. The camera trucks left with small amplitude at slow speed as she takes six long strides across the crust, the wind tugging at her coat and streaming fine sand past her boots with a dry hiss. Early in the shot she shields her brass-rimmed goggles with one gloved hand; midway she lowers it and quickens her pace; in the final second she stops two steps from the hull and raises her right palm toward the pitted metal.
[Shot 2] At 00:05.000, the camera cuts to a low close-up of <Subject 1> from in front of the hull, her face filling the right half of the frame, the brass rims of her goggles catching the last of the sunset and the scarf fluttering against her chin. The camera pushes in with small amplitude at slow speed as she presses her palm flat against the metal with a soft thud. Behind her shoulder a row of large amber lights switches on one after another along the hull, each with a deep electrical thunk, the warm glow sliding across her cheek and the edge of her hood while sand ticks against the plating. She pulls the scarf down from her mouth with her free hand, lips parted, and blinks once; in the final second she holds still as the last light burns steady and a low hum swells from inside the hull.

overall_soundscape:
Steady desert wind hisses over the salt and flaps the canvas coat. Boots crunch on the crust, a palm meets metal with a soft thud, each light switches on with a deep electrical thunk, and a low hum rises from inside the hull.

non_diegetic_music:
N/A
```
