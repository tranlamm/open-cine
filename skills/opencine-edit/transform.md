# Transform and restyle: keep the shot, change the character or the medium

The footage keeps its **timing, motion, camera path and layout**; what changes is the **character** (a runner → a robot) or the **medium** (photo → oil painting, anime). The tool is `gen_edit`: video-to-video on a **window** of footage (`"S03@1.2-6.2"`, ≤ 15 s, inside one scene). The output has the window's exact length and timing, so its item plays the footage's real sound in step (`plan_solve` attaches `sound`).

| Mode | Model (your private Space) | `refs[0]` | Quota reserved, 5 s window |
|---|---|---|---|
| `swap`: the person becomes another character; background and camera stay | SCAIL-2 | **key frame**: the window's first frame edited into the new character, same pose, position and size | ~420 (Wan ~490) |
| `restyle`: the whole frame takes a new medium; layout and motion stay | LTX-2.3 + Union-Control (depth or canny) | the window's first frame, restyled | ~230 |

Changing **face + body + clothing** of the person in an existing clip while keeping the performance (dance, talking) is the `opencine-swap` skill (`gen_edit` model `h3-swap`, any character picture, no key frame needed). Other costs: a 2K `gen_image` still ~215 s (one ref), a 5 s H3 clip ~364 s. Read `hf_quota()` (`swap_5s`, `restyle_5s`) before each batch.

## When to use

- The user says "turn into…", "change the character", "suit up", "make him a robot / superhero / mascot", "transform": **transform** (§ Transform).
- The user says "change the style / vibe", "make it a painting", "oil painting", "anime", "manga", "fairy tale", "watercolour", "cartoon": **restyle** (§ Restyle).
- Only for stretches with a clear person or object and one continuous take inside the window. A window containing a hard cut: split it into 2 windows.
- Write in Brief `notes`: the mode, the window map, the character design or the look. The G1 summary states the mode, the number of windows and the quota (days of quota).
- **ShotPlan**: a swap / restyle slot is `source.transform` = `{scene, in_s, out_s, mode, look}`; the `id` in `gen_edit` = the slot number (`"5"`), so `plan_solve` picks up the approved clip. Suit-up clips and inserts are normal `generate` slots.

## Transform (example: a runner → armour flies in and locks on → a running robot)

1. **Map.** Pick the swap window `W = S0x@t1–t2` (2–5 s, person clearly visible full or half body, one take). Put `t1` **mid-stride** (one foot in the air): that is the cut into the robot. Choose how the armour arrives:
   - **A (full)**: real footage up to `t0` → an H3 first+last suit-up clip (first = `S0x@t0`, last = the key frame at `t1`) → the robot runs from `t1`. `t1 − t0` ≈ the suit-up clip's playing length (5.17 s conformed plays ~4.1 s).
   - **B (cheap, safe)**: real footage → 2–4 close inserts (forearm plate, knee plate, visor closing) of 0.4–0.8 s, getting shorter → `flash` or `whip` on the visor lock → the robot runs from `t1`.
   - Done when: `scene_notes` has `t0`, `t1`, `t2`, option A/B, and the insert list.
2. **Design.** Write **one** English phrase describing the new character; paste it verbatim into every later prompt. Armour hugging the body (less new background revealed, the swap fits better). Example: `a sleek humanoid robot suit of matte graphite plates with orange accents: segmented forearm gauntlets, shin guards, rounded knee caps, layered shoulder pauldrons, a chest plate with a small glowing core, and a smooth visored face plate`.
3. **Stills** (one `gen_image` batch, then `gen_sheet` → **G2**):
   - key frame: `refs=["S0x@t1"]`, template P1, one take;
   - each insert: a "before" still (a close angle from the real frame, recipe 5 of [camera.md](camera.md)) and an "after" still = an edit **of that before still** (template P2), so both ends share the framing;
   - Done when: the key frame matches pose, size and position of frame `t1` (compare in `gen_sheet`), and the armour matches the design in every still.
4. **Clips** (after G2; one `gen_edit` batch + one `gen_video` batch, then `gen_sheet` → **G3**):
   - swap: `gen_edit(pid, slots=[{id, source: "S0x@t1-t2", mode: "swap", refs: [key frame], subject: "runner", prompt: P3}])`;
   - suit-up (option A): `gen_video` with first `S0x@t0`, last = the key frame (or `gen/clips/<swap>~<seed>.mp4@0` once the swap is approved, for an exact match), prompt P4, `seconds` 6.58–8;
   - inserts: `gen_video` with first = the before still, last = the after still, prompt P5, `seconds` 5.
   - Done when: no clip has a `garbage` / `length` / `cut` flag, and the robot in the swap clip's first frame matches the key frame's pose.
5. **Edit.** Real footage (hook: the runner, curiosity text) → suit-up or inserts → the running robot. On the final lock (visor closing): `ramp` down to 0.35–0.5× and back to 1× just before the cut, the lock on a strong beat, a 2–4 frame `flash` or a `whip` in the running direction. Cut into the robot **on action** (mid-stride). SFX: whooshes as plates fly, servo whine, metal clank + a short hiss on each lock, an electric hum as the visor lights; after the cut, heavier footsteps (`sfx_list(pid, "metal clank")`, "servo", "hiss"). The swap item keeps the `sound` `plan_solve` attached.

## Restyle (example: cycling round a lake → an oil painting)

1. **Window map.** Each restyled shot is one window ≤ 5 s (a short shot in the edit means a short window: cheaper). A long single take: chain the windows, each next window taking `refs[0] = "gen/clips/<previous window>~<seed>.mp4@last"` to hold one look. Done when: the window list + total quota are in `scene_notes`.
2. **Look.** One shared English look phrase (medium, strokes, palette, light). `control`: `depth` by default (keeps layout and camera path); `canny` for manga, line art, flat-shaded anime. Bicycles or machinery: always include the structure-keeping sentence (template S1).
3. **Look stills** (cheap; where the look is decided): `gen_image(pid, slots)` with `refs=["S0x@in_s"]` and prompt S1 for each window's first frame → **G2**. Done when: the user picked the look; every still shares it.
4. **Clips**: `gen_edit(pid, slots=[{id, source: "S0x@a-b", mode: "restyle", refs: [approved look still], control, prompt: S2}])` → `gen_sheet` → **G3**. Done when: no flags; `gen_sheet` shows the structure held (road, bike, person not melting) and no flicker between first, middle and last frames.
5. **Edit**: restyle items get `color: "none"` (the solver sets it). **Real → painting within one moment**: the footage item `S0x`, then the restyle item of the same window with `transition_in` `dissolve` (0.5–1 s) or `mask`; the restyle item's `in_s` = (footage second where the change starts) − `a`, so both sides overlap on the same moment. Music and SFX to suit the look (oil painting: soft piano / strings; anime: bright music, whooshes).

## Prompt templates (English)

Qwen-Image-2.1: instruction first, change pushed until obvious, one shared keep sentence ([gen.md](gen.md)). Several images: `<image1>`, `<image2>`.

- **P1 key frame** (`refs=["S0x@t1"]`; add `<image2>` = an approved design still if any): `Turn the runner in the image into a robot: replace his skin, hair, face and clothing with <DESIGN>, fitted to his body so the robot holds his exact running pose, limb angles, stride and proportions, at the same position and size in the frame; the armor is lit by the scene's own light, with matching reflections and a contact shadow. This is a local edit: keep the framing, zoom level, camera angle, background, light direction and colour grade unchanged.`
- **P2 insert "after"** (`refs=[before still]`): `Encase the forearm and fist in the image in a segmented robot gauntlet from <DESIGN>: plates wrap the arm from wrist to elbow and the back of the hand, fully closed and locked, seams tight, a small status light glowing on the wrist. Only the forearm and hand change; keep the framing, zoom level, arm position, background blur, light direction and colour grade unchanged.` Knee: `knee cap and shin guard plates closed around the knee and calf`. Face: before = helmet on, visor open; after = face plate closed, visor lit.
- **P3 swap** (SCAIL-2 was trained on long prompts: 90–140 words **describing the video after the change**, never "replace" / "swap", never mention a mask): `A humanoid robot in <DESIGN> runs along <the riverside path> toward the camera at a steady pace, arms swinging, plates flexing at the knees and elbows, the chest core glowing softly; <the morning light> glints on the graphite plates and casts a moving shadow on <the asphalt>. <Trees and railings> pass behind; the camera <tracks backward ahead of it> at the same distance, as in the original shot …` (fill to ~100 words: the new character's look, the action, the setting, the light, the camera).
- **P4 full-body suit-up** (H3 first+last, Context-IR as in [gen.md](gen.md)): `integrated_multimodal_description: [Shot 1] Live-action, cinematic, <light/lens>, a <medium-wide tracking> shot frames the runner shown in Picture 1, preserving the street, the light and the layout. The camera <tracks backward ahead of him> as he keeps running toward the lens at a steady pace, in real time at a natural pace; armor plates fly in from both sides of the frame, slow just before contact and snap onto his forearms and shins, then knee caps and shoulder pauldrons clamp on and a chest plate closes over his torso with a burst of sparks, then a helmet folds over his head and the face plate slides down and seals, his stride never breaking, and the fully armored robot settles into the pose, spacing, and composition established by Picture 2 at the end of the shot. The whole shot is one continuous take.` + `overall_soundscape: Rhythmic footsteps and breathing, whooshes of incoming plates, sharp metallic clanks, servo whirs and short pneumatic hisses as each piece locks; by the end the footfalls ring heavy and metallic.` Every piece has a path: where it comes from, slows, locks, rests. More than 3 beats: split into 2 clips (limbs → torso, helmet).
- **P5 insert** (H3 first+last, 5 s): `… an extreme close-up frames the clenched forearm shown in Picture 1, preserving the skin, light and background blur. The camera pushes in with small amplitude as the first armor plate flies in from off-screen right at once, slows just before contact and clamps around the wrist with a small spark, in real time; a second plate slides along the forearm and snaps shut, then the back-of-hand plate drops into place, the status light flickers on, and the arm settles into the armored forearm, framing, and composition established by Picture 2 at the end of the shot. …`
- **S1 look still** (`refs=["S0x@a"]`): `Transform the photograph into <MEDIUM> across the entire image. Preserve the original vertical composition, viewpoint and depth, the cyclist's pose, position and size, the bicycle's frame, round wheels and spoke geometry, the road and the horizon line, and the arrangement of everything in the scene. <STYLE> Let the strokes describe the wheels' round rims and straight radial spokes and keep the bicycle's mechanical structure exact. Keep the scene recognizable and compositionally faithful while every area, including the sky and the water, is fully rendered as <MEDIUM>.`
  - oil painting: MEDIUM `a luminous impressionist oil painting on fine canvas`; STYLE `Construct every object from confident layered brushstrokes, broken color, subtle impasto highlights and painterly transitions; warm late-afternoon light, cool violet-blue shadows.`
  - fairy tale: `a classic fairy-tale storybook illustration`; `soft gouache and watercolour washes, delicate dark-brown ink outlines, a warm pastel palette of honey, rose and sage, a gentle golden glow, flowers and vines along the path, soft paper texture.`
  - manga (`control: "canny"`): `a black-and-white manga panel`; `crisp black ink line art on white paper, dot screentone mid-tones, solid black shadows, fine hatching, speed lines behind the bicycle.`
  - anime (`canny` for flat line work): `a 2D anime film frame`; `cel shading, clean even line art, flat colour fills with one hard shadow tone, a hand-painted background with soft cumulus clouds and saturated summer colours.`
- **S2 restyle** (LTX: 4–8 sentences in one paragraph, scene → look → action → camera; do not re-describe every background object). Describe the action of the **whole window**: look at the `sheet_scene` of the whole window, not just the first frame, and name every object that is picked up or enters the frame after the first frame. Control keeps only shapes, so an object missing from the prompt is redrawn as something else that fits the prompt (a white pan becomes a dish eaten with chopsticks). `<MEDIUM>, <STYLE in one sentence>. A cyclist rides along a lakeside path past trees and still water, pedalling at an easy pace, then lifts one hand to wave. The camera follows alongside at the same distance, as in the original. Soft <light> keeps one consistent direction across the frame.`

## Self-review (on top of R1–R10)

- **T1 Matching cut**: `review_sheet(cuts=[…])` at the cut into the swap / restyle clip: pose, position and running direction continuous. A visible jump → regenerate the suit-up with last = `…swap…mp4@0`, or cover it with `flash` / `whip`.
- **T2 One design, one look**: the armour is identical in every shot; the look is identical in every window.
- **T3 Background and structure**: the swap does not make the background drift; the restyle does not make wheels, the road or faces melt or flicker, and objects in hand do not turn into something else. Fault → re-roll the slot (new seed); restyle: change `control` (an object changed: add it to S2); swap: write P3 more precisely about the character.
- **T4 Sound**: swap / restyle items carry a matching `sound` (hint `out of step` if not).

## Feedback

- "The armour is ugly", "change the design": edit the design phrase → redo the key frame and insert stills (G2) → swap, suit-up, inserts (G3).
- "The robot is off-pose", "the old person shows through": regenerate the swap with a key frame that hugs the person's shape more closely; still wrong → new seed, shorter window.
- "Stronger / weaker style": change the look at the still stage (G2), not on the clip. "Structure is warped": restyle with `control: "canny"` or a less distorting look.
- General preferences (likes warm oil painting, hates flashes): `taste_note`.
