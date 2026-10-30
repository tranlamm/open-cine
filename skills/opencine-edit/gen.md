# Generating stills and clips (S5: `gen_image`, `gen_video`)

- Fixed models: stills **Qwen-Image-2.1** (2K by default, 1536×2688 for 9:16), clips **MiniMax-H3 + Turbo LoRA**, video-to-video `gen_edit` **SCAIL-2** (swap) / **LTX-2.3 Union** (restyle) ([transform.md](transform.md)). Leave `model` out: stills and video-to-video refuse it; for clips, `h3-turbo-hq` / `h3` are for film projects when the user picks them (the `opencine-film` skill).
- Prompts are **always in English** (Qwen-Image-2.1 and MiniMax-H3 were trained mostly on Chinese and English). Translate the user's words; gate summaries describe the prompt's idea in the user's language. Never ask the models to draw words; text belongs to the overlay.
- Refs / `first` / `last`: **frame refs** `"S03@2.5"` / `"C01@12.4"`. The engine takes the full-resolution source frame, the sharpest within ±2 frames in the same scene. Never use `index/frames` images or sheets. A frame of a clip: `"gen/clips/7~12.mp4@0"`, `"…@1.5"`, `"…@last"` (joining a generated clip to a swap clip, chaining restyle windows). New angles are always generated from the source frame, never from a generated still (they drift).
- Pick the angle and the camera move from [camera.md](camera.md) when writing the ShotPlan (`generate.camera`, `generate.motion`); here you turn them into prompts.

## Number of generated shots and quota

**Content decides how many slots are generated; quota only decides the order they run in.** Every angle or action the footage lacks, or shows weakly, is worth a slot; videos that need many new views (ads, cooking) can have 10–20 generated slots. The user weighs the cost and accepts waiting for the quota to reset.

- Canvas is always `full`.
- **Before spending quota**: `gen_check(pid, kind, slots)` (free) gives the prompt hints, what this take changes from the last one (`changes.axes`: change one thing per re-roll), the quota reserved, and `quota`: whether the batch fits the daily quota left (`fits`; `over` = slots no account can still hold). Fix the hints, then run `gen_image` / `gen_video`.
- **One take per slot** (stills and clips): no `variants` > 1, no two slots with the same content and different seeds to pick from. A bad take → re-roll one (new seed, fixed prompt).
- Estimate with the table below (seconds reserved; actual use is lower) and state it in G1/G2/G3: total, number of batches, about how many days of quota. Daily quota = `hf_quota().limit_s` (2,400 s × number of HF accounts; the engine moves to the next account when one runs out). What is really left: `hf_quota()` (from the Hub; `calls` = how many of each call still fit today).
- One call runs on one account and never spans two: with several accounts `left_s` is their sum and `max_call_s` is the largest call still possible. Whether a batch fits: `gen_check` → `quota`, never `left_s` divided up.
- Past the daily quota, calls run on HF credits ($1 / 10 GPU minutes) when `overquota` is on; when the credits run out they stop with `QuotaExhausted`.
- More than the quota left: run in **batches**, hook, payoff and main shots first.
- A batch stopped by `QuotaExhausted`: finished slots are kept; open the gate only when the set is complete. End the turn with a question on the last line: "Done X/Y; Z wait for the HF quota reset at HH:MM (`reset_at`). Tell me to continue after that?" Next turn, call **exactly the same command** again: finished slots come from the cache at no quota.

| Per call | Quota reserved (s) |
|---|---|
| clip `full`, `seconds` 5 · 6.58 · 8 · 10 | 484 · 564 · 654 · 812, including the conditioner's prompt rewrite (a Context-IR prompt skips it: 120 s less). Exact figures for the current setup: `hf_quota().calls.*.needs_s` |
| clip first+last | + 6 |
| 2K still, per variant | about 215 with one ref; exact figure for the refs / aspect from `gen_check` |
| music (`music_gen`), `seconds` 30 · 60 | 81 · 117 (actual use ~18 s for a 30 s track) |

## Stills: Qwen-Image-2.1

- `refs[0]` is the main frame. Output is 2K 1536×2688; the engine scales it exactly 2× down to the clip canvas `full` 768×1344. Landscape frame or off-centre subject: `crop={x, y, w, h}` (0–1 of `refs[0]`); without it the centre 9:16 is cut. Extra refs keep their framing.
- The prompt is **one paragraph, instruction first** (Replace / Add / Remove / Re-photograph…), the change pushed until it is **obvious** (a timid edit that comes back looking like the ref is a failure), then **one keep sentence** naming what to keep by category, without re-describing details (describing what to keep makes it drift):
  - Edit: `Replace the paper cup with a glass of iced lemonade. Keep the hand, the counter, the framing and the light unchanged.`
  - New camera angle. The keep sentence must **not** mention framing / composition / layout / viewpoint (that locks the old angle): `Re-photograph this scene as a new shot from a different camera position: <shot size + angle>, the camera <height, distance, side> pointing <direction>. <Subject> now <position in frame, share of the height, which side shows>; <what is newly revealed>. <lens mm>, <depth of field>, same light direction and colour grade. Keep the same person, face, hairstyle, clothing, colours and setting as the image.` Recipes per angle: [camera.md](camera.md).
- Several refs: call them `<image1>`, `<image2>` in `refs` order (`Put the bottle from <image2> on the table in <image1>.`); with one ref, "the image". "Picture N" is H3's naming, never for stills. State each image's role; a character picture only supplies identity (`<image2> defines his face, hair and outfit only, not his pose`). Never describe the face in words (it lowers the likeness).
- The model invents lettering and signs: write `plain, unlettered surfaces`, `distant signs blurred and unreadable`.
- Avoid: negative sentences ("no…", "don't…": there is no negative prompt), "9:16" / "vertical" / "4K" / "masterpiece", two camera moves in one prompt, rotations over 90°, new angles that hinge on small text / logos or many fingers.
- Angle changes are not guaranteed (2.1 sometimes keeps the old angle): still generate one take; if it comes back almost identical to the ref, rewrite the instruction more strongly (camera position, what is newly revealed), not just the seed.
- Each take is one file `gen/stills/<id>~<seed>.png` (`<id>.png` = the latest); put each item's exact `path` in the gate.
- Check `flags` before G2: `soft` (blurrier than the ref), `colour_shift`, `blank`, `garbage`.

## Clips: MiniMax-H3 (Turbo)

This section is for shorts cut from footage; film projects (multi-shot, 24 fps, no conform) follow the `opencine-film` skill. A generated clip must cut next to real footage without showing: **real-time pace**, action filling the whole length. Slow motion is done in the EditPlan (`speed`, `ramp`) on a real-time clip.

- Prompt in Context-IR form, **one shot**; the engine adds the alignment line itself and skips the rewrite (saves 120 s of quota):

```
integrated_multimodal_description: [Shot 1] Live-action, cinematic, <light / depth of field>, a <shot size + angle> frames <subject> shown in <Picture 1>, preserving <its appearance, colours and the layout>. The camera <move> [with <small|large> amplitude] [at fast speed] as <action that starts at once>, in real time at a natural pace; <beat 2>, then <beat 3>, <final state>. The whole shot is one continuous take.

overall_soundscape: <one sentence of physical sound>

non_diegetic_music: N/A
```

- **The keep clause comes before the motion** (`shown in <Picture 1>, preserving …`): H3 needs it. Then the action starts in the first second → develops → a final state that holds, with background motion (steam, fabric, hair, dust) so the frame never dies.
- **Length**: ~120–200 words for a 5 s clip. Add visible / audible detail (materials, light direction, background motion, sound), never extra events. No negative sentences: the checkpoint is CFG-distilled with no negative prompt, so "no steam" only adds the word "steam"; describe the state you want.
- **Action**: 2–3 linked beats with concrete verbs, enough for the whole `seconds`. Two slices: `as the knife slices down through the pork at once, in real time at a natural pace; the slice tips over onto the board, then the knife cuts a second slice, which falls too`. End on a clear state; the EditPlan cuts before the still part.
- **Camera**: the camera sentence comes second, tied to the moment the action starts (`… as she lifts the cup`). At most two moves in sequence (`pushes in, then arcs left`); no timestamps (H3 reads them as cuts). Leave the speed out (medium) or `at fast speed` for energetic shots.
- Camera words (names from the H3 guide): Static Shot · Push In / Pull Out (camera body) · Zoom In / Out (lens) · Pan Left/Right · Truck Left/Right · Tilt Up/Down · Pedestal Up/Down · Arc Shot · Tracking Shot · Shake Slightly/Strongly · POV · Roll Clockwise/Counterclockwise. Amplitude: `with small|large amplitude`. Outside the list: orbit → arc; crane → pedestal + tilt; whip pan → `pans … with large amplitude at fast speed`; handheld → `shakes slightly`; reveal → `…, revealing …`.
- Turbo is weak at **large motion of hands and small objects** (smears, ghosting around fingers): build rhythm from several medium actions, not one very fast one. Hands hold / lift / set down / slice; liquids in thin streams; avoid 360° orbits and shape-shifting liquids.
- **`seconds`** = the action's real length + ~0.5 s, from 5 to 10. H3 rounds up to 17n+5 frames at 24 fps: 5 → 5.17 s · 6.58 · 8 · 10 → 10.1 s; Turbo is only validated from 5.17 s. A shot that must be long is generated long once.
- **Speed in the edit**: `retime="conform"` (default) plays 24 → 30 fps, 1.25× faster (10.1 s → 8.1 s), offsetting the model's slow bias. If `gen_sheet` shows a rushed action (walking, talking), set the EditPlan item's `speed: 0.8` for true speed. `"interp"` keeps the speed but tends to warp hands and liquids.
- First+last (`last`): one shot describing a **path**: `<subject> begins in the position and framing established by Picture 1 … settles into the pose, spacing, and composition established by Picture 2 at the end of the shot.` Infer the camera move from the difference between the frames (tighter → push in, rotated → arc, higher → pedestal up + tilt down, wider → pull out), amplitude by how different. Both frames share light and objects.
- A plain prompt still works (the conditioner rewrites it, +120 s; the rewrite is stored and reused on re-rolls).
- Output: colours matched to `first`, BT.709, flickering first frames trimmed; muted unless the item sets `gain`. The EditPlan points at the approved `path` (`gen/clips/<id>~<seed>.mp4`), never `<id>.mp4` (overwritten on re-roll); `out` ≤ `item.frames`.
- Before G2/G3: `gen_sheet(pid, [path…])` in gate order to catch what CPU checks miss (extra fingers, melting objects, wrong subject). Obvious failure → re-roll; the rest goes into the gate summary. After approval: `review_decide(…, picks={"3": "gen/clips/3~4821.mp4"})`; `plan_solve` uses the approved take.
- `flags` before G3:
  - `garbage` (noise, failed run) → re-roll.
  - `cut` / `multishot` → rewrite as a Context-IR prompt with **one** `[Shot 1]` (a new seed keeps the rewrite's cut).
  - `frozen` / `static` → too little motion: more beats, stronger verbs, concrete distance / step counts (`walks four steps to the window`; `with large amplitude` describes the camera, not people).
  - `first_frame_mismatch`, `colour_drift`; `flash_head` was trimmed already.

## Sharpening

**Only when the user asks** for sharper / higher-resolution clips; never offer it yourself or put it in G1. Then: `gen_enhance(pid, [slot ids…])` for the generated clips in the edit (FlashVSR+, 768×1344 → 1080×1920, same framing and sound; ≈ 108 s of quota per 5 s clip, 182 s per 10 s clip) → point each item's `source.gen` at the `.1080p.mp4` file (`in` / `out` unchanged) → `plan_save` → render the final → report the quota used. Never sharpen twice.

## Fixing small faces

Faces in wide shots come out smeared or waxy (face under ~120 px tall): `gen_face(pid, [slot ids…])` redraws only the face region at high resolution (a square crop following the face, strong denoise when the face is small, mouth held to the sound) → point `source.gen` at the `.face.mp4` file → `plan_save`. A face already large enough returns `skipped` with the reason. Offer it when G3/G4 shows a broken face in a wide shot; cost `hf_quota().calls.face_5s` (≈ 440 s per 5 s clip). Run it before `gen_enhance`.

## Re-rolls

Same slot, new `seed` (and a fixed prompt when the prompt caused the fault). Re-rolls are unlimited: re-roll on a flag or when the user dislikes a take, until the user is happy.
