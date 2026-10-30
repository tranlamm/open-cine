---
name: opencine-swap
description: Directs a character swap on a clip the user already has with the `opencine` MCP tools - the person in a dance, talking or action video becomes another character (new face, hair, body and clothing) while motion, expression, timing, camera, background and the original sound stay (MiniMax-H3 character-swap LoRA, gen_edit model h3-swap), with review gates in chat. Use when the user asks to swap, replace or change the person or face in their video, to put themselves or a character into a clip, or replies to a gate of a swap project (mode "swap"). Not for turning someone into a robot or mascot inside an edit, or restyling footage, which is the `opencine-edit` skill, nor for films made from a prompt, which is the `opencine-film` skill.
---

# opencine-swap: Director for character swaps in an existing clip

The user gives a clip (dance, talking, acting…) and a new character (a picture or a description). You remake the clip with the **new character**: face, hair, body and clothing change; **motion, expression, timing, camera, setting and the original sound stay**. Engine: `gen_edit` mode `swap` with `model` `h3-swap` / `h3-swap-hq` (MiniMax-H3 Ref2VA + the character-swap LoRA v1.1; the original clip is a *guide* followed frame by frame).

## Ground rules

- Reply in the user's language; keep it brief. Before a long step, write one line saying what runs and roughly how long.
- **Consent and identity**: put a real person's face in only when the user says it is them, or they have that person's consent. Never make a clip that impersonates a celebrity or a real person to deceive viewers. Fictional, drawn or mascot characters need no question.
- **Ask before spending quota**: no `gen_image` / `gen_edit` before the user has answered G1.
- Slow tools (`ingest`, `render`, `gen_*`, `critic_compare`) may return a `job_id`; poll `job_status(pid, job_id)` until `done`.
- Out of quota (`QuotaExhausted`): wait for the reset; never switch model or method on your own to dodge it. Finished windows are kept; end the turn with "Done X/Y; Z wait for the HF quota reset at HH:MM (`reset_at`). Tell me to continue after that?" and next turn call exactly the same command again (finished windows come from the cache).

## Workflow

Copy this checklist into your todo list (or your reply) and tick items as you go:

```
- [ ] W0 Project created or resumed, clip and character read
- [ ] W1 Window map written
- [ ] W2 DETAILS written
- [ ] W3 Brief + ShotPlan saved, G1 approved (answers recorded)
- [ ] W4 Character still made, G2 approved (skip if the user's picture is used as is)
- [ ] W5 Windows swapped, G3 approved
- [ ] W6 Edit, preview self-reviewed, critic run, G4 approved, final path given
```

### W0 Project and footage

1. New project: `project_create(paths=[clip, character picture…], name="…", mode="swap")` (stills are the character). Existing project: `project_status(pid)` and continue from the step it shows.
2. `taste("swap")` → `hf_quota()`.
3. `ingest(pid)` → poll `job_status` until `done` → `scenes(pid)` → `sheet_overview(pid)` → `sheet_scenes(pid, [scenes with people])`. The user's character pictures: `sheet_images(pid)`.
4. `scene_notes(pid, {…})`: who is to be replaced (several people: describe position / clothing so you can name them), hard cuts, fast-motion or occluded stretches, whether they talk (lip movement).

### W1 Window map

1. Split the stretch to swap into windows of **3–5 s** (10 s at most, but > 5 s tends to drift the face).
2. Windows are contiguous (`out_s` of one = `in_s` of the next), each inside **one scene** (a hard cut = a window boundary).
3. Put boundaries on the original cuts, or on a slow, low-motion moment (never mid-spin or mid-fast-move): each window is generated separately, so a boundary can shift slightly.

### W2 Character description `DETAILS`

1. One English list from head to toe, **starting with the gender**, no article, no final full stop, ≤ 45 words, written from **what the character picture shows** (not the intent): `Female, average height, slim build, oval face, long wavy auburn hair, light freckled skin, oversized cream knit sweater, black leggings, white chunky sneakers`.
2. Paste it verbatim into every slot. Save it in `Brief.notes` with the character picture's path.

### W3 Brief, ShotPlan, then G1

1. `plan_save(pid, "brief", {…})`: `goal`, `style` = the new character, `notes` = DETAILS + the window map.
2. ShotPlan: **no template**, `deliverable` matching the original clip (`1080x1920` portrait, `1920x1080` landscape). One slot per window: `source.transform = {scene, in_s, out_s, mode: "swap", look: "<character>"}`, `dur_s` = [length, length], `transition_out: "cut"`, no `text`, no `sfx`, no music.
3. `plan_save(pid, "shotplan", plan)` → `storyboard_sheet(pid)`.
4. Open **G1** (§ Review gates). Summary: number of windows, the character, the § G1 questions, the total quota (windows × price) and days of quota (daily quota = `hf_quota().limit_s`). More than one day: ask whether to run in batches.

### W4 Character still, then G2 (when a new picture is needed)

1. `gen_image(pid, slots)`: a **full-body, standing straight, facing the camera, plain grey background** still, `aspect` `"9:16"` (templates C1–C3).
2. `gen_sheet(pid, [paths…])` → open **G2**.
3. Approved: `review_decide(pid, "G2", "approved", note, picks={…})` → rewrite DETAILS from the approved still.

### W5 Swap the windows, then G3

1. One `gen_edit(pid, slots)` batch, one slot per window: `{id: "<slot number>", source: "S01@a-b", mode: "swap", model, refs: [character picture], prompt: DETAILS, subject, aspect}`.
   - Same `refs[0]` and the same DETAILS for every window.
   - `subject` = who to replace, in English, when > 1 person shows (`"the dancer in the red dress"`); otherwise leave it out.
   - The engine cuts the character picture out onto grey and matches the clip's colours to the window's background.
2. Read `flags` (`length`: shorter than the window; `garbage`; `frozen`) → `gen_sheet` → § Self-review → open **G3**.
3. Approved: `review_decide(pid, "G3", "approved", note, picks={"1": "gen/clips/1~4821.mp4", …})`.

### W6 Edit, then G4 and the final

1. `plan_solve(pid)` → EditPlan: each item covers its whole window, carries `sound` (the original sound at the exact moment), transition `cut`, no added music or text (unless the user asks). Unswapped stretches (partial swap) are ordinary footage items.
2. `render(pid, preset="preview")` → `review_sheet(pid, version, cuts=[window boundaries])` → § Self-review.
3. `critic_compare(pid, a, b, feedback, gate="G4")`: optional (needs the Codex CLI, `OPENCINE_CRITIC=codex`); `winner: "skipped"` → continue; a loss or regression → fix once, exactly there. Never mention it to the user.
4. Open **G4** with the preview in `media`.
5. Approved: `render(pid, preset="final")` → give the user the returned `path` (on Telegram: also `telegram_send("<pid> final", [path])`). Sharpening (`gen_enhance`) only when the user asks.

## G1 questions

Numbered, each with your proposal and its price from `hf_quota()`:

1. **Model**: `h3-swap` (576×1024, 8 steps; `calls.h3swap_5s`, ~575 s per 5 s window), proposed; `h3-swap-hq` (two passes: 416×736, then a latent upscale to 768×1344 + 3 steps, the second pass with the author's LMS sharpness LoRA at 0.7; clearer detail and less "AI look", unmeasured; `h3swap_hq_5s`, ~1.5×).
2. **Character**: (a) the user's picture, used as is when it shows the **whole body, a clear face, a clean background**; (b) only a face picture or a difficult one → `gen_image` one full-body still from it (2K ≈ 215 s with one ref), reviewed at G2; (c) only a description → `gen_image` with no `refs` (text-to-image), a full-body still, reviewed at G2.
3. **Clothing**: the LoRA takes the clothing **from the character picture** too. To keep the original outfit, the character picture must wear it: `gen_image` with `refs=[character picture, "S01@t"]` (template C2). Propose based on what the user wrote.
4. **Stretch to swap**: the whole clip (default) or only part of it.

## Still templates (Qwen-Image, `gen_image`)

- **C1 full body from a face picture or a difficult one** (`refs=[user's picture]`): `Full-body photo of the person in <image1>, same face, hair and skin, standing straight and facing the camera, arms relaxed at the sides, whole body from head to shoes in frame, wearing <OUTFIT>, plain flat mid-grey studio background, soft even front light, sharp focus.`
- **C2 keeping the original outfit** (`refs=[character picture, "S01@<t>"]`, t = a moment where the outfit is clearly visible): `Full-body photo of the person in <image1>, same face, hair, skin and body shape, now wearing exactly the outfit worn by the person in <image2> (same garments, colours, fit and shoes), standing straight and facing the camera, whole body in frame, plain flat mid-grey studio background, soft even front light.`
- **C3 from a description only** (no `refs`): `Full-body photo of <DETAILS as a sentence>, standing straight and facing the camera, arms relaxed, whole body from head to shoes in frame, plain flat mid-grey studio background, soft even front light, photorealistic.`

## Self-review (G3, G4)

- **S1 Identity**: the same face, hair, build and clothing in every window and throughout each window; nothing left of the old person's face, hair or clothes (check hands, nails, teeth, accessories).
- **S2 Performance**: pose, hands, gaze, expression and lip movement follow the original; no lag.
- **S3 Background**: redrawn from the guide (not pasted pixels), so small text and logos may change; no drift, no flicker.
- **S4 Window boundaries**: `review_sheet(cuts=…)`: character and background continuous across each boundary. A visible jump → move the boundary to a calmer moment or an original cut, then regenerate the two neighbouring windows.
- **S5 Colour and sound**: colours match the original (a small `extra.colour.de_after`, whites not blown); the original sound in step (hint `out of step` if not).

## Review gates in chat

Open a gate:
1. Self-check first (`storyboard_sheet`, `gen_sheet`, `review_sheet`).
2. `review_open(pid, gate, summary, media=[paths…])`. The engine adds gate, pid and round, and at G1–G3 the `Cost:` line; `media` is numbered in the order given.
3. Show it to the user:
   - Terminal or IDE chat: the `message` as it is, then each item as `n. <abs path>`.
   - Telegram (the user asked for it, or you run in a Telegram chat): `telegram_send(message, [item abs paths in order])`, then reply one line: `<gate> sent to Telegram.`
   - `telegram_send` errors with "Telegram is off" and you run on Hermes: write the `message`, then each item as `n.` with a line `MEDIA:<abs path>` under it (Hermes attaches the file).
4. End the turn there: the user replies in chat.

Next turn:
1. Read the reply ("all ok" · "ok 1,3; 2 lower camera angle" · "cancel").
2. `review_decide(pid, gate, status, note, picks)` **before** acting: `approved` → next step; `changes` → fix and open a new round; `cancelled` → stop. Partial approval at G2/G3: `changes` with `picks` for the approved windows; redo only the rest.
3. Act on it. G1 and G4: at most 3 rounds; G2/G3: until the user is happy. A closed gate cannot be reopened; a window re-rolled after G3 is shown in the next G4 round, with `picks` in that `review_decide`.

Resuming: `project_status(pid)` first, trusted over memory. Files the user adds later (another character picture, a better copy of the clip): `project_add(pid, paths)`, then `ingest(pid)`.

Ending a turn: only right after opening a gate, after giving the final path, or when you need a decision (the last line is a question ending in "?"). Never end on "next I will…": do it now.

## Feedback and re-rolls

Fix at the lowest level, one change at a time: move a boundary / trim in the EditPlan → re-roll one window with a new seed → a shorter window (3 s) → fix the character picture (full body, plain background, clear face; then re-roll every window) → `h3-swap-hq`.
- "Doesn't look like the character", "the old face shows through" → a character picture with a clearer face and full body, DETAILS matching the picture, shorter windows, a new seed.
- "Lost the original outfit" → a character picture from template C2.
- "Blurry", "looks AI", "texture flicker" → `h3-swap-hq`; still not sharp enough → `gen_enhance` (when the user asks).
- "Face smeared / waxy" in wide shots (small face) → `gen_face(pid, [slot ids])`: redraws only the face at high resolution, mouth held to the original sound; a face already large is skipped with the reason. Point the item's `source.gen` at the `.face.mp4` file. Run it before `gen_enhance`; cost `hf_quota().calls.face_5s`.
- "Swapped the wrong person" → describe `subject` more precisely (position, clothes, colours).
- General preferences (likes keeping the original outfit, hates visible boundaries) → `taste_note(text, scope="swap", evidence="<pid>@G4")`.

## Limits (tell the user honestly when relevant)

- Each window ≤ 10 s, most stable at 3–5 s. Small faces, occlusion, fast spins, hard angles, strong light changes and long clips make identity drift.
- The background is redrawn from the original, not kept pixel for pixel. Turning a person into a robot, armour or a mascot (no face to keep) fits SCAIL-2 better: the transform mode of the `opencine-edit` skill.
- Several people in frame: one person per run (`subject`).
