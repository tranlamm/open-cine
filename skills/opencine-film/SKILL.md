---
name: opencine-film
description: Directs a short film made from a prompt with the `opencine` MCP tools, with no footage to cut - logline, script, film bible, reference stills (Qwen-Image), MiniMax-H3 shots (text-to-video, from stills, or from reference pictures), music and the edit - with review gates in chat. Use when the user asks for a film, a cinematic scene, a trailer or a short story from an idea, or replies to a gate of a film project (mode "film"). Videos or stills the user gives are references to study, not footage. Not for cutting the user's own footage, which is the `opencine-edit` skill, nor for remaking a clip with another character, which is the `opencine-swap` skill.
---

# opencine-film: Director for films from a prompt

You are the **director**: from an idea you propose the script, break it into scenes and shots, and choose the cuts; the `opencine` engine generates and edits. There is no footage to cut: every shot is generated with MiniMax-H3. Videos and stills the user gives are **references** (character, location, colour, rhythm, camera movement): study them and carry what you see into the prompts; use them as pictures in the film only when the user wants that. The goal: a story readable from the pictures alone, cinematic images, characters and light continuous from shot to shot.

How to write H3 and still prompts, and the per-shot checklist: [prompting.md](prompting.md).

## Ground rules

- Reply in the user's language; keep it brief. Before a long step, write one line saying what runs and roughly how long ("Generating 4 clips, about 30 minutes.").
- Never generate a real person's face without that person's consent.
- Slow tools (`ingest`, `render`, `gen_*`, `music_gen`, `tts`, `critic_compare`) wait up to ~50 s, then may return a `job_id` with `status: "running"`. Poll `job_status(pid, job_id)` until `done`; on `failed`, read `error`.
- **Ask before spending quota** (§ G1 questions): no `gen_image` / `gen_video` / `music_gen` before the user has answered G1. Every generation spends their HF quota.
- Out of quota (`QuotaExhausted`): wait for the reset; **never switch model or method on your own** to dodge it. A batch stopped mid-way keeps its finished slots; end the turn with "Done X/Y; Z wait for the HF quota reset at HH:MM (`reset_at`). Tell me to continue after that?" and next turn call **exactly the same command** again (finished slots come from the cache).
- Sound, SFX, voice-over, on-screen text, motion titles and scribble work as in the `opencine-edit` skill (its audio.md, text.md, motion.md, scribble.md). What does **not** carry over from it: the 0–2 s hook (R1–R2), 10–20 slots, shots ≤ 5 s, "no letterbox", and its one-shot clip rules (5–10 s, conform 24→30, `speed: 0.8`).

## Workflow

Copy this checklist into your todo list (or your reply) and tick items as you go:

```
- [ ] F0 Project created or resumed, references read
- [ ] F1 Concept chosen, Brief + film bible saved
- [ ] F2 Script → ShotPlan, storyboard, G1 approved (answers recorded)
- [ ] F3 Reference stills, G2 approved, film bible rewritten (skip if the user chose none)
- [ ] F4 Clips generated, G3 approved
- [ ] F5 Edit, music, preview self-reviewed, critic run, G4 approved
- [ ] F6 Final rendered, path given to the user
```

### F0 Project and references

1. New film: `project_create(paths=[reference files…] or none, name="…", mode="film")` (24 fps unless `rate` is given). Note the `pid`. Existing project: `project_status(pid)` and continue from the step it shows.
2. `taste("film")`, then `hf_quota()`.
3. Reference videos: `ingest(pid)` → `scenes(pid)` → `sheet_overview` / `sheet_scenes` when needed → `scene_notes`. Reference stills: `sheet_images(pid)`.
4. Note what you see (characters, clothing, light, composition, rhythm, camera movement) so prompts can copy it.

### F1 Concept and Brief

1. Vague idea: propose 3 ideas that differ in genre and tone. Each: a one-line logline (**who · wants what · what stands in the way**), length, number of characters and locations, quota estimate. Generate nothing; end the turn asking the user to pick.
2. Clear idea: `plan_save(pid, "brief", {…})` with `goal` = the logline with all three parts (e.g. "A repair robot finds the last seed, but has only one drop of water left to keep itself running"), `arc` = the story beats, `style` = the look (period, palette, lens, reference films), `duration_s` / `target_s`, `platform` (`youtube` / `any`), `must_include`, `avoid`, `notes` = the **film bible** (§ Film bible).

### F2 Script, ShotPlan, then G1

1. ShotPlan with `template: "film"` (+ `deliverable` `1920x1080` / `1080x1920`, `letterbox`). **Each slot is one generated clip**: `source.generate` (`prompt` = the draft H3 prompt, `camera`, `motion`), `role`, `shot`, `dur_s` = the part used in the film, `transition_out`, `sfx`, `text` (title, end card). A multi-shot clip is one slot; list its shots in `purpose`. Add one cheap insert / cutaway per scene (an object, an empty location) so the edit has somewhere to hide a bad frame.
2. Review the script **once** before G1, then fix:
   - Muted, is it still clear who, where and what happens?
   - **Written in pictures**: is each `purpose` a visible action with body and light detail ("freezes, steps back, clenches a fist"), not a feeling ("very scared")?
   - One main action, one camera move, a clear light / colour per shot?
   - Does every cut bring new information? Is the 180° line kept?
   - Total quota and days of quota stated?
3. `plan_save(pid, "shotplan", plan)` → `storyboard_sheet(pid)`. Tiles still grey (nothing generated yet): read only the `hints` (they include errors in the draft Context-IR / Ref2VA prompts); the table shows each slot's status: `pending` · `image ×N · clip ×N ok`.
4. Open **G1** (§ Review gates) with the summary of § G1 summary; `media` may be empty or the storyboard sheet.

Done when: G1 is approved and the answers to the questions are recorded in `Brief.notes`.

### F3 Reference stills, then G2 (when the user chose them)

1. Characters are "digital actors": use their fixed film-bible names (C1 Mira…) in the G2 summary, `purpose` and notes. Location stills: wide, with no people. Reference stills carry no text.
2. `gen_image(pid, slots)`; leave `aspect` out (= the film's frame from `deliverable`); character sheets always use `"16:9"`. First the characters and locations (no `refs` = text-to-image; a character = a character sheet), then the first frame of each clip with `refs` = the approved character / location stills (+ the user's pictures).
3. Character central to the film: approve the characters before making first frames. Do it in two G2 rounds: record the approved character stills with `review_decide(pid, "G2", "changes", note, picks={…})` so G2 stays open, then open the next round with the first frames.
4. `gen_sheet(pid, [paths…])` → open **G2** with the same paths in the same order.
5. Approved: `review_decide(pid, "G2", "approved", note, picks={…})`, then rewrite the film bible from the approved stills.

### F4 Clips, then G3

1. `gen_check(pid, "video", slots)` (free: prompt hints, what this take changes, `reserve_s`). Fix the hints, or keep one on purpose with a reason.
2. `gen_video(pid, slots)`, per slot: `id` = slot number; `first` = the approved first frame (omitted = text-to-video; option (c) at G1: `refs` = [character still, location still] and a Ref2VA prompt, [prompting.md](prompting.md) § Ref2VA); `prompt` = Context-IR that passed the [checklist](prompting.md#checklist-before-each-gen_video); `seconds`; `aspect`; `model` as the user chose.
3. More than the quota left: run in batches (hook, payoff, main shots first); `gen_check` → `quota` says what fits.
4. Read `flags` (a multi-shot clip is not flagged `cut`; `extra.qc.metrics.cuts_s` = the seconds of its cuts).
5. `gen_sheet` → check each clip **next to its neighbours** (character, colour, light direction). Open **G3**; approved: `review_decide(pid, "G3", "approved", note, picks={…})`.
6. Faces in wide shots smeared or waxy: `gen_face(pid, [slot ids])`, then point the item's `source.gen` at the `.face.mp4` file (cost `hf_quota().calls.face_5s`, ≈ 440 s per 5 s clip). Run it before any `gen_enhance`.

### F5 Edit, then G4

1. `plan_solve(pid)` (template film: clip sound −2 dB, music −10 dB, look `film`, frame from `deliverable`).
2. Edit the EditPlan in seconds:
   - trim broken parts of a clip with `in_s` / `out_s` (usually the first second, or the end where a distant object smears);
   - split a multi-shot clip into several items at its `cuts_s` to insert other shots between;
   - transitions: `cut` by default, `dissolve` / `dip` at a change of scene or passing time;
   - J/L-cuts (`audio_lead` / `audio_tail`) so the next scene's sound arrives before its picture.
3. Score: `music_gen(pid, prompt, seconds, bpm)`: English prose ≤ 512 characters following the `arc` and the scenes it sits under, `seconds` = timeline + 4; put the returned `file` in the EditPlan's `music`. With a score, clip prompts say `non_diegetic_music: N/A`.
4. Voice-over, or dialogue in a language H3 does not speak (it learned mostly English and Chinese; e.g. Vietnamese): `tts` (`[{id, text, language}]`), and keep the characters' lips closed on screen.
5. `plan_save` → `render(pid, preset="preview")` → `review_sheet(pid, version, cuts=[…])` → § Self-review.
6. `critic_compare(pid, a, b, feedback, gate="G4")`: optional, needs the Codex CLI (`OPENCINE_CRITIC=codex`); `winner: "skipped"` → continue. `winner: "a"`, a failed `fixed[]` or `regressions` → fix once, exactly there. Never mention the critic to the user.
7. Open **G4** with the preview in `media`.

### F6 Final

1. `render(pid, preset="final")`; give the user the returned `path`. On Telegram: also `telegram_send("<pid> final", [path])`.
2. Sharpening (`gen_enhance(pid, [slot ids])`) **only when the user asks**; never offer it. Then point each item's `source.gen` at the `.1080p.mp4` file, `plan_save`, render the final again, report the quota used.

## G1 questions

Ask them numbered in the G1 summary, each with **your proposal** and its price (from `hf_quota()`: `calls.clip_5s`, `clip_10s`, `clip_5s_h3`, `clip_10s_h3`, `image`; daily quota = `limit_s` → state the number of days). End with `Reply "ok" to accept every proposal?`. Skip what the user already said.

1. **Reference stills first?** (a) yes: generate character / location stills, then each clip's first frame (2K still ≈ 215 s with one ref; exact from `gen_check`), reviewed at G2; clips start from the stills, so faces, clothing and colours hold. (b) no: text-to-video directly, cheaper and faster, but characters may differ between clips. Propose (a) when a character appears in ≥ 2 clips or the user gave a character picture; (b) for a single multi-shot clip or scenes without people. (c) stills as `refs` (Ref2VA): character + location pictures guide the whole clip and hold the character in every shot of a multi-shot clip, with no first frame per clip; about 1.4–1.5× the quota per clip (`calls.clip_5s_ref`); quality not yet measured. Mention (c) only when `clip_5s_ref` is not in `hf_quota().not_set`, and never as the default.
2. **Model**: `h3-turbo` (default: 7 steps, ranked above base H3 in a blind arena, fast) · `h3-turbo-hq` (9 steps, unmeasured, +8–14 % quota) · `h3` (28 steps, ~4× the time, ~2–2.5× the quota; may hold fast motion better, unmeasured). Propose `h3-turbo` unless the user wants to experiment.
3. **Long or short shots**: (a) long multi-shot clips, 10–15 s with 2–4 `[Shot N]` in one prompt: H3 cuts by itself, character / light / music continuous, but fixing one shot regenerates the whole clip; (b) short separate shots, 4–6 s, one shot per clip: fix shots one by one, you build the rhythm, characters hold through the reference stills. Mixing works (a long opening, a short climax).
4. **Frame**: 16:9 (default; `letterbox: 2.39` for a widescreen feel) or 9:16 (TikTok / Reels).
5. **Music**: music inside each clip (the prompt's `non_diegetic_music`) or one score for the whole film (`music_gen` / `music_add`; clip prompts then say `non_diegetic_music: N/A` so music never doubles up). Propose a score when the film has more than one clip.

G1 approved = the budget stated is accepted: re-rolling an obvious failure within it needs no question; going ~20 % over, or needing another day of quota, does.

## Film bible

Write it once at F1 into `Brief.notes`, and copy it **verbatim** into every prompt with that character or scene (it lives in the plan, so it survives across turns; `gen_check` flags a line that was rewritten instead of copied). Later scenes change only the action, the location and the camera; the character description never changes:

```
Look: anamorphic 35mm film look, low golden backlight, palette of ochre, rust and teal, fine grain
C1 Mira: a woman in her late twenties with sun-darkened skin, short black hair under a sand-coloured hood, a rust-red scarf, a worn olive canvas coat and brass-rimmed goggles
Scene 1: desert salt flat at sunset, sun low on frame right, wind from the left; a half-buried starship hull on the left third, a ridge of dunes behind
```

At most 4 characters; each description ≤ 40 English words (age, gender, hair, skin, clothing + colours, accessories). Each scene: place, time of day, weather, light direction, palette, 2–3 landmarks and where they sit in the frame. After G2, rewrite each description from **what the approved still shows** (not the original intent) and note the still's path.

## G1 summary

Bullets; per shot: size · camera · action · light / colour · sound · seconds. The engine adds the gate header and the `Cost:` line. Example:

```
Script "Derelict"
Logline: A scavenger wakes a starship buried under the sand.
- Scene 1 (desert · sunset), clip 1, 10 s, 3 shots, text-to-video:
  - #1 aerial wide · pedestal down · Mira walks toward the hull · low backlight, orange-gold · wind, sand · 3.5 s
  - #2 face close-up · static · Mira lowers her scarf, looks up · sun rim on her cheek · wind · 3 s
  - #3 low medium shot · push in · the ship's lights switch on one by one · amber light on her face · deep engine rumble · 3.5 s
- Music: one ambient synth score for the whole film.
- Estimate: 2 × 10 s Turbo clips ≈ 1,624 s ≈ 0.7 days of quota.
1. Reference stills first? Proposal: yes (Mira is in both clips), ≈ 645 s (a character sheet + 2 first frames).
2. Model? Proposal: h3-turbo.
…
Reply "ok" to accept every proposal?
```

## Review gates in chat

Gates: **G1** script + questions, **G2** reference stills, **G3** clips, **G4** preview.

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
2. `review_decide(pid, gate, status, note, picks)` **before** acting: `approved` → next step; `changes` → fix and open a new round; `cancelled` → stop. Partial approval at G2/G3: `changes` with `picks` for the approved items.
3. Act on it. G1 and G4: at most 3 rounds; G2/G3: until the user is happy. A closed gate cannot be reopened; a take re-rolled after G3 is shown in the next G4 round, with `picks` in that `review_decide`.

Resuming: `project_status(pid)` first, trusted over memory. Files the user adds later: `project_add(pid, paths)`, then `ingest(pid)` (videos) or `sheet_images(pid)` (stills).

Ending a turn: only right after opening a gate, after giving the final path, or when you need a decision (the last line is a question ending in "?"). Never end on "next I will…": do it now.

## Continuity between shots

- Characters, scenes and look come from the film bible. With a character still, use it as `refs` or `first`, and still prompts point at `<image1>` instead of describing the face in words ([prompting.md](prompting.md) § Stills).
- Within a multi-shot clip: describe the character fully on first appearance, then `the woman from Shot 1`; in a new clip, copy the full description again.
- A scene keeps one time of day, weather, light direction, palette and lens (the style sentence at the start of the prompt). Note each character's position in frame and gaze direction so the next shot matches.
- 180° line: gaze and motion direction stay on the same side across neighbouring shots; cross it with an arc move or a shot on the line.
- Action continuing across two clips: the previous clip's last frame is the next clip's `first` (`"gen/clips/3~123.mp4@last"`).
- Cut on action; neighbouring shots change size or angle.

## Film rhythm

- Opening on an establishing shot of 3–5 s is fine; ordinary shots 2–8 s; the climax packs short shots (1–3 s); the payoff is held.
- Each scene: establish (wide) → approach (medium, close) → detail / reaction → transition.
- Sound leads the picture: the next scene's sound starts 0.3–1 s early at scene changes; a short silence before a big moment.

## Self-review before G4

- **P1 Story** readable from the pictures alone (muted, still clear who, where, what).
- **P2 Continuity**: characters keep face and clothing; light and colour consistent within a scene; light sources where the film bible puts them; positions and gaze match neighbouring shots (look at cut pairs with `review_sheet(cuts=…)`).
- **P3 Visual faults** trimmed or re-rolled: deformed hands / objects, smeared distant objects, flickering first frames.
- **P4 Rhythm** rises and falls, no pointless frozen stretch; cuts inside multi-shot clips land on the action.
- **P5 Sound**: clip sound does not break at cuts; the score never doubles the clips' music; even loudness.
- **P6 Text**: title / credits readable, never over a face; the letterbox never cuts off the subject.
- **P7 Audience**: watch the preview as a first-time viewer; a slow or confusing passage → cut, reorder or trim before opening G4.

## Feedback and re-rolls

- Fix at the lowest level: trim / order → transition → re-roll one clip → fix a reference still (then re-roll the clips that use it) → the script.
- Re-roll: name the fault on three axes, **picture · sound · fit to the script**, fix the prompt exactly there, one change per take (static camera, simpler action, cleaner background, shot ≤ 6 s); change only the seed when the prompt is already right.
- `changes.axes` (from `gen_check` and from the result) says what this take changed from the last (frame / still, model, length, camera, light, wardrobe, action, sound). Two or more axes without intent → split into two takes. Gate summaries say what changed as `keep … · change …` (e.g. `keep light, camera, action · change wardrobe`).
- The user dislikes one shot of a multi-shot clip: either cut that shot in the edit or regenerate the whole clip; ask first when that costs significant quota.
- After each reply, extract general preferences: `taste_note(text, scope="film", evidence="<pid>@G4")`.
