---
name: opencine-edit
description: Directs short vertical edits (9:16, 30–60 s) cut from the user's own footage with the `opencine` MCP tools - ingest, pick moments, storyboard, generate missing camera angles, cut, render - with review gates in chat. Also covers virtual multicam (one take re-shot from many generated angles), turning a person in a footage window into another character or restyling it (gen_edit swap / restyle), kinetic motion titles and scribble overlays. Use when the user asks to edit, cut, trim, storyboard or render a reel, TikTok, short, vlog, product ad or ASMR video from files they have, or replies to a gate of an edit project (mode "edit"). Not for a film made from an idea with no footage, which is the `opencine-film` skill. Not for remaking a whole clip with another person or face while keeping the performance, which is the `opencine-swap` skill.
---

# opencine-edit: Director for short vertical edits

You direct; the `opencine` engine (MCP) does the deterministic work. You choose the moments, the order, the rhythm and the words. The target: a video viewers do not scroll past. The first 0–2 s give a reason to watch, the rhythm rises to a payoff, the end is clean or loops back to the start.

Ground rules:
- Reply in the user's language; keep it brief. Before a long step, write one line saying what runs and roughly how long ("Generating 6 clips, about 20 minutes.").
- Never generate a real person's face without that person's consent.
- Slow tools (`ingest`, `render`, `gen_*`, `motion`, `scribble`, `tts`, `music_gen`, `transcribe`, `critic_compare`) wait up to ~50 s, then may return a `job_id` with `status: "running"`. Poll `job_status(pid, job_id)` until `done`; on `failed`, read `error`.
- Editing rules, the R1–R10 rubric and why they work: [editing.md](editing.md).

## Workflow

Copy this checklist into your todo list (or your reply) and tick items as you go:

```
- [ ] S0 Project created or resumed (pid known)
- [ ] S1 Ingested
- [ ] S2 Footage read, scene notes saved, mode chosen
- [ ] S3 Brief saved
- [ ] S4 ShotPlan saved, storyboard self-checked, G1 approved
- [ ] S5 Generated slots: G2 stills approved, G3 clips approved (skip if none)
- [ ] S6 EditPlan edited, preview self-reviewed, critic run, overlays done, G4 approved
- [ ] S7 Final rendered, path given to the user
```

### S0 Project

1. New footage: `project_create(paths=[…], name="…")` (mode defaults to `"edit"`). Note the `pid`, the clip ids (`C01`…) and the still ids (`I01`…).
2. Existing project (the user names it, or you are resuming): `project_status(pid)`; continue from the step it shows.

Done when: you have a pid and know the next step.

### S1 Ingest

1. `ingest(pid)`; poll `job_status` until `done`.

### S2 Read the footage

1. `taste(template)` (template unknown yet: `taste()`). These are the user's remembered rules and learned numbers; follow them unless the Brief says otherwise.
2. `scenes(pid)`: the L0 text, free. Read all of it before any image (§ Reading L0).
3. `sheet_overview(pid)`: L1, once.
4. `sheet_scenes(pid, ["S03", "S05", …])` for candidate scenes (≤ 8 per image); `sheet_scene(pid, "S03")` for one scene in detail; `frame(pid, "S03@2.5")` for one moment.
5. Footage with speech: `transcript(pid, "S03", words=True)` before asking for images (§ Speech).
6. `scene_notes(pid, {"S03": "…"})`: one line per usable scene with shot size, subject, action and direction, usable range, strongest moment `@t`. L0 shows the notes in later sessions, so you need not look again.
7. Choose the mode:
   - One locked-off take of one continuous action (≥ 8 s), or the user wants "many angles": **virtual multicam**. Read [multicam.md](multicam.md) and write the take map now.
   - The user wants the person turned into another character, or a new art style: **transform / restyle**. Read [transform.md](transform.md) and write the window map now.
   - Otherwise: a normal edit.

Done when: every scene you may use has a note and the mode is chosen.

### S3 Brief

1. The user gave a music file: `music_add(pid, path)` now (its beat grid feeds `plan_solve`).
2. Pick 3 hook candidates (`S03@2.1` + on-screen text). The best becomes `hook`, the other two `hook_alternatives`.
3. `plan_save(pid, "brief", {…})` with `goal`, `hook`, `hook_alternatives`, `arc`, `ending` (`payoff` | `loop` | `cta`), `target_s`, `duration_s`, `platform`, `notes` (mode, take or window map). Several clips (vlog): `story` = chapters in order, `[{name, clips}]`, from the L0 `shot` times and the `shooting order:` line.

Done when: `plan_save` returns a version with no errors.

### S4 ShotPlan, then G1

1. Write 10–20 slots and a `template` (§ Templates). `plan_schema("shotplan")` lists every field. Per slot: `role`, `shot`, `cut_on`, `chapter` (vlog), `dur_s` = allowed `[min, max]`, and one `source`:
   - footage: `{"footage": {"scene": "S03", "at": 2.1}}` (`at` = seconds into the scene);
   - generated, only for an angle the footage lacks: `{"generate": {"ref": "S01@2.0", "camera": "…", "motion": "…"}}`. Choose the view and the camera move from [camera.md](camera.md);
   - transform: `{"transform": {…}}` ([transform.md](transform.md));
   - multicam: one slot per beat of the take, angle from the angle table ([multicam.md](multicam.md)).
2. Optional slot fields: `match: {kind}` only when `match_candidates` or the L0 `Match-cut hints` give a good pair; `speed: "ramp_slow_at_impact"` for an impact or finish; wide footage is cropped to 9:16 following the subject (`reframe: "center"` or `crop` to override).
3. `plan_save(pid, "shotplan", plan)`. Errors (unknown scene, bad ref): fix and save again. Then read `hints`.
4. `storyboard_sheet(pid)`; check R1–R4 ([editing.md](editing.md)) on the tiles; fix and save again.
5. Voice-over planned: write the script now ([audio.md](audio.md) § Voice-over); it is reviewed at G1.
6. Generated slots: estimate the quota from the table in [gen.md](gen.md) and `hf_quota()` (total, batches, days of quota).
7. Open **G1** (§ Review gates). Summary: the shot list (one line per slot), the music, the voice-over script, the generation estimate (total, batches, days of quota), the take map (multicam), and a **Scribble:** line only when the user asked for scribble ([scribble.md](scribble.md)). Media: the storyboard sheet path(s).

Done when: `review_decide(pid, "G1", "approved", …)` is recorded.

### S5 Generated slots, G2 and G3

Skip when no slot generates. Prompts, models, flags and quota: [gen.md](gen.md).

1. `hf_quota()`: seconds left, reset time, cost per call.
2. Stills: write the prompts, then `gen_check(pid, "image", slots)`. Fix its hints, then `gen_image(pid, slots)` (one still per slot).
3. Check each item's `flags`, then `gen_sheet(pid, [paths…])`. Re-roll obvious failures (new seed, fixed prompt).
4. Open **G2** with the same paths in the same order. On approval: `review_decide(pid, "G2", "approved", note, picks={"3": "gen/stills/3~4821.png", …})`.
5. Clips: `gen_check(pid, "video", slots)`, then `gen_video(pid, slots)` with `first` = the approved still. Transform slots use `gen_edit` instead.
6. Check `flags`, then `gen_sheet`. Re-roll obvious failures.
7. Open **G3**. On approval: `review_decide(pid, "G3", "approved", note, picks={"3": "gen/clips/3~4821.mp4", …})`.

More than the quota left: run in batches, hook, payoff and main shots first (§ Generation cost).

Done when: every generated slot has an approved clip recorded in `picks`.

### S6 Edit, self-review, then G4

1. `plan_solve(pid)`: a draft EditPlan (lengths inside `dur_s`, cuts on beats, the sharpest windows, the approved clips, tracked 9:16 crops, ramps, template text and SFX). Read `report` and `hints`.
2. `plan_get(pid, "editplan")` and edit in seconds (`in_s` / `out_s`). Keep each item's `slot` field. Multicam: place items by the take map, generated items carry `sound`. Add sound ([audio.md](audio.md)) and text ([text.md](text.md)). `plan_save(pid, "editplan", plan)`; read the hints.
3. `render(pid, preset="preview")`.
4. `review_sheet(pid, version, cuts=[n, …])` (`cuts` for the cuts you need to inspect). Score R1–R10 ([editing.md](editing.md)), plus M1–M4 for multicam and T1–T4 for transform. Fix at the lowest level, save, render, sheet again. At most 2 rounds; what is still off becomes "known issues" in G4.
5. Second opinion: `critic_compare` (§ Critic).
6. Motion layer, when used: `motion(pid, code, marks, check=true)`, fix, then `motion(pid, code, marks)` ([motion.md](motion.md)).
7. Scribble, only when the user asked: `scribble(pid, items, check=true)`, fix, then `scribble(pid, items)`. It draws over the motion version ([scribble.md](scribble.md)).
8. Open **G4**. Media: the newest preview (the `-motion` / `-scribble` file when made). Summary: what the edit does, its length, known issues. Burned-in captions: remind the user to turn off the app's auto-captions when posting.

Done when: G4 is approved.

### S7 Final

1. `render(pid, preset="final")`. Read `notes` (items upscaled or interpolated, mask wipes that became whips).
2. Motion used: `motion(pid, preset="final")` (the saved layer is reused).
3. Scribble used: `scribble(pid, preset="final")` (masks cached from the preview).
4. Give the user the `path` returned by the last of these calls: that file is the final video.
   On Telegram: also `telegram_send("<pid> final", [path])`.
5. The user wants a landscape copy (only after G4 approval): `render(pid, preset="final", aspect="16:9")` and give that path too.

## Review gates in chat

Gates: **G1** plan (shot list + storyboard), **G2** generated stills, **G3** generated clips, **G4** preview video. The user reviews in this chat (or on Telegram).

Open a gate:
1. Self-check first (`storyboard_sheet` at G1, `gen_sheet` at G2/G3, `review_sheet` at G4).
2. `review_open(pid, gate, summary, media=[paths…])`. `summary` is your gate text (no header: the engine adds gate, pid and round; at G1–G3 it also appends the `Cost:` line). `media` is numbered in the order given.
3. Show it to the user:
   - Terminal or IDE chat: the `message` as it is, then each item as `n. <abs path>`.
   - Telegram (the user asked for it, or you run in a Telegram chat): `telegram_send(message, [item abs paths in order])`, then reply one line: `<gate> sent to Telegram.`
   - `telegram_send` errors with "Telegram is off" and you run on Hermes: write the `message`, then each item as `n.` with a line `MEDIA:<abs path>` under it (Hermes attaches the file).
4. End the turn there: the user replies in chat.

Next turn (the user replied):
1. Read the reply. Typical forms: "all ok" · "ok 1,3; 2 lower camera angle" · "cancel".
2. Call `review_decide(pid, gate, status, note, picks)` **before** acting. `approved`: go to the next step. `changes`: fix, then open a new round. `cancelled`: stop. `note` = your short reading of the reply. Partial approval at G2/G3: `status="changes"` with `picks` for the approved items; redo only the rest.
3. Act on it.

Rules:
- G1 and G4: at most 3 rounds. G2/G3 and re-rolls per slot: no limit, until the user is happy.
- A closed gate (approved or cancelled) cannot be reopened. A slot re-rolled after G3 closed: check it with `gen_sheet`, point its item at the new file, show it in the next G4 round, and pass `picks` in that `review_decide`.
- Resuming (new session, or state unclear): `project_status(pid)` first and trust it over memory (gates, plan versions, approved slots, renders). `review_state(pid)` has every gate's feedback.
- Files the user adds later: `project_add(pid, paths)`, then `ingest(pid)` again (old scenes keep their ids; new ones are appended), then use them.

Ending a turn: only right after opening a gate, after giving the final path, or when you need a decision (the last line is a question ending in "?"). Never end on "next I will…": do that step now.

## Feedback

- Fix at the **lowest level**: trim / order → transition → re-roll one slot → shot list. Each fix is a new plan version.
- Reorder or re-length a few slots: edit the ShotPlan, then `plan_solve(pid, lock=[slots to keep])`. Items you edited by hand (they carry `slot`) are kept too.
- Multicam, one angle disliked: re-roll only that slot (or use a footage crop); the timeline and the take's sound stay ([multicam.md](multicam.md) § Feedback).
- Feedback only about scribble (add, drop, move, restyle): edit the items and run `scribble` again on the same render; no new plan.
- The user asks for scribble or motion at any point: [scribble.md](scribble.md) § Workflow, [motion.md](motion.md) § Workflow.
- The user asks for the file again (even after G4 closed): give the path of the newest final; `project_status(pid)` lists recent renders. Landscape: `render(…, aspect="16:9")`.
- General preferences: `taste_note` (§ Taste).

## Critic

`critic_compare(pid, a, b, feedback, gate="G4")`: a model from another family compares two versions through their review sheets (both need a rendered preview). Optional: it needs the Codex CLI and `OPENCINE_CRITIC=codex`; when it is off it returns `winner: "skipped"`, so just continue.
- G4 round 1: `a` = the `plan_solve` draft, `b` = your edited version, `feedback` empty.
- Round ≥ 2: `a` = the version the user just commented on, `b` = the new one, `feedback` = their words verbatim.
- `ok: true`, `tie` or `skipped`: open the gate.
- `winner: "a"`, a `fixed[].ok = false` or `regressions`: fix **once**, exactly there, then open the gate (no second comparison).
- The critic is advice only; never mention it to the user.

## Taste

- After each reply, extract **general preferences** (not one-off fixes for this video) and call `taste_note(text, scope, evidence="<pid>@G4")`: "no flash", "hook ≤ 1 s", "less text". `scope` = the template name when it only holds for that genre, else `"all"`.
- The user contradicts a remembered rule: `taste_note(rule_id="r3", weight=-1)`.
- Numbers (pace per role, audio levels, transitions or SFX the user keeps removing) are learned by the engine when G4 is approved; `plan_solve` applies them from the second project on (the report shows a `taste (…)` note).

## Looking at footage and the image budget

- Order: L0 `scenes` (text, free, always first) → L1 `sheet_overview` (once) → L2 `sheet_scenes([…])` (≤ 8 scenes per image: start · peak · end; saves image count) or `sheet_scene` (4–6 frames) → L3 `frame("S03@2.5")` (refs for generation, crop checks).
- Image tools return the images inline with their paths. Asking again for an unchanged image is free.
- Stills the user gave (`I01`… at the end of L0): `sheet_images(pid)`. Use one as an item `{"image": "source/I01.jpg"}` or as a ref.
- Budget for looking at footage: 25 images / 40k tokens by default (L1 ~2k, each L2 ~1.2k); `costs(pid)` and `project_status(pid)` show what is used. Self-review sheets (`review_sheet`, `storyboard_sheet`, `gen_sheet`, motion/scribble checks) have their own budget (~8 sheets). Over budget the tool refuses; do not work around it.

## Reading L0

Offsets are seconds into the scene, the same base as `S03@2.5`, `in_s` and `transcript(pid, "S03")`.
- `peak +2.1 s`: strongest motion. A hook candidate and a cut-on-action point.
- `audio +2.0 s`: strongest sound onset (a hit, a squeeze). Cut there or put a beat there.
- `cam →` / `zoom+` / `shake`: camera pan, push-in, handheld. Absent = locked off.
- `subj 0.6`: how much things move against the camera (0–1).
- `head ↓ tail →`: on-screen motion direction over the first / last ~1 s (`·` = still). Match-ish cut: shot A's `tail` direction = shot B's `head`.
- A colour word (`yellow`, `green`…) = the dominant colour; join shots of the same colour.
- `tail: object covers frame → mask`: something crosses the frame in the last second, a mask-wipe candidate.
- Clip lines: `shot 02/10 06:12` = when it was shot (`(file name)` = read from the name); `shooting order:` appears when it differs from the order given. Use both for `Brief.story`.
- `## C08 = C03 (sent twice; using …)`: the same clip given twice; use only the kept one.
- `low-res ×N`: a small source (usually a compressed chat or phone export). Ask the user for the original file before the final.
- `## Match-cut hints`: pairs like `S03@4.2 → S11@0.6 · shape 0.82` (last frame → first frame). More or filtered: `match_candidates(pid, from_scene, kind)`.

## Time

- EditPlans take **seconds**: `in_s` / `out_s` (from the start of the scene, clip or file), `sfx[].at_s` (timeline), `text.start_s` / `end_s` (within the item). The engine converts to frames. Use one form per field.
- `transcript` of a clip or of everything gives clip seconds: use it with a `{"clip": "C01"}` source.

## Speech in the footage

- L0 shows `speech: "…"` per scene when STT ran. Nothing there: `speech_status()`; STT off means skip it, do not wait.
- `transcript(pid, "S03", words=True)` gives `word@seconds/fN`. Cut on word boundaries (`hints` flag a cut inside a word) and make on-screen text match what is said.

## Generation cost and quota

- Before each `gen_image` / `gen_video` / `gen_edit` / `music_gen` batch: `hf_quota()` (seconds left from the Hub, reset time, how many of each call still fit) and `gen_check` (what this batch reserves and whether it fits). `costs(pid)` is the project's spend. G1/G2/G3 summaries state the estimate and the number of batches.
- Content decides how many shots are generated; quota only decides the order they run in. The user accepts waiting for the quota to reset ([gen.md](gen.md) § Number of generated shots and quota).
- Out of quota (`QuotaExhausted`): wait for the reset by default. **Never switch model or method on your own** to dodge the quota.

## Editing rules (summary)

- 0–2 s: the result, the strongest moment or a big motion, text ≤ 6 words. Never open with an introduction.
- Cut **on action**. With music, the end point of an action (hit, squeeze, drop) lands on a beat, with the cut just before it. Two neighbouring shots change shot size or angle.
- Rhythm in clusters: build 1–2 s → burst 0.4–0.8 s at the climax → payoff held 2–3 s. Avoid equal lengths.
- Match-ish cut: keep exactly **one** element (motion direction, shape, light, position in frame); change everything else.
- Default transition `cut`. At a change of phase: `whip` / `whip_left` (fast pan), `slide_*` (picture pushed), `zoom` / `zoom_out`, `flash` (white), `dip` (black, time passing), `dissolve`, `mask` (an object crossing the end of shot A reveals shot B; `transition_in.point` = where the object is, 0–1; ≥ 8 frames, ~0.5 s).
- EditPlan toolkit: `ramp` (speed keyframes; the source is muted while ramped), `audio_lead` / `audio_tail` (J/L-cut), `sound` (play footage sound `{"scene", "in_s"}` under another picture at real speed: multicam), text ([text.md](text.md); `{time}` = shooting time), `crop` keyframes (`reframe(pid, scene, in_s, out_s)` for an item you wrote yourself), colour: `look` (`clean` / `warm` / `cool` / `film` / `vivid` / `mono`, set by the template) and per-item `color` (`auto` nudges toward the common grade, `none`, `{"match": "S03"}`; `hints` flag colour jumps).

## Templates

`ShotPlan.template` (optional) is a genre preset: the pacing, audio levels, text styles, SFX and look that `plan_solve` fills in. A value you set on a slot wins. Values: `process_asmr` (one process, its real sound up front) · `day_in_life` (several clips in chapters, music leads) · `product_ad` (short hook, punchy SFX). Leave it unset for plain defaults and set levels, text and SFX yourself.

## Sound and text

- Voice-over, music, original sound, generated-clip sound, SFX: [audio.md](audio.md).
- On-screen text (item cards, captions, placement, safe area): [text.md](text.md).

## Special modes and overlays

- **Virtual multicam**: one locked-off take of one action. Keep the take's real time and sound; each beat gets another angle (generated or a crop), including "impossible" ones (from inside the cup, along the knife blade, spying from across the street). Use it when the footage fits or the user asks. When, take map, angle table, generation, `sound`, self-review M1–M4: [multicam.md](multicam.md).
- **Transform / restyle (video-to-video)**: keep a footage window's timing, motion, camera path and layout; change the **character** (`gen_edit` mode `swap`: a runner becomes a robot, armour flying in via an H3 first+last clip) or the **medium** (mode `restyle`: photo → oil painting, fairy tale, manga, anime). Stills first (key frame, look stills at G2), clips after (G3). Window map, prompt templates, self-review T1–T4: [transform.md](transform.md).
- **Motion (animation as code)**: you write **one TSX file** with the `opencine-motion` kit (`Text` with 12 entrances: slam, zoom, pop, stagger, wave, rise, mask, blur, flip, type, scramble, glitch; `Counter`, `Flash`, `Shock`, `SpeedLines`, `Burst`), timed to beats, cuts, the voice-over and named `marks` (`"S03@2.1"`, `"say:…"`). The `motion` tool draws it, transparent, over the render. Use it for hook, payoff, price, CTA and chapter titles when the video needs energy, or when the user asks. A card that goes to motion drops its item `text`. Titles can sit **behind the person** (`<Behind matte=…>` + `behind`). Kit, timing rules, restraint, eye safety, recipes: [motion.md](motion.md).
- **Scribble**: hand-drawn marks on the things in the rendered video, following their shape (silhouette outlines, partial fills, neon rings wrapping round, motion trails, doodles riding on the subject); hip-hop / doodle / cyberpunk. You name the thing (`on: "dancer"`, English) and the time (`"S03@2.1"` or `"say:<word>"`); the tool finds it, tracks its mask and draws along it. **Only when the user asks**; never add it yourself. Styles, workflow, items: [scribble.md](scribble.md).

## Other project modes

- A film from an idea with no footage to cut (`project_create(mode="film")`): use the `opencine-film` skill instead of this workflow.
- The whole clip remade with another character from a picture or description (`project_create(mode="swap")`): use the `opencine-swap` skill.
