# Virtual multicam: one take, every shot size, every angle

The footage has only **one locked-off camera filming one continuous action**. The edit keeps **the take's real time and real sound** and changes only **the picture**: each beat of the action is "re-shot" from another angle, either generated (S5) or a crop of the footage itself. The viewer feels 8–15 cameras, including cameras in places no camera fits (inside the cup, along the knife blade, behind a window across the street).

## When to use

Recognise it at S2, before the Brief:
- L0 has 1–2 clips, one scene ≥ 8 s, a still camera (no `cam` field), a person or hands doing one thing continuously (brewing, cooking, cutting, pouring, assembling, make-up), or talking while doing it.
- The footage has only 1–2 shot sizes: a normal edit would repeat one framing (R4).
- The user says "many angles", "keep changing angles", "multicam", "shoot it from another angle", "like several cameras", "weird angles", or "make it livelier" when the footage has one angle.

Skip it, or use it only in part:
- Footage already has many angles and clips (vlog): edit normally, add a few generated slots ([camera.md](camera.md)).
- The main content is a face talking continuously: change size with `crop` only (generated shots do not lip-sync); generate only for hands or object beats.
- The user wants it cheap or fast, or quota is low: a crop version (punch-ins at several sizes) is still multicam; keep generated angles for hook, peak and payoff.

Write in the Brief: `notes: "multicam: S01 0.0–24.5 s · 10 angles (7 generated, 3 footage)"`, `arc` = the beats of the take. The G1 summary states the mode, the take map and the cost.

## Take map (S2 → S4)

1. Read L0 (`peak`, `audio`, `speech`), `transcript(words=True)`, `sheet_scene` and a few `frame`s. Split the take into **beats** of 0.8–2.5 s; each beat is one action with a clear start and end (pick up, pour in, stir, set down, say a sentence).
2. Cut points: at the start of a movement or just before a contact (cut on action), on a word boundary.
3. Take longer than `target_s`: drop dead parts (waiting, repeats, "um"s). Each drop sits exactly on an angle-change cut, so the viewer sees no gap. Never drop mid-sentence.
4. Save the map in the scene's `scene_notes`, then put it in G1:

```text
0.0–1.4  says the opening line, hand holds the cup   · footage MS (original)
1.4–3.0  pours boiling water into the filter          · generated: high-angle tele through foreground leaves
3.0–4.1  coffee drips                                 · generated: macro level with the cup rim
4.1–5.9  stirs, looks into the cup                    · generated: from inside the cup looking up (camera.md 13)
5.9–7.0  "smells good?"                               · footage crop CU face
```

## Angle table

Each beat takes one cell. Neighbouring beats differ by **≥ 2 shot sizes** or by **angle family**.

| Family | Angles | Good for |
|---|---|---|
| Wide | WS establishing; EWS spying through a window, car, fence (camera.md 15) | openings, rhythm changes, a rest for the eye, endings |
| Eye level | MS (usually the original footage), 3/4 (8), over-the-shoulder (6) | anchoring the space, spoken lines |
| Close | CU face / hands, ECU, macro (5) | emotion, detail, ASMR contact points |
| High | high angle (3), top-down (4) | process overview, table layout |
| Low | low hero (1), worm's-eye (2) | strong subject, payoff, an object raised against the sky |
| Immersive | POV hands (7); object POV: from inside the cup / pot looking out (13), along a tool (14) | surprise, hook |
| Depth | blurred foreground (10), big object in the foreground (17), from behind (9) | mystery, rhythm changes |
| Following the object | object in mid-air (16), level with the pour | motion peaks |

Numbers in brackets are the still recipes in [camera.md](camera.md).

**Ordering rules:**
- One of the first 3 shots lets the viewer understand the space (the original MS/WS or an establishing shot). The hook may be the strangest angle.
- "Impossible" angles (13–15): 2–4 per 20 s, on memorable beats (hook, peak, payoff), never adjacent.
- The camera follows what moves: a flying object is tracked, a pour is seen level with the stream, a lift is seen from worm's-eye.
- **Speech**: the mouth is clearly visible only in the original footage (close crops are fine). A generated angle during speech must not show the mouth: from behind, far away, hands, the object, object POV.
- 180° line: keep the camera side. To cross it, put a top-down or POV shot in between.
- Rhythm: 1–2.5 s per angle; 0.6–1 s at the climax; payoff held ≥ 1.5 s (low hero or a wide pull-out).
- The world stays the same across angles: light from the same side, objects in the same places, each object in its state for that beat (the cup already full or not).

## Generation (S4–S5)

- **Footage slot**: `footage.at` = start of the beat; change size with `crop`.
- **Generated slot**: `generate.ref = "S01@<beat start>"` (the pose at the cut); `camera` = the angle-table cell + lens; `motion` = **exactly the take's action during that beat**, in order, plus the camera move (usually static or a small push in, like a real camera on a tripod).
- **Stills**: `gen_image(pid, slots)` with `refs=["S01@<beat start>"]`, the re-photograph recipe from camera.md. One take per angle.
- **Clips**: `gen_video(pid, slots)` with `first` = the approved still, the prompt = the beat's action at real pace, `seconds` 5 (a beat ≤ 2.5 s uses only part of the clip). When the end of the beat must match the pose for the next cut (object set down, hand gone), use first+last: still B = the same angle from `S01@<beat end>`, `refs` = [the beat-end frame, still A] (camera.md § Match cut from a generated still). That costs one more still.
- **Cost**: each generated angle is one 2K still (~215 s) + one 5 s clip (~364 s with a Context-IR prompt) ≈ 580 s reserved. 8 angles ≈ 4,600 s ≈ 2 days of one account's quota (2,400 s/day). Exact numbers: `gen_check`. Short on quota: generate hook, peak and payoff first; the other beats use footage crops.

## EditPlan (S6)

`plan_solve` gives the draft (look, music, crops); then re-place each item by the take map. You can also write the EditPlan directly:
- Each item is one beat, exactly the beat's length, `transition_in` `cut`. Overlapping transitions shift the take's time.
- **Footage item**: `in_s` = beat start, `out_s` = beat end; its own sound is the take's sound.
- **Generated item**: `"sound": {"scene": "S01", "in_s": <beat start>}`. The take's sound plays under the generated picture at real speed; no `gain` needed. The clip's `in_s` ≥ 0.3 s, at the point where the clip's action matches the take's action at the cut; `out_s` = `in_s` + beat length × `speed`. If the clip's action is faster or slower than the take, set `speed` 0.8–1.25 so the **contact point** on screen lands on the take's contact sound (`audio +x s` in L0).
- When each item's sound starts where the previous one stopped, the take runs continuously (the engine joins them seamlessly). The hint `the take's sound skips/repeats` flags offsets under 0.5 s. Deliberate shortenings jump ≥ 0.5 s, on a word boundary.
- Levels: the take's sound leads (0 dB); music low (−14…−18 dB) or none; foley only where the take lacks sound.
- Colour: generated items get `color: {"match": "S01"}`.
- The `review_sheet` timeline prints `audio S01 4.20 s` after each item with `sound`: scan that column to confirm the sound is continuous.

```jsonc
{"items": [
  {"slot": 1, "source": {"scene": "S01"}, "in_s": 0, "out_s": 1.4},
  {"slot": 2, "source": {"gen": "gen/clips/2~7710.mp4"}, "in_s": 0.4, "out_s": 2.0,
   "sound": {"scene": "S01", "in_s": 1.4}, "color": {"match": "S01"}},
  {"slot": 3, "source": {"gen": "gen/clips/3~1203.mp4"}, "in_s": 0.5, "out_s": 1.82, "speed": 1.2,
   "sound": {"scene": "S01", "in_s": 3.0}, "color": {"match": "S01"}},
  // slot 4: inside-the-cup angle, "sound" from 4.1
  {"slot": 5, "source": {"scene": "S01"}, "in_s": 5.9, "out_s": 7.0,
   "crop": {"x": 0.3, "y": 0.05, "w": 0.45, "h": 0.45}}
]}
```

## Self-review (on top of R1–R10)

- **M1 Angles**: every cut changes ≥ 2 shot sizes or the angle family; no two near-identical shots.
- **M2 Continuity**: objects in the right hand and the right state; motion direction consistent across cuts (`review_sheet(cuts=[…])`).
- **M3 Sound sync**: on-screen contact points match the take's sound (±2 frames); generated angles never show a speaking mouth.
- **M4 World**: light, colour, clothing and objects match across angles. A broken generated angle (warped hand, deformed object): re-roll it or replace it with a footage crop.

## Feedback

- "Change the angle of shot 5", "this angle is ugly": re-roll only that slot, timeline unchanged; or replace it with a crop.
- "More strange angles", "more angles": split a long beat in two and use unused cells of the angle table.
- "Generate less", "cheaper": keep generated angles for hook, peak and payoff; crops for the rest.
- General preferences ("love the inside-the-cup angle", "drop the spying angle"): `taste_note`.
