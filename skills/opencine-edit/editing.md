# Editing rules, self-review rubric, match-ish cuts

Contents: Rubric R1–R10 · How to self-review · Why these rules work · Match-ish cuts.

## Rubric R1–R10

Score on `review_sheet` (images + timeline table + `hints`); for a ShotPlan, score R1–R4 on `storyboard_sheet`. Each item: pass / fail + the item number.

| # | Question | Typical fix |
|---|---|---|
| R1 Hook | Does frame 0 (`hook 0s`) show a clear subject / result / motion? Text (if any) ≤ 6 words and readable? | Put the strongest moment first; drop the intro |
| R2 Promise | First cut by ~2 s; by 3 s does the viewer know what the video is about? | Shorten shot 1, add hook text |
| R3 Rhythm | Lengths vary in clusters, never 3 equal shots in a row; a burst at the climax; payoff held ≥ 1.5 s? | Change `out_s`, merge / split items |
| R4 Continuity | No unintended jump cut (same scene, same framing, < 1 s apart); neighbours change shot size or angle; motion direction consistent; subject inside the crop (hint)? | Change crop / size (`reframe`), insert a cutaway, or a deliberate punch-in |
| R5 Cut points | Cuts inside an action (not on a dead frame); with music, action end points on the beat; no cut inside a word? | Move `in_s` / `out_s` a few frames; speech across a cut → `audio_lead` / `audio_tail` (J/L-cut) |
| R6 Match-ish | At each planned match cut, does the pair `#n-1 out` / `#n in` clearly share **one** element? | `review_sheet(cuts=[n])`, adjust in / out or crop |
| R7 Text | One idea per card (≤ 8 words), on screen ≥ max(1 s, words ÷ 3) s, never over a face / subject / the action (hint `hides the …`; look at frames with text); captions follow the speech and never share a band with item text; emphasis ≤ 15 % of words? | `position` `auto`, shorter text, move to a shot with room ([text.md](text.md)), hold longer; `captions.breaks` / `emphasis` |
| R8 Ending | Payoff clear and long enough; last frame leads back to the first (loop) or ends cleanly; total length matches the Brief; no black or frozen frames? | Add a loop shot, trim |
| R9 Colour | Shots of one scene share one grade (no white / yellow or bright / dark jumps); generated clips blend with the footage? | `color: {match: "S03"}` on the odd item; a shared `look`; hints "colour jumps" |
| R10 Sound | Music runs throughout and ducks under speech; voice-over continuous (no `voice-over … stops for` hint); each main contact action has sound (original, generated clip or foley); no SFX file repeated back to back? | Add music; close voice gaps; turn on a generated clip's `gain` or add foley from `sfx_list`; vary accent files |

## How to self-review (S6, before G4)

1. `plan_save` → fix the `hints` that are right. Hints never block; you may keep one on purpose (say why in the item's `note`).
2. `render(preset="preview")` → `review_sheet(pid, version, cuts=[…])`; score R1–R10 on the returned images.
3. Any fail → fix at the lowest level → new plan version → render → `review_sheet`. At most **2 rounds**; what is left goes into the G4 summary as "known issues: …". Never score a new version on old images.
4. Second opinion: `critic_compare` (SKILL.md § Critic). A judge from another model family sees only images, table and feedback. Self-correction without new evidence tends to make things worse ([Huang et al. 2024](https://arxiv.org/abs/2310.01798)), and pairwise comparison is more reliable than absolute scores.

## Why these rules work

- **The opening decides**: on Snapchat, watch time past 5 s is bimodal — viewers swipe at once or watch long ([ECCV 2024](https://arxiv.org/abs/2410.00289)). TikTok advises stating the main point in the first 3 s ([TikTok Ads](https://ads.tiktok.com/help/article/creative-best-practices)).
- **Cutting on action hides the cut**: about a third of cuts that coincide with the start of a motion go unnoticed ([Smith & Henderson 2008](https://bop.unibe.ch/JEMR/article/view/2264)). For an "invisible" cut, cut mid-action; for a jolt (pattern interrupt), change shot size hard or `whip` on purpose.
- **Beat**: viewers feel an edit is on the music when the **end point** of an action hits the beat, not its start ([Takehana 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6711538/)). Read `beats`, set `in_s` so the hit / squeeze lands on a beat (a `*` beat for the climax), cut just before it.
- **Rhythm**: shot lengths in modern films correlate with their neighbours, a 1/f pattern ([Cutting 2010](https://journals.sagepub.com/doi/10.1177/0956797610361679)): short clusters then a held shot, never a metronome.
- **Platforms**: Instagram distributes less of reels that are low-resolution, bordered, muted or mostly text ([Instagram](https://about.instagram.com/blog/announcements/instagram-ranking-explained)): no letterbox, keep the sound, text only supports.
- **Jump cuts / 30°**: two shots of one scene must differ enough (shot size or angle). Craft experience, no measurement.

## Match-ish cuts

The two shots need not look alike: keep **one** shared element, change everything else (object, angle, space).

- Motion direction: suitcase zip left→right → jacket zip left→right → a drawer pulled.
- Shape: round fan blades → a wheel → a spinning plate.
- Light: a screen lighting up the frame → an eye opening.
- Position in frame / action end point: the hand in the same place at the cut.

How to cut: leave shot A when the action is **almost done** (the zip nearly across the frame, the blades nearly covering it); enter shot B while a similar action is **still going** at the same position. A chain of 5: A an action at eye level → B the same action from low angle → C same direction, another object → D a shape / light match → E into a new space.

Finding pairs: the L0 `Match-cut hints` / `match_candidates(pid, from_scene, kind)` (`motion`, `shape`, `light`, `color`, `position`; score 0–1; ≥ 0.7 is worth trying). Set the two slots' `at` from the pair and `match: {kind}` on the first; `plan_solve` opens the second at its best-matching frame (report: `match shape 0.82 from slot 6`).

Real footage rarely matches by itself: take shot B / C from a generated slot (write the new angle in the `gen_image` prompt; recipes and composition matching: [camera.md](camera.md)). Check with `review_sheet(cuts=[n])`; off in position → move `in_s` / `out_s` a few frames or adjust `crop`.
