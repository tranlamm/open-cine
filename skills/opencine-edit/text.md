# On-screen text: cards, captions, placement

You choose the **content, style and timing**; the engine chooses **where**. Leave `position` empty (= `"auto"`).

Two text layers; they never share a band while both show (a hint says so). Big rhythmic cards (hook, payoff, price) can go to the motion layer instead of item text: [motion.md](motion.md).

## Item cards (`items[].text`, one idea per card)

- Hook, 0–2 s: `title` ≤ 6 words (usually mid-frame), on screen from frame 0, `anim` `pop` or `wipe`.
- Step labels, benefits: `label` ≤ 4 words (usually top), `rise` or `pop` (with a pop SFX).
- Payoff: 1–3 big words, or let the picture speak; `pop`, or `shake` once per video.
- CTA: `title` + `box: true`, `wipe`, held ≥ 1.5 s (price, name, "link in bio").
- `emphasis`: the memorable word (benefit, price, feeling) in the accent colour at 112 %.
- `type` (typewriter) only for a vlog clock or "POV:", never for a hook (it takes 0.6 s to read).

## Captions (`captions`)

Captions follow the voice-over word by word; the style comes from the template.
- `chunk`: 2–4 syllables, upper case, the word being said in yellow (narration that sells).
- `karaoke`: the whole sentence, words change colour as they are said (long narration).
- `line`: plain subtitles (quiet voice, ASMR).

`emphasis` on 5–15 % of the words (benefits, prices, the payoff word). Chunks broken in the wrong place: `breaks`. Bright or busy footage: `box: true`.

## Safe area and fonts

- The engine keeps text inside the safe area: 15 % from the top, 35 % from the bottom, 11 % at each side; bottom-band captions stay inside the middle 64 %.
- Fonts: Anton (display) and Montserrat (body); both cover accented Latin, Vietnamese included.
- A video with burned-in captions: the G4 summary reminds the user to turn off the app's auto-captions.

## What `auto` does

- A card sits in one of three bands of the safe area: **top** (from the 15 % line), **center**, **bottom** (above the 35 % line).
- The engine measures the card with the real font (wrapping inside the 11 % margins). Every 0.25 s while the text shows, it compares that box with the item's cropped / zoomed picture:
  - the **subject** tracked at ingest (face > moving thing > salient area; generated clips and stills are tracked the first time they are needed): how much the card covers;
  - how **busy** the picture under the card is (food, texture, printed text): text on busy ground is hard to read and hides content.
- It picks the band that hides the least. Near ties follow each style's usual band: `title` center → top → bottom; `label` and `time` top → bottom → center; `plain` bottom → top → center.
- While captions run, their band is excluded.
- `review_sheet` prints the chosen band after each card: `text "So crispy" (top)`.

## When to set `top` / `center` / `bottom`

Only on purpose: the user asked, a label points at something in that band, or a series of step labels must stay in one place. A forced band that hides the subject gets a hint from `plan_save`.

## Hints and fixes

- `hides the face 60% of the 1.5 s it shows at "center"; "top" hides 0%: use position "auto"` → drop `position`, or use the band suggested.
- `hides the … wherever it goes (auto → top)`: the subject nearly fills the frame. Fix in this order:
  1. shorten the text (≤ 4 words) or lower `size`;
  2. move `start_s` / `end_s` to where the subject is smaller, or move the card to the next item with room (a wider shot);
  3. change the `crop` so the subject sits to one side and leaves room;
  4. a hook on a frame-filling close-up: let the picture speak, bring the text in after 0.5 s or on the next shot.
- A **face**: never cover it. **Food or a product filling the frame** (motion / saliency): a card may cover a corner, as long as it leaves the action clear (knife contact, the pour, the product logo); busy ground → `box: true`.

## Check by eye (R7)

The engine tracks **one** subject at a time. It does not see hands or small objects outside the main subject, printed text or logos on a product, or a second person. In `review_sheet`, look at every frame that has text. Text covering something important: set another `position` band and give the reason in the item's `note`.
