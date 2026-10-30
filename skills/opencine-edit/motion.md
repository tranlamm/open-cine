# Motion: kinetic text and effects written as code

Contents: When to use · Workflow · The layer file · The `opencine-motion` kit · Text behind the subject · Rules · Recipes.

Animation as code: change one word, number or timing and everything else stays; every render comes out the same. You write **one TSX file** with the `opencine-motion` kit (Remotion, in the engine's `fx/`). The `motion` tool draws it, transparent, over the render.

Item text (`items[].text`, ASS) stays the default: fast, and it avoids the subject on its own. Motion is the layer of **big, rhythmic text**: hook, payoff, price, CTA, chapter titles, with effects on the exact hit.

## When to use

- **The user asks**: in chat or in the first request.
- **Your call** when the video needs energy (cut to the beat, product ads, sport, trends, style switches) and text is what pulls the viewer at the hook, payoff or CTA. Say in the G4 summary that motion was used.
- **Not for**: ASMR, quiet vlogs, or when the user (or `taste`) says "less text" or "no effects".
- One card, one layer. A card moved to motion drops its item `text` in the EditPlan (`plan_save`, then render again). Voice captions (`captions`) stay in ASS; motion cards keep out of the bottom band while captions run.

## Workflow

1. EditPlan done; `render(pid, preset="preview")` as usual.
2. Write the layer, then `motion(pid, code, marks, check=true)`: a sheet with one still per span once everything has come in (review budget). Check: readable, not over a face or the action, inside the safe area, on the beat. Fix and check again, at most 2 rounds.
3. `motion(pid, code, marks)` → `renders/<render>-motion.mp4` (a job). With scribble, `scribble(pid)` then draws over the motion version. Put the newest file in G4's `media`.
4. After G4 approval: `render(pid, preset="final")` → `motion(pid, preset="final")` (no `code`: the saved layer is used) → scribble final if any → give the user the final path.

A layer that fails to build (syntax) or throws while drawing (e.g. an unknown mark) comes back as the error verbatim; fix the code and call again.

## The layer file

```tsx
import {Text, Counter, Burst, Flash, Shock, SpeedLines, Show, Behind, useMotion, nextBeat, beatPulse} from 'opencine-motion';

export const spans: Array<[number, number]> = [[0, 1.8], [6.2, 8]];  // only these ranges render

export default function Layer() {
  const {strong, mark} = useMotion();
  const hit = mark('hit');                       // marks={"hit": "S03@2.1"}
  return (<>
    <Flash at={hit} />
    <Text text="SO CRISPY" at={hit - 0.03} dur={1.5} enter="slam" exit="pop" emphasis={['crispy']} />
    <Counter to={49} prefix="$" at={nextBeat(6.2, strong)} dur={1.6} band="bottom" />
  </>);
}
```

- Units: times are timeline seconds (as `sfx.at_s` and the `review_sheet` table); `x`, `y` are 0–1 of the frame; `size` is a fraction of the frame height.
- `spans`: every `[at, at + dur]` of the layer's elements, merged. Without `spans` the whole video renders (much slower).
- `useMotion()` gives `t` (current second), `beats`, `strong` (strong beats), `cuts` (start second of each shot), `words` (`{text, start, end}` of the voice-over), `marks`, `mark(name)` (throws if missing), `safe` (`top` 0.15, `bottom` 0.35, `side` 0.11 in portrait), `width`, `height`, `fps`.
- `marks` (a tool parameter) names moments instead of hard-coding seconds: `"S03@2.1"` is that footage moment wherever the EditPlan puts it, `"say:forty-nine"` is when the voice-over says it, or plain seconds. The engine resolves them to timeline seconds.
- Beyond the kit, write any React / SVG / CSS: compute from `useMotion().t`, use `interpolate`, `spring`, `random` (seeded) from `'remotion'`. Never `Math.random()` or wall-clock time: frames must come out identical on every render. For other Remotion techniques, see the `remotion-best-practices` skill if you have it.

## The `opencine-motion` kit

| Component | For | Main props |
|---|---|---|
| `Text` | any animated text | `text` (`\n` breaks lines), `at`, `dur`, `enter`, `exit`, `by` `char`/`word`, `stagger`, `font` `display` (Anton, upper case by default) / `body` (Montserrat ExtraBold), `size`, `color`, `gradient: [top, bottom]`, `emphasis`, `accent`, `mark` (highlighter sweep behind the emphasised word), `stroke`, `strokeColor`, `glow`, `box` (pill background), `rotate`, `band` / `x`, `y`, `maxWidth`, `align`, `pulse` `beats`/`strong`, `upper` |
| `Counter` | prices, stats counting up then bouncing | `to`, `from`, `count` (seconds counting), `decimals`, `prefix`, `suffix`, plus every text-style prop of `Text` |
| `Flash` | full-frame flash on a hit | `at`, `dur` 0.14, `color`, `strength` |
| `Shock` | shockwave ring from the point of impact | `at`, `x`, `y`, `size`, `color` |
| `SpeedLines` | manga speed lines | `at`, `dur`, `x`, `y`, `count`, `color`, `inner` |
| `Burst` | sparks, confetti, emoji bursting out | `at`, `x`, `y`, `kind` `spark`/`confetti`/`emoji`, `emoji`, `count`, `colors`, `spread` |
| `Show` | wraps your own elements, visible only in `[at, at + dur)` | `at`, `dur` |
| `Behind` | what it wraps sits behind a person or object (§ Text behind the subject) | `matte` |
| `beatPulse(t, beats)`, `nextBeat(t, beats)`, `beatsIn(a, b, beats)` | bounce on the beat, snap to a beat | |

Entrances (`enter`):

| `enter` | Feel | Good for |
|---|---|---|
| `slam` | drops in from big, shakes | hook, payoff, a stomp / drop (with `Flash` + `Shock`) |
| `zoom` | grows from small, bounces | numbers, product names |
| `pop` | letters pop up | short labels, keywords |
| `stagger` | letters / words jump up, slight rotation | step labels, benefits |
| `wave` | like `stagger`, then a gentle wave | playful vibe, light music |
| `rise` | slides up, fading in | narration, captions |
| `mask` | each line rises from behind an edge | two-line titles, editorial |
| `blur` | sharpens from a blur | cinematic, slow openings |
| `flip` | 3D flip up | chapter changes |
| `type` | typed, with a cursor | "POV:", clock times; never a hook |
| `scramble` | random characters resolving to text | tech, mystery, price reveals |
| `glitch` | RGB split, jitter | cyberpunk, style switches, theme changes |

`exit`: `fade` (default), `pop`, `blur`, `drop`, `glitch`, `none`.

## Text behind the subject (`<Behind>`)

The person or object passes **in front of** the text, like CapCut's or Edits' "text behind subject". Good for a big hook title, or a chapter title the person walks through.

```tsx
<Behind matte="me">
  <Text text="WALKING" at={0.1} dur={1.4} enter="rise" exit="glitch" y={0.36} size={0.34} />
</Behind>
<Flash at={stomp} />   {/* outside <Behind>: stays in front of the person */}
```

- Pass `behind={"me": {"on": "person", "from": 0.0, "to": 1.54, "at": 0.4}}` to the `motion` tool (stored with the code; the final needs no repeat).
  - `on`: a few English words for the thing ("person", "the dancer", "red car"); default `person`.
  - `from`, `to`: seconds or moments (`"S03@2.1"`) that cover everything inside `<Behind>` while it shows. Outside that range, the wrapped content does not show.
  - `at`: when the thing is clearly visible, to find it (default: each shot's start inside the range). Set it when the thing enters mid-shot. The mask spreads both backward and forward from `at`.
  - `x`, `y`: pick one person among several. Two people, one title each: two mattes, two names.
- Only what is inside `<Behind>` goes behind the person. Do not set `visibility` inside the wrapped part.
- **Placement**: big display text (`size` 0.25–0.4) on background (sky, wall), so the person's head, body or arm cuts through **part** of the text.
- **Check**: `check=true` reports, per matte, how much of the text it covers at which second.
  - About 10–35 % looks good.
  - Over 40 %: hard to read; move the text, resize it, or shorten the range.
  - Near 0 %: the person never crosses the text; use plain text instead.
- **Restraint**: 1–2 behind-subject titles per video.
- **Cost**: the first time ~0.3 s per frame in the range (~10 s per second of video at 30 fps), plus 7 s to load the model. Cached per EditPlan version, so the final reuses the preview's. Each matte ≤ 15 s.
- **Limits**: hair edges and fingers are a little rough (EdgeTAM masks at 256²); head, body, arms and legs are clean, even in fast swings. Where the thing leaves the frame, nothing is masked.

## Rules

- **On the beat**: hitting text (`slam`, `zoom`) gets `at` 0.03 s before the beat so its largest frame lands on it. `Flash`, `Shock`, `SpeedLines` only on real hits: a strong beat, a landing, the payoff, the price.
- **Restraint**: the message first, the effect second. At most 1 big text card and 1 effect on screen at once. Each video uses 2–3 `enter` styles as a system (e.g. `slam` for hook / payoff, `stagger` for labels); two consecutive cards never share one.
- **Eye safety**: at most 3 flashes per second (WCAG 2.3.1); in total usually no more than 1 `Flash` per 2 s.
- **Readable**: hook ≤ 6 words, labels ≤ 4. Once in, text holds still for at least max(0.8 s, words ÷ 3) s. Keep the default `stroke`. Bright or busy ground: add `box` or `glow`. At most 2 text colours + 1 accent per video.
- **Placement**: `band` (`top` / `center` / `bottom`, safe area already applied) or `y` between `safe.top` and `1 − safe.bottom`. Never over a face or the action (check on the sheet). In 16:9, `safe` adapts.
- **Prices**: `$49`, `49K`, or a `suffix` unit with `upper={false}` to keep it lower case. Both fonts cover accented Latin (Vietnamese included).
- **Sound**: hitting text and effects get an SFX in the EditPlan (`impact`, `whoosh`, `pop`) at that second.

## Recipes

- **Beat-slam hook**: `Flash` + `SpeedLines` 0.5 s + `Shock` at the impact + `Text enter="slam" exit="pop"` 2–4 words, `emphasis` on 1 word.
- **Step label**: `Text font="body" band="top" enter="stagger" mark="#FF2BD6" emphasis={['15 minutes']}`.
- **Price**: `Counter` `band="bottom"` `gradient={['#FFF3A0', '#FFB800']}` + `Burst kind="spark"` when the count ends (`at + count`).
- **Style switch**: theme name `enter="glitch" exit="glitch" glow="#00E5FF" pulse="strong"`, `at` on the `cuts` of each theme.
- **Following the voice-over**: `words.filter(w => …)` to slam a keyword exactly as it is said (`at={w.start - 0.03}`).
