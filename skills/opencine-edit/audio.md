# Sound: voice-over, music, original sound, generated-clip sound, SFX

Five layers, one job each. The voice-over sits on top; music, original sound, generated-clip sound and SFX sit under it.

## Voice-over (TTS)

A voice-over is **one continuous piece of speech** that follows the video's line of thought, not a caption per shot.

- **Writing**: in the filmmaker's voice (how they address people, the words they use in the `transcript`), natural sentences as if telling a friend. Each sentence is about what is on screen or the feeling of that moment. Voice and on-screen text complement each other; they never repeat.
- **Spell things out**: numbers, units, dates, decimals and abbreviations become words ("two point five kilograms"); the TTS only reads whole numbers reliably. Blocks under 2 s come out unstable.
- **Amount**: about 2.5 words per second in English (about 3 in Vietnamese); the share of the runtime covered comes from the template. Example: 33 s at 70 % coverage ≈ 23 s ≈ 55–70 words.
- **Blocks**: each chapter (or cluster of shots about one idea) is one block of 2–3 sentences = **one** `tts` line, read in one go with natural pauses at punctuation; usually 2–4 blocks.
- **Pauses**: 0.3–0.8 s between blocks; 1–1.5 s at a chapter change or to let a satisfying sound ring out (sizzle, crunch, pour). The hint `voice-over … stops for` flags a gap that is too long.
- **Placing**: `tts(pid, [{id, text, language?}])` in one batch returns `duration_s` per block. Place each as `sfx: [{"file": "audio/voice/<id>.wav", "at_s": …}]`, where the next block's `at_s` = previous `at_s` + previous `duration_s` + pause. A sentence starts when the picture it talks about appears (0.2–0.5 s early is fine); a sentence longer than its picture runs across the cut. `language` = `"en"`, `"vi"`…; omitted = the engine's default speech language.
- The script goes into G1 for approval. Changing one block re-synthesises only that block.

## Music

Background music runs the whole video and ducks under speech on its own (`music.duck_db`, default −10).

- The user gave a file: `music_add(pid, path)`.
- None: generate one with `music_gen(pid, prompt, seconds, bpm)`:
  - `prompt`: English prose, ≤ 512 characters, telling the arrangement in time order as you would brief a band: genre, instruments, opening → lift → peak → ending, and the scene it sits under. Example: "An upbeat lo-fi hip hop track for a morning coffee vlog. It opens with warm Rhodes chords and a soft kick, adds a bouncy bass as the day starts, and builds toward the payoff before fading out." BPM and key go in the parameters (`bpm`, `key`), never in the prompt.
  - `seconds` = timeline length + 4.
  - `bpm` = the cutting tempo. Starting points: ads 110–128, vlogs 90–110, ASMR 70–90. One 4/4 bar lasts 240/BPM s; pick a BPM that makes shots 1–2 bars long.
  - Another take: another `seed`. A `warning` about an octave error: cut on every other beat, or change the seed.
- Both return `file` (use it as `music.file`), the BPM and the strong beats. `beats(pid, name)` prints the grid as `seconds/frame`, `*` = strong.
- The end point of an action (hit, squeeze, drop) lands on a beat, with the cut just before it; `hints` flag cuts off the beat.
- The planned music (a description or the file) goes into G1.

## Original sound

- The footage's own sound, at the template's level.
- Satisfying sounds (sizzle, crunch, squeeze) play in full; never cut through them.

## Generated-clip sound

- H3 clips come with sound generated with the picture, following `overall_soundscape` ([gen.md](gen.md)). Generated clips are muted by default.
- When the sound matches the action, turn it on with `gain` on the item (usually −6…0 dB).

## SFX

`sfx_list(pid, query)` searches with English words ("knife chop", "sizzle", "pour", "whoosh"). The library holds imported packs (`<pack>/<name>`, with tags and duration) and a synthesised set (whoosh, swish, pop, click, tick, riser, impact, slice, scribble). Two uses:

- **Foley** for an on-screen action that lacks sound (knife on the board, glass set down): on the exact contact frame, −12…−6 dB.
- **Accents** on transitions and text (whoosh on a whip, pop as a label appears, riser → impact into the payoff): −10…−4 dB.

Consecutive accents use different files.
