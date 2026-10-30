# Camera angles and camera moves for generated slots

A generated slot earns its quota when it shows **a view the footage does not have**: another shot size, an unusual angle, another lens, a deliberate camera move. Choose them at S4 (`generate.camera` = shot size + angle + lens, `generate.motion` = action + camera move); write the prompts at S5 following [gen.md](gen.md).

## Choose by what the shot must do

| Need | Still (new view) | Clip (camera move) |
|---|---|---|
| Curiosity hook | odd macro detail, worm's-eye | push in on a detail; pull out to reveal the whole |
| Strong, energetic subject | low-angle hero | tracking backward ahead of them, low |
| Overview of a process | top-down flat lay | top-down, pedestal up + slight roll |
| Close action, ASMR | ECU macro, shallow DOF | static or small push in |
| Phase change, placing in context | wide establishing | large pull out |
| Immersion | POV hands | POV + shakes slightly |
| Product, finished result | 3/4 low angle; rim backlight | arc shot around it |
| Mystery, depth | blurred foreground (glass, leaves, door frame); from behind | truck past the foreground, revealing… |
| Tension, off-kilter | Dutch angle | roll clockwise small; shakes |
| Surprise, "impossible camera" | from inside an object; along a tool; spying from afar (13–15) | static; POV + shakes slightly |
| One take re-shot from many angles | the angle table in [multicam.md](multicam.md) | static or small push in |

- Neighbouring shots differ by ≥ 1 shot size or ≥ 30° of angle. Unusual angles (Dutch, worm's-eye, reflections) ≤ 1–2 per video.
- Keep the footage's light and colour so the generated shot blends in: change the angle, not the world.

## Stills: Qwen-Image-2.1 recipes

Open with `Re-photograph this scene as a new shot from a different camera position:` (unless noted) and end with the keep sentence **K** = `Keep the same person, face, hairstyle, clothing, colours and setting as the image.` (objects: `Keep the same objects, their colours, materials and the surface.`). Replace `[…]` with what is in the frame.

1. **Low-angle hero**: `from knee height, tilted upward: a low-angle hero shot where [he] towers over the lens, head near the top of the frame, the [ceiling] now visible above, vertical lines converging; 24 mm wide-angle look.`
2. **Worm's-eye**: `the camera on the floor pointing almost straight up at [her]: blurred floor across the bottom edge, shoes and legs large, head small at the top against the [sky]; 16 mm ultra-wide.`
3. **High angle**: `from two metres above and in front of [her], tilted steeply down: she looks up at the lens, smaller in the lower half, the [floor] filling the background.`
4. **Top-down flat lay**: `from directly overhead, pointing straight down: a flat lay of the [table] with [cup, notebook, phone] arranged as in the scene, [hands] entering from the bottom; even soft light, everything in focus.`
5. **Macro / ECU**: `Move the camera very close to [the cup rim]: an extreme close-up where it fills most of the frame, fine texture visible, very shallow depth of field, the rest melting into bokeh in its own colours; 100 mm macro look.`
6. **Over-the-shoulder**: `the camera just behind [her] left shoulder: her blurred shoulder and the back of her head in the lower-left foreground, [the man opposite] sharp in the upper middle; 50 mm.`
7. **POV hands**: `a first-person view from [his] eye position looking down: his hands enter from the bottom corners holding [the phone], [blue] sleeves at the wrists, [the street] receding out of focus.` (Fingers distort easily: have the hand hold one object, few fingers visible.)
8. **3/4 / profile**: `the camera 45 degrees to [her] left at eye level: a three-quarter view, [the window] now behind her; 85 mm.` (A 90° profile is riskier.)
9. **From behind, opening onto the space**: `Move the camera behind [him] at shoulder height: seen from the back in the lower-left third, looking over [the city at dusk], which fills the rest of the frame.` (The hardest.)
10. **Foreground framing**: `from a few metres back, behind [the doorway / out-of-focus leaves / rain-streaked glass], which frames the edges as a blurred foreground; [she] sharp and smaller in the gap; 85 mm telephoto compression.`
11. **Backlight / silhouette**: `from the opposite side so [the window] is directly behind [her]: a dark figure with a bright rim of light on hair and shoulders, gentle flare.` Keep: `Keep her outline, hairstyle, clothing shapes and setting.`
12. **Wide establishing**: `Pull the camera far back and up into a wide establishing shot: [she] about one fifth of the frame height in the lower third, the [cafe walls and tables] continuing past the original edges; 24 mm, deep focus.`
13. **From inside an object** (cup, pot, fridge, oven): `from inside [the empty cup] on the table, pointing straight up through its round rim: the [white inner wall] forms a soft dark ring around the edges; through the opening [his face and hands] lean over the lens from above, [pouring from the kettle]; 14 mm ultra-wide, [the sky] behind him.`
14. **Along a tool** (knife, spoon, tap): `at the tip of [the knife], looking along the blade: the steel runs sharp from the bottom edge toward [the onion half], which fills the upper middle, large and close; [his fingers] soft at one side; 60 mm macro, very shallow depth of field.` (Keep: the object sentence.)
15. **Spying from afar**: `Pull the camera far back across [the street] to behind [a window of the house opposite]: the dark window frame and faint reflections on the glass fill the edges; through the glass [he] is small in the middle distance at [his table]; 135 mm telephoto compression, slightly hazy.` (Hard.)
16. **Object in mid-air**: `Move the camera to table height beside [his open hand], pointing slightly up: [the orange] hangs sharp in mid-air above the palm, about a quarter of the frame, [his face] soft behind it; 85 mm, very shallow depth of field.`
17. **Big object in the foreground**: `from table height just behind [the teapot], which fills the left third as a large, sharp foreground; [he] sits beyond it in the middle of the frame, [the trees and sky] above; 20 mm wide-angle look.`

- 13–15 are the "impossible camera" angles: memorable because they are strange, so use them sparingly ([multicam.md](multicam.md)). If the still comes back looking like the old angle, rewrite the camera position and what is newly revealed more strongly.
- Dutch angle: a rotate + crop in the EditPlan is cheaper and more exact than generating it.

## Clips: H3 camera-move recipes

Each recipe is the text after `[Shot 1] Live-action, cinematic,`; add `The whole shot is one continuous take.` and the `overall_soundscape` / `non_diegetic_music: N/A` fields as in the template in [gen.md](gen.md). Every recipe runs at real-time pace with a 2–3 beat action ([gen.md](gen.md) § Clips).

- **Push in to reveal a detail** (hook, emotion): `shallow depth of field, a medium shot frames the woman shown in <Picture 1>, preserving her appearance and colours. The camera pushes in with small amplitude toward her hands as she flips the ring box open at once, in real time at a natural pace; she tilts it toward the lens, and the ring catches the light in close-up.`
- **Arc around a product** (finished result, payoff): `a low-angle close shot frames the glass shown in <Picture 1> on the counter, preserving its colours. The camera arcs left around the glass as a hand sets it down at once, in real time at a natural pace; ice settles with a clink, condensation runs down its side, and a highlight slides along the glass.`
- **Low tracking** (energy): `a low-angle medium-wide shot frames the runner shown in <Picture 1>, preserving her appearance. The camera tracks backward ahead of her at fast speed as she starts jogging toward the lens at once, in real time at a natural pace; her arms swing, streetlights pass behind her, ending mid-stride.`
- **Top-down rising** (process overview): `a top-down close shot frames the bowl shown in <Picture 1>, preserving its colours. Steam rises at once as the camera pedestals up with small amplitude while rolling clockwise with small amplitude, revealing the full table; a hand sets chopsticks beside the bowl in real time.`
- **POV hands** (immersion): `a POV shot from the barista's eyes looks down at the hands, jug and cup shown in <Picture 1>. The camera shakes slightly as the right hand tilts the jug at once and pours a thin, steady stream in real time at a natural pace; a heart forms in the cup, then the jug lifts away.`
- **Pull out to the context** (phase change, ending): `a close-up frames the potter's hands on the wheel shown in <Picture 1>. The camera pulls out with large amplitude as the hands raise the clay on the spinning wheel, in real time at a natural pace; shelves and the sunlit studio enter the frame, ending on a wide shot of the potter at the wheel.`
- **Multicam angle** (one beat of the take, [multicam.md](multicam.md)): the action exactly as in the take, camera nearly still: `soft natural light, a high-angle telephoto medium shot through out-of-focus leaves frames the man at the table shown in <Picture 1>, preserving his appearance and the layout. Static shot as he lifts the kettle at once and pours hot water into the filter, in real time at a natural pace; steam rises, then he sets the kettle down to his right.`

## Combining techniques

- **Controlled camera move** (first+last): still B = a new angle of **the same source frame** (e.g. recipe 8, a 45° three-quarter view); the clip's `first` = the source frame, `last` = still B, and the prompt's arc / push in follows the difference between the two frames ([gen.md](gen.md)). Costs one more still (~215 s with one ref); exact totals from `gen_check` / `hf_quota()`. The move lands more precisely than when H3 picks it.
- **Match cut from a generated still**: B must match the composition of A's last frame → `refs` = [B's source frame, A's last frame]: `Re-photograph the scene of <image1> with the camera angle, shot size and composition of <image2>: [describe it in words: a round object in the middle of the frame, half the width]. <image2> supplies only the camera angle and layout; keep the person, clothing, colours and setting of <image1>.` (Untested.)
- **Continuing a camera move**: shot A pushes in → cut → shot B keeps pushing in at the same speed; or A `trucks right` → B moves left→right. Keep one element ([editing.md](editing.md) § Match-ish cuts).
- **Whip between phases**: end the clip with `pans right with large amplitude at fast speed` + `transition_in: whip` on the next item. Turbo smears easily: only at the clip's end, where the smear becomes the whip streak.
