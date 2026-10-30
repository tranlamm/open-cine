"""Genre templates for ``plan_solve``: rhythm, sound levels, text, SFX and look defaults.

The Director picks ``ShotPlan.template``; this table is what the solver does with it. A slot's
own fields always win.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# how far into its dur_s range a slot of each role leans (climax burst short, payoff held)
BASE_WEIGHT = {"hook": 0.7, "climax": 0.35, "build": 0.9, "process": 1.0, "setup": 1.2,
               "payoff": 1.8, "loop": 0.8, "outro": 1.2}


@dataclass(frozen=True)
class Template:
    name: str
    role_weight: dict[str, float] = field(default_factory=lambda: dict(BASE_WEIGHT))
    footage_db: float = 0.0  # the footage's own sound
    speech_db: float | None = None  # footage where someone speaks (transcribed words); None = footage_db
    music_db: float = -6.0
    # role → (style, anim); the band is "auto" (``render.textplace``: away from the subject)
    text: dict[str, tuple[str, str]] = field(default_factory=dict)
    text_default: tuple[str, str] = ("plain", "none")
    chapter_transition: str | None = None  # between Brief.story chapters when the slot sets none
    chapter_time: bool = False  # a {time} clock on each chapter's first footage slot
    auto_sfx: dict[str, str] = field(default_factory=dict)  # event → sfx name (see SFX_EVENTS)
    sfx_db: float = -8.0
    look: str | None = None  # EditPlan.look (opencine.render.look); None = none
    deliverable: str = "1080x1920"  # EditPlan.deliverable unless ShotPlan.deliverable says
    # a generated clip's own soundtrack (H3's soundscape) in dB; None = muted (the footage's
    # sound and the music carry the edit)
    gen_db: float | None = None

    def text_for(self, role: str | None) -> tuple[str, str]:
        return self.text.get(role or "", self.text_default)


# events the solver can attach sound to: a whip/slide transition, a text card appearing, a
# title card (a riser leading into it), the cut into the payoff
SFX_EVENTS = ("push", "text", "title", "payoff")

TEMPLATES: dict[str, Template] = {
    "process_asmr": Template(
        "process_asmr", footage_db=0.0, music_db=-12.0,  # the real sound is the point
        text={"hook": ("title", "pop")}, text_default=("label", "pop"),
        auto_sfx={"push": "whoosh"}, sfx_db=-10.0, look="clean"),
    "day_in_life": Template(
        "day_in_life",
        role_weight={**BASE_WEIGHT, "setup": 1.0, "process": 0.9, "build": 0.8, "climax": 0.4,
                     "payoff": 1.6},
        footage_db=-12.0, speech_db=0.0, music_db=-3.0,  # music carries it, spoken lines stay
        text={"hook": ("title", "fade")}, text_default=("label", "pop"),
        chapter_transition="whip", chapter_time=True,
        auto_sfx={"push": "whoosh", "text": "pop"}, sfx_db=-10.0, look="warm"),
    "product_ad": Template(
        "product_ad",
        role_weight={**BASE_WEIGHT, "hook": 0.6, "setup": 1.0, "process": 0.8, "build": 0.7,
                     "payoff": 1.4, "outro": 1.4},
        footage_db=-10.0, speech_db=0.0, music_db=-6.0,
        text={"hook": ("title", "pop"), "outro": ("title", "fade")},
        text_default=("label", "pop"),
        auto_sfx={"push": "whoosh", "text": "pop", "title": "riser", "payoff": "impact"},
        sfx_db=-6.0, look="vivid"),
    # a film from a prompt (skill film): every shot generated, landscape, scenes held, the
    # clips' own sound (engines, rain, steps) is the scene's sound, the score sits under it
    "film": Template(
        "film",
        role_weight={**BASE_WEIGHT, "hook": 1.0, "setup": 1.3, "process": 1.1, "build": 1.0,
                     "climax": 0.7, "payoff": 1.6, "outro": 1.5},
        footage_db=0.0, music_db=-10.0, gen_db=-2.0,
        text={"hook": ("title", "fade"), "outro": ("title", "fade")},
        text_default=("plain", "fade"), chapter_transition="dip",
        sfx_db=-10.0, look="film", deliverable="1920x1080"),
}
DEFAULT = Template("")  # no template: plain defaults


def template(name: str | None) -> Template:
    return TEMPLATES.get((name or "").strip().lower(), DEFAULT)
