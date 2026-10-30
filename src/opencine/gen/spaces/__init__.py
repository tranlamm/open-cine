"""Code that runs on the user's Space duplicates, never imported by the engine.

One folder per Space role. A whole file is uploaded as it is (``SpaceSpec.files``); a snippet
goes into one of the upstream Space's own files next to an anchor (``SpaceSpec.patches``,
applied by ``hf_space.apply_patches``). Snippets are stored dedented and indented on read.
"""

from __future__ import annotations

import re
import textwrap
from importlib.resources import files

_PLACEHOLDER = re.compile(r"\{\{([A-Z0-9_]+)\}\}")


def code(name: str, indent: int = 0, **values: object) -> str:
    """The text of ``name`` (e.g. ``"scail/ui.py"``), each ``{{KEY}}`` set to ``values[KEY]``
    and every line indented by ``indent`` spaces."""
    text = files(__name__).joinpath(name).read_text("utf-8")
    keys = set(_PLACEHOLDER.findall(text))
    if keys != values.keys():
        raise KeyError(f"{name}: placeholders {sorted(keys)}, values {sorted(values)}")
    text = _PLACEHOLDER.sub(lambda m: str(values[m.group(1)]), text)
    return textwrap.indent(text, " " * indent) if indent else text
