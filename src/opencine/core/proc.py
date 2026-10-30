"""Child-process helpers: no console windows on Windows, executable lookup."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

NO_WINDOW: dict[str, Any] = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def find_exe(name: str) -> str:
    """``shutil.which`` that prefers a runnable ``.exe``/``.cmd`` over npm's extensionless shim on Windows."""
    found = shutil.which(name)
    if os.name == "nt" and found and not Path(found).suffix:
        for ext in (".exe", ".cmd", ".bat"):
            if alt := shutil.which(found + ext):
                return alt
    return found or name
