"""Generation providers. ``get_provider`` builds one by name (imports lazily)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from opencine.core.project import Project
    from opencine.gen.base import GenProvider

ProviderName = Literal["hf", "fake"]
NAMES: tuple[str, ...] = ("hf", "fake")


def get_provider(name: str, project: Project | None = None) -> GenProvider:
    if name == "hf":
        from opencine.gen.providers.hf_space import HfSpaceProvider

        return HfSpaceProvider()
    if name == "fake":
        from opencine.gen.providers.fake import FakeProvider

        return FakeProvider()
    raise ValueError(f"unknown provider {name!r} (expected one of {', '.join(NAMES)})")
