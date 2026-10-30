"""``opencine gen hf-setup``: create the private ZeroGPU duplicates on every HF account.

For each account (``OPENCINE_HF_TOKENS``, else ``HF_TOKEN``) and role, a missing duplicate is
made by ``duplicate_space`` (upstream at the pinned revision + the engine's patches), named
``<user>/<name>`` and pointed at that account's own conditioner. The ``OPENCINE_HF_SPACE_<ROLE>``
lines (one entry per account; existing entries keep their place, the first names the Space in
cache keys) go to ``~/.opencine/.env``. Each duplicate's files are then compared with the first
account's so every account runs the same Space. Nothing needs to be public.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from opencine.core.config import ENV_FILE, get_settings
from opencine.gen.base import AuthError, ProviderError
from opencine.gen.config import SPACE_HOSTS, GenSettings, SpaceRef, get_gen_settings
from opencine.gen.providers import hf_space
from opencine.gen.providers.hf_accounts import Account, make_accounts

# Roles the Director uses (video, swap, restyle and rife are optional extras); a PRO account
# can host 10 ZeroGPU Spaces.
DEFAULT_ROLES = ("image", "cond", "video_turbo", "video_ref", "swap_h3", "music", "enhance")


def setup_roles(roles: list[str] | None, s: GenSettings) -> list[str]:
    """``roles`` (default: those already set, else ``DEFAULT_ROLES``), each after the roles it
    calls (the conditioner first, since the H3 Spaces point at it)."""
    out: list[str] = []
    for role in roles or [r for r in hf_space.SPACES if s.hf_spaces(r)] or DEFAULT_ROLES:
        if role not in hf_space.SPACES:
            raise ProviderError(f"unknown role {role!r}: {', '.join(hf_space.SPACES)}")
        role = SPACE_HOSTS.get(role, role)  # a hosted endpoint is set up with its host's Space
        out += [r for r in (*hf_space.SPACES[role].needs, role) if r not in out]
    return out


def _with(s: GenSettings, role: str, refs: list[SpaceRef]) -> GenSettings:
    return s.model_copy(update={f"hf_space_{role}": ",".join(map(str, refs)) or None})


def write_env(path: Path, values: dict[str, str]) -> None:
    """Set ``KEY=value`` lines in a .env file: existing keys are replaced in place, new ones
    appended, everything else left untouched."""
    raw = path.read_bytes().decode("utf-8") if path.exists() else ""
    nl = "\r\n" if "\r\n" in raw else "\n"
    lines, left = raw.splitlines(), dict(values)
    for i, line in enumerate(lines):
        key = line.split("=", 1)[0].strip()
        if "=" in line and key in left:
            lines[i] = f"{key}={left.pop(key)}"
    lines += [f"{k}={v}" for k, v in left.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((nl.join(lines) + nl).encode("utf-8"))


def _notes(acct: Account, roles: list[str]) -> list[str]:
    """Steps only the account owner can take: get PRO, accept gated model licences."""
    out = [] if acct.pro is not False else [
        f"{acct.name}: not PRO (ZeroGPU duplicates and the 40 min/day quota need it)"]
    api = hf_space._make_api(acct.token)
    for repo in sorted({g for r in roles for g in hf_space.SPACES[r].gated}):
        try:
            api.auth_check(repo)
        except Exception:  # noqa: BLE001 — gated and not accepted (or unreachable)
            out.append(f"{acct.name}: accept the licence at https://huggingface.co/{repo} "
                       f"(needed by {', '.join(r for r in roles if repo in hf_space.SPACES[r].gated)})")
    return out


def _sync(accts: list[Account], role: str, s: GenSettings) -> dict[str, Any]:
    """Compare each account's duplicate of ``role`` with the first account's files."""
    refs = [(a, a.space(role, s)) for a in accts]
    refs = [(a, r) for a, r in refs if r]
    if len(refs) < 2:
        return {}
    base = hf_space.space_tree(hf_space._make_api(refs[0][0].token), refs[0][1].id)
    out: dict[str, Any] = {}
    for a, ref in refs[1:]:
        tree = hf_space.space_tree(hf_space._make_api(a.token), ref.id)
        diff = sorted(p for p in base.keys() | tree.keys() if base.get(p) != tree.get(p))
        out[a.name] = {"same_as": refs[0][1].id} if not diff else {"differs_from": refs[0][1].id,
                                                                    "files": diff[:20]}
    return out


def setup_spaces(roles: list[str] | None = None, *, refresh: bool = False, dry_run: bool = False,
                 env_file: Path | None = ENV_FILE, settings: GenSettings | None = None,
                 tokens: list[str] | None = None) -> dict[str, Any]:
    """Create missing duplicates of ``roles`` on every account, write the
    ``OPENCINE_HF_SPACE_<ROLE>`` lines to ``env_file`` (None = only return them) and compare
    the accounts' files. ``refresh`` re-applies secrets, variables and patches to existing
    duplicates; ``dry_run`` reports what would change and changes nothing."""
    s = settings or get_gen_settings()
    tokens = tokens if tokens is not None else get_settings().hf_account_tokens()
    if not tokens:
        raise AuthError("set HF_TOKEN, or OPENCINE_HF_TOKENS for several accounts, first")
    accts = make_accounts(tokens, s, hf_space._whoami, lookup=True, refresh=True)
    roles = setup_roles(roles, s)
    report: dict[str, Any] = {"accounts": [{"name": a.name, "pro": a.pro, "orgs": list(a.orgs)}
                                           for a in accts], "roles": {}, "notes": []}
    for a in accts:
        report["notes"] += _notes(a, roles)
    for role in roles:
        spec, rows = hf_space.SPACES[role], {}
        refs = s.hf_spaces(role)
        name = refs[0].id.partition("/")[2] if refs else spec.repo or role.replace("_", "-")
        for a in accts:
            ref = a.space(role, s)
            if ref and not refresh:
                rows[a.name] = {"space": ref.id, "status": "kept"}
                continue
            target = ref.id if ref else f"{a.name}/{name}"
            if dry_run:
                rows[a.name] = {"space": target, "status": "would refresh" if ref else "would create"}
                continue
            cond = a.space("cond", s)  # this account's conditioner, set up first
            out = hf_space.duplicate_space(role, target, token=a.token, settings=_with(
                s, "cond", [cond] if cond else []))
            new = SpaceRef(out["space"], out["sha"])
            known = any(r.id == new.id for r in refs)
            refs = [new if r.id == new.id else r for r in refs] + ([] if known else [new])
            s = _with(s, role, refs)
            rows[a.name] = {"space": new.id, "sha": new.sha, "status": "refreshed" if ref else "created",
                            **({"reverted": out["reverted"]} if out.get("reverted") else {})}
        report["roles"][role] = rows
    env = {f"OPENCINE_HF_SPACE_{r.upper()}": ",".join(map(str, s.hf_spaces(r))) for r in roles
           if s.hf_spaces(r)}
    report["env"] = env
    for role in roles:
        if sync := _sync(accts, role, s):
            report["roles"][role]["sync"] = sync
    if env_file is not None and not dry_run:
        write_env(env_file, env)
        report["env_file"] = str(env_file)
    return report
