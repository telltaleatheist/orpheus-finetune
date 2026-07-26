"""Machine-specific roots, resolved once.

Every path in this project used to be hardcoded to the author's machine. They now come
from environment variables, optionally loaded from a `.env` file in the repo root.

Deliberately NO fallback defaults: an unset root raises with the name to set and what it
means. A silent default sends training output somewhere you did not intend and you find
out an hour later, which is worse than failing on line one.

    from config import FT_ROOT, TITLES_ROOT

Copy `.env.example` to `.env` and fill it in. `.env` is gitignored.
"""

from __future__ import annotations

import os
from pathlib import Path

_ENV_FILE = Path(__file__).resolve().parent / ".env"

_DESCRIPTIONS = {
    "ORPHEUS_FT_ROOT": (
        "Where training runs are written: datasets, LoRA adapters, merged models. "
        "Wants tens of GB and fast local disk. On WSL keep it on the Linux filesystem "
        "(e.g. /home/<you>/xtts_ft), NOT under /mnt/c — the 9p mount is painfully slow."
    ),
    "ORPHEUS_TITLES_ROOT": (
        "Root of the YouTube title-model corpus, containing source/ and build/. "
        "Only needed for the title-model scripts, not for voice cloning."
    ),
}


def _load_env_file() -> None:
    """Populate os.environ from .env, without overriding anything already set."""
    if not _ENV_FILE.is_file():
        return
    for raw in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env_file()


def root(name: str) -> Path:
    """Return the Path for an environment root, or raise explaining how to set it."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set.\n\n{_DESCRIPTIONS.get(name, '')}\n\n"
            f"Set it in {_ENV_FILE} (copy .env.example) or export it in your shell:\n"
            f"    export {name}=/path/to/dir"
        )
    return Path(value)


def optional_root(name: str) -> Path | None:
    """Same, but returns None when unset — for scripts where the root is genuinely optional."""
    value = os.environ.get(name)
    return Path(value) if value else None


# Resolved lazily via module __getattr__ so importing config.py never raises; only the
# script that actually needs a given root pays for it being unset.
def __getattr__(name: str) -> Path:
    if name == "FT_ROOT":
        return root("ORPHEUS_FT_ROOT")
    if name == "TITLES_ROOT":
        return root("ORPHEUS_TITLES_ROOT")
    raise AttributeError(name)
