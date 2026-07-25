"""Fail when tracked project text contains known UTF-8 mojibake markers."""

from __future__ import annotations

import subprocess
from pathlib import Path


MOJIBAKE_MARKERS = (
    "\u00c3\u00a2",
    "\u00c2\u00b7",
    "\u00e2\u20ac",
    "\u00e2\u2020",
    "\u00e2\u201d",
    "\ufffd",
)
TEXT_SUFFIXES = {".py", ".md", ".toml", ".json", ".cff", ".txt", ".yml", ".yaml"}


def main() -> int:
    tracked = subprocess.check_output(
        ["git", "ls-files"],
        text=True,
        encoding="utf-8",
    ).splitlines()
    failures: list[str] = []
    for name in tracked:
        path = Path(name)
        if not path.exists() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            failures.append(f"{name}: not valid UTF-8")
            continue
        if any(marker in text for marker in MOJIBAKE_MARKERS):
            failures.append(f"{name}: contains a known mojibake marker")
    if failures:
        print("\n".join(failures))
        return 1
    print(f"Checked {len(tracked)} tracked paths: text encoding is valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
