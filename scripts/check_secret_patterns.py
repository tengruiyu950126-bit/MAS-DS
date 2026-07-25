"""Offline tracked-text credential pattern check with redacted output."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "OpenAI-style token": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "GitHub-style token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
}
TEXT_SUFFIXES = {".py", ".md", ".toml", ".json", ".cff", ".txt", ".yml", ".yaml"}


def main() -> int:
    tracked = subprocess.check_output(
        ["git", "ls-files"],
        text=True,
        encoding="utf-8",
    ).splitlines()
    findings: list[str] = []
    for name in tracked:
        path = Path(name)
        if not path.exists() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), 1):
            for secret_type, pattern in PATTERNS.items():
                if pattern.search(line):
                    findings.append(f"{name}:{line_number}: possible {secret_type}")
    if findings:
        print("\n".join(findings))
        return 1
    print("No configured credential pattern was found in tracked text.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
