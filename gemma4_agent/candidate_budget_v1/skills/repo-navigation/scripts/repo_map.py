#!/usr/bin/env python3
"""Print a compact, read-only Python repository map (standard library only)."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "build", "dist"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--limit", type=int, default=80)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    paths = []
    for path in root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        paths.append(path)
    paths.sort(key=lambda p: str(p).lower())

    for path in paths[: max(0, args.limit)]:
        rel = path.relative_to(root).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError, UnicodeError):
            print(f"{rel}: <unreadable or syntax error>")
            continue
        symbols = [
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        suffix = ", ".join(symbols[:12])
        if len(symbols) > 12:
            suffix += ", …"
        print(f"{rel}: {suffix or '<no top-level symbols>'}")

    if len(paths) > args.limit:
        print(f"... {len(paths) - args.limit} more Python files omitted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
