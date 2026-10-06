#!/usr/bin/env python3
"""Validate a competition submission directory and create a root-layout ZIP."""

from __future__ import annotations

import argparse
import os
import re
import stat
import sys
import zipfile
from pathlib import Path, PurePosixPath


ALLOWED_EXTENSIONS = {".yaml", ".yml", ".md", ".txt", ".py", ".json", ".safetensors"}
MAX_UNPACKED_BYTES = 3_221_225_472
EXPECTED_MODEL = "gemma-4-31b-it-qat-w4a16-ct"
INCLUDE_RE = re.compile(r"!include\s+([^\s#]+)")
CONFIG_PATH_RE = re.compile(r"^\s*config_path:\s*[\"']?([^\s\"'#]+)", re.MULTILINE)
MODEL_RE = re.compile(r"^\s*model:\s*[\"']?([^\s\"'#]+)", re.MULTILINE)


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def checked_relative(raw: str, base: Path, root: Path, label: str) -> Path:
    raw = raw.strip().strip("\"'").replace("\\", "/")
    rel = PurePosixPath(raw)
    if rel.is_absolute() or not rel.parts or any(part in {"..", ""} for part in rel.parts):
        fail(f"unsafe {label} path: {raw!r}")
    candidate = (base.joinpath(*rel.parts)).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        fail(f"{label} escapes submission root: {raw!r}")
    if not candidate.exists():
        fail(f"missing {label}: {raw!r} (resolved to {candidate})")
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-dir", type=Path, default=Path(__file__).parent / "submission")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "submission.zip")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    root = args.submission_dir.resolve()
    if not root.is_dir():
        fail(f"submission directory does not exist: {root}")
    root_candidates = [
        root / name
        for name in ("agent.yaml", "agent.yml", "root_agent.yaml", "root_agent.yml")
        if (root / name).is_file()
    ]
    if len(root_candidates) != 1 or root_candidates[0].name != "agent.yaml":
        fail("provide exactly one root config, named agent.yaml")

    files: list[Path] = []
    total_bytes = 0
    for current, dirs, names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in list(dirs):
            path = current_path / name
            if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
                fail(f"symlink/junction is not allowed: {path}")
        for name in names:
            path = current_path / name
            if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
                fail(f"symlink/junction is not allowed: {path}")
            rel = path.relative_to(root)
            if path.suffix.lower() not in ALLOWED_EXTENSIONS:
                fail(f"file extension is not allowed: {rel}")
            if path.suffix.lower() == ".py" and "skills" not in rel.parts:
                fail(f"Python files are allowed only inside a skill: {rel}")
            if stat.S_ISLNK(path.stat().st_mode):
                fail(f"symlink is not allowed: {path}")
            files.append(path)
            total_bytes += path.stat().st_size

    if total_bytes >= MAX_UNPACKED_BYTES:
        fail(f"unpacked size must be below 3 GiB (found {total_bytes:,} bytes)")

    yaml_files = [path for path in files if path.suffix.lower() in {".yaml", ".yml"}]
    for yaml_file in yaml_files:
        content = yaml_file.read_text(encoding="utf-8")
        for match in INCLUDE_RE.finditer(content):
            checked_relative(match.group(1), yaml_file.parent, root, "!include")
        for match in CONFIG_PATH_RE.finditer(content):
            checked_relative(match.group(1), yaml_file.parent, root, "config_path")
        for match in MODEL_RE.finditer(content):
            if match.group(1) != EXPECTED_MODEL:
                fail(f"unsupported model {match.group(1)!r} in {yaml_file.relative_to(root)}")

        for match in re.finditer(r"(?m)^\s*skills:\s*\n((?:[ \t]+-[ \t]+[^\n]+\n?)*)", content):
            for skill_path in re.findall(r"(?m)^\s+-\s+([^\s#]+)", match.group(1)):
                checked_relative(skill_path, yaml_file.parent, root, "skill")

    skill_manifests = list(root.glob("skills/**/SKILL.md"))
    for manifest in skill_manifests:
        frontmatter = manifest.read_text(encoding="utf-8").split("---", 2)
        name_match = (
            re.search(r"(?m)^name:\s*([a-z0-9]+(?:-[a-z0-9]+)*)\s*$", frontmatter[1])
            if len(frontmatter) >= 3
            else None
        )
        if name_match is None:
            fail(f"SKILL.md name must use lowercase kebab-case: {manifest.relative_to(root)}")
        if name_match.group(1) != manifest.parent.name:
            fail(
                f"SKILL.md name must match its directory ({manifest.parent.name!r}): "
                f"{manifest.relative_to(root)}"
            )

    try:
        import yaml

        class IncludeLoader(yaml.SafeLoader):
            pass

        IncludeLoader.add_constructor("!include", lambda loader, node: loader.construct_scalar(node))
        for yaml_file in yaml_files:
            yaml.load(yaml_file.read_text(encoding="utf-8"), Loader=IncludeLoader)
    except ImportError:
        print("NOTE: PyYAML is unavailable locally; syntax parsing will occur in Kaggle's compiler.")
    except Exception as exc:
        fail(f"YAML syntax check failed: {exc}")

    if args.check_only:
        print(f"OK: {len(files)} files, {total_bytes:,} unpacked bytes, one root agent.yaml")
        return 0

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(root).as_posix())
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        if "agent.yaml" not in names:
            fail("ZIP validation failed: agent.yaml is not at the archive root")
        if any(name.startswith("submission/") or name.startswith("../") for name in names):
            fail("ZIP validation failed: archive paths are not rooted at agent.yaml")
        if archive.testzip() is not None:
            fail("ZIP validation failed: CRC error")
    print(f"Created {output} ({output.stat().st_size:,} bytes); agent.yaml is at ZIP root")
    return 0


if __name__ == "__main__":
    sys.exit(main())
