#!/usr/bin/env python3
"""Generate a Kaggle competition notebook that builds the starter ZIP."""

from __future__ import annotations

import json
import hashlib
import argparse
from pathlib import Path
from pprint import pformat


HERE = Path(__file__).parent.resolve()
SUBMISSION = HERE / "submission"
NOTEBOOK_DIR = HERE / "kaggle_notebook"
CODE_FILE = "gemma4_agent_baseline_v1.ipynb"


def cell(cell_type: str, source: str) -> dict:
    return {
        "id": hashlib.sha1((cell_type + "\0" + source).encode("utf-8")).hexdigest()[:10],
        "cell_type": cell_type,
        "metadata": {},
        "source": source.splitlines(keepends=True),
        **({"execution_count": None, "outputs": []} if cell_type == "code" else {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-dir", type=Path, default=SUBMISSION)
    parser.add_argument("--notebook-dir", type=Path, default=NOTEBOOK_DIR)
    parser.add_argument("--kernel-id", default="shreyanshsingh7777/gemma-4-developer-agent-baseline-zip-v1")
    parser.add_argument("--title", default="Gemma 4 Developer Agent - Baseline ZIP v1")
    parser.add_argument("--code-file", default=CODE_FILE)
    parser.add_argument("--cpu-only", action="store_true", help="Run the ZIP builder without requesting a GPU")
    args = parser.parse_args()
    submission_dir = args.submission_dir.resolve()
    notebook_dir = args.notebook_dir.resolve()

    files = {
        path.relative_to(submission_dir).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(submission_dir.rglob("*"))
        if path.is_file()
    }
    if "agent.yaml" not in files:
        raise SystemExit("submission/agent.yaml is required")
    file_literal = pformat(files, width=110, sort_dicts=True)

    cells = [
        cell("markdown", """# Gemma 4 Developer Agent package\n\nThis competition notebook writes a validated agent ZIP to `/kaggle/working/submission.zip`, with `agent.yaml` at the archive root. It embeds only the selected agent files; it does not use local issue solutions, hidden labels, or answer patches. The competition scoring harness supplies the required Gemma 4 model when it evaluates the ZIP.\n\nRun all cells, then submit this notebook version's generated ZIP.\n"""),
        cell("code", """from pathlib import Path\nimport json, os, platform, subprocess\n\nWORK = Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path.cwd()\nCOMP = Path('/kaggle/input/competitions/gemma-4-developer-agent')\nprint('Working directory:', WORK)\nprint('Python:', platform.python_version())\ntry:\n    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True, capture_output=True, timeout=10)\n    print('Visible GPUs:', gpu.stdout.strip() or gpu.stderr.strip() or 'none')\nexcept Exception as exc:\n    print('GPU inventory unavailable:', type(exc).__name__)\ntasks = COMP / 'tasks.jsonl'\nif tasks.is_file():\n    with tasks.open(encoding='utf-8') as stream:\n        count = sum(1 for line in stream if line.strip())\n    print('Public task rows:', count)\nelse:\n    print('Public tasks.jsonl is not mounted; ZIP build does not depend on it.')\n"""),
        cell("markdown", "## Build a clean, root-layout submission archive\n\nThe files below are embedded from the selected agent directory. The ZIP writer uses only that explicit file map, so notebook outputs, logs, training scripts, or inspection wheels cannot leak into the competition archive.\n"),
        cell("code", f"FILES = {file_literal}\n"),
        cell("code", """import re, zipfile\nfrom pathlib import PurePosixPath\n\nEXPECTED_MODEL = 'gemma-4-31b-it-qat-w4a16-ct'\nALLOWED = {'.yaml', '.yml', '.md', '.txt', '.py', '.json', '.safetensors'}\nassert 'agent.yaml' in FILES and len([p for p in FILES if p in {'agent.yaml','agent.yml','root_agent.yaml','root_agent.yml'}]) == 1\nfor rel, content in FILES.items():\n    path = PurePosixPath(rel)\n    assert not path.is_absolute() and '..' not in path.parts, rel\n    assert path.suffix.lower() in ALLOWED, rel\n    if path.suffix.lower() == '.py':\n        assert 'skills' in path.parts, rel\n    if path.suffix.lower() in {'.yaml', '.yml'}:\n        for model in re.findall(r'^\\s*model:\\s*([^\\s#]+)', content, re.M):\n            assert model == EXPECTED_MODEL, (rel, model)\n        for included in re.findall(r'!include\\s+([^\\s#]+)', content):\n            target = (PurePosixPath(rel).parent / included)\n            target = PurePosixPath(*[p for p in target.parts if p not in {'.', ''}])\n            assert '..' not in target.parts and target.as_posix() in FILES, (rel, included)\n\nzip_path = WORK / 'submission.zip'\nwith zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:\n    for rel, content in sorted(FILES.items()):\n        archive.writestr(rel, content.encode('utf-8'))\nwith zipfile.ZipFile(zip_path) as archive:\n    names = archive.namelist()\n    assert 'agent.yaml' in names and not any(n.startswith('../') for n in names)\n    assert archive.testzip() is None\n    assert set(names) == set(FILES)\nprint('Created:', zip_path)\nprint('Archive bytes:', zip_path.stat().st_size)\nprint('Root agent present:', 'agent.yaml' in names)\nprint('Files:', len(names))\n"""),
        cell("markdown", "## Next step\n\nSubmit this notebook version's `submission.zip` as the competition artifact. The package uses the competition's required Gemma 4 checkpoint and includes no adapter.\n"),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
            "kaggle": {"isInternetEnabled": False, **({} if args.cpu_only else {"accelerator": "NvidiaL4"})},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_dir.mkdir(parents=True, exist_ok=True)
    (notebook_dir / args.code_file).write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    metadata = {
        "id": args.kernel_id,
        "title": args.title,
        "code_file": args.code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": not args.cpu_only,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": ["gemma-4-developer-agent"],
        "model_sources": [],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
    }
    if not args.cpu_only:
        metadata["machine_shape"] = "NvidiaL4"
    (notebook_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote notebook and metadata to {notebook_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
