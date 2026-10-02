#!/usr/bin/env python3
"""Turn resolved public swegemma runs into Gemma 4 tool-call SFT conversations.

Input artifacts are `tasks.jsonl`, `task_results.jsonl`, `patches/`, and
`traces/` from a public `swegemma eval --results-dir ...` run. This script
never runs a model or reads hidden evaluation data.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


ALLOWED_TOOLS = {
    "run_command", "submit_patch", "get_status", "read_file", "edit_file",
    "write_file", "get_code_neighbors", "search_similar_code", "get_code_subgraph",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                rows.append(value)
    return rows


def row_id(row: dict[str, Any]) -> str:
    for key in ("instance_id", "task_id", "id"):
        if row.get(key) is not None:
            return str(row[key])
    raise ValueError("task/result row lacks instance_id, task_id, or id")


def safe_id(value: str) -> str:
    # Matches swegemma.harness.verification.save_trace_artifact.
    return value.replace("/", "__")


def parse_response(value: str) -> Any:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return str(value)


def build_messages(task: dict[str, Any], trace: dict[str, Any], system: str) -> list[dict[str, Any]] | None:
    problem = str(task.get("problem_statement", task.get("problem", ""))).strip()
    if not problem:
        return None
    repo = str(task.get("repo", "unknown repository"))
    hints = str(task.get("hints_text", task.get("hints", ""))).strip()
    user = f"Repository: {repo}\n\nProblem statement:\n{problem}"
    if hints:
        user += f"\n\nHints:\n{hints}"
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    entries = trace.get("entries", [])
    if not isinstance(entries, list):
        return None

    pending: list[tuple[str, dict[str, Any]]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        event_type = entry.get("type", entry.get("event_type"))
        if event_type == "tool_call":
            name = str(entry.get("tool", entry.get("tool_name", "")))
            arguments = entry.get("args", entry.get("tool_args", {}))
            if name not in ALLOWED_TOOLS or not isinstance(arguments, dict):
                return None
            pending.append((name, arguments))
        elif event_type == "tool_response":
            name = str(entry.get("tool", entry.get("tool_name", "")))
            result = entry.get("result", entry.get("tool_result", ""))
            match_index = next((i for i, pair in enumerate(pending) if pair[0] == name), None)
            if match_index is None:
                return None
            call_name, arguments = pending.pop(match_index)
            # Google's Gemma 4 template expects mapping arguments and places
            # the response in the same assistant message as tool_responses.
            messages.append({
                "role": "assistant",
                "tool_calls": [{"function": {"name": call_name, "arguments": arguments}}],
                "tool_responses": [{"name": call_name, "response": parse_response(result)}],
            })
        elif event_type in {"text", "final"}:
            text = str(entry.get("content", "")).strip()
            if text:
                messages.append({"role": "assistant", "content": text})
        # Deliberately omit private model-thought events from the SFT data.

    if pending or not any(message.get("tool_calls") for message in messages):
        return None
    return messages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True, type=Path, help="public training tasks.jsonl")
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--system-prompt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--holdout-ids", type=Path, help="newline-separated public holdout IDs to exclude")
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--split", choices=("public-train",), required=True,
                        help="intentionally refuses private/hidden splits")
    args = parser.parse_args()

    holdout = set()
    if args.holdout_ids:
        holdout = {line.strip() for line in args.holdout_ids.read_text(encoding="utf-8").splitlines() if line.strip()}
    tasks = {row_id(row): row for row in load_jsonl(args.tasks)}
    results = load_jsonl(args.results_dir / "task_results.jsonl")
    system = args.system_prompt.read_text(encoding="utf-8").strip()
    candidates = []
    for result in results:
        ident = row_id(result)
        if ident in holdout or result.get("resolved") is not True or ident not in tasks:
            continue
        stem = safe_id(ident)
        patch = args.results_dir / "patches" / f"{stem}.patch"
        trace_path = args.results_dir / "traces" / f"trace_{stem}.json"
        if not patch.is_file() or not patch.read_text(encoding="utf-8", errors="replace").strip() or not trace_path.is_file():
            continue
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        messages = build_messages(tasks[ident], trace, system)
        if messages:
            candidates.append({"instance_id": ident, "split": args.split, "messages": messages})

    random.Random(args.seed).shuffle(candidates)
    selected = candidates[: max(0, args.max_examples)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Kept {len(selected)} / {len(candidates)} resolved public traces; excluded {len(holdout)} holdout IDs")
    print("Review one rendered example with the Gemma 4 processor before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
