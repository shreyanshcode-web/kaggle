#!/usr/bin/env python3
"""Build a private L4x4 notebook for paired Gemma agent evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from pprint import pformat

HERE = Path(__file__).parent.resolve()
OUT = HERE / "kaggle_eval_notebook"
CODE_FILE = "gemma4_l4x4_agent_ablation_v1.ipynb"
MODEL = "gemma-4-31b-it-qat-w4a16-ct"


def read_agent(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): p.read_text(encoding="utf-8")
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def cell(kind: str, source: str) -> dict:
    value = {
        "id": hashlib.sha1((kind + "\0" + source).encode("utf-8")).hexdigest()[:10],
        "cell_type": kind,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


def main() -> None:
    variants = {
        "lean_v3_8m": read_agent(HERE / "candidate_v2"),
        "lean_v4_6m": read_agent(HERE / "candidate_v4"),
    }

    for label, files in variants.items():
        roots = [p for p in files if p in {"agent.yaml", "agent.yml", "root_agent.yaml", "root_agent.yml"}]
        if len(roots) != 1 or "agent.yaml" not in files:
            raise SystemExit(f"{label}: expected exactly one root agent.yaml")
        if f"model: {MODEL}" not in files["agent.yaml"]:
            raise SystemExit(f"{label}: required Gemma model alias missing")

    cells = [
        cell("markdown", f"""# Gemma 4 L4x4 paired agent experiment

This private notebook compares the scored V3 candidate with a focused V4 prompt/tool variant on the same eight public development issues using the official competition Gemma 4 31B W4A16 model on Kaggle's four-L4 runtime. V3 has a public score of 0.08. The local development panel is a small tuning signal; it does not guarantee a higher Kaggle score.

The model receives only the issue and repository through the official evaluator. Reference patches are not passed to the agent. No adapter is used because a current competition report found that patched vLLM can load Gemma 4 LoRA weights while applying no logit change. Internet is disabled.
"""),
        cell("code", """from pathlib import Path
import glob, importlib, json, os, shutil, subprocess, sys

os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS'] = '1'
os.environ['VLLM_ENGINE_READY_TIMEOUT_S'] = '1200'
os.environ['VLLM_NO_USAGE_STATS'] = '1'
os.environ['OTEL_SDK_DISABLED'] = 'true'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

WORK = Path('/kaggle/working')
DATA_DIR = Path('/kaggle/input/competitions/gemma-4-developer-agent')
WHEELHOUSE_DIR = Path('/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse')
MODEL_PATH = Path('/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2')
assert (DATA_DIR / 'tasks.jsonl').is_file(), DATA_DIR
assert MODEL_PATH.exists(), MODEL_PATH
assert WHEELHOUSE_DIR.is_dir(), WHEELHOUSE_DIR

gpu = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                     capture_output=True, text=True, check=True).stdout.splitlines()
print('Visible GPUs:', gpu)
assert len(gpu) == 4 and all('L4' in name for name in gpu), gpu

# Install the competition's pinned offline wheelhouse, following the official starter.
for pattern in ('/usr/local/lib/python*/dist-packages/*cutlass*.pth',
                '/usr/local/lib/python*/site-packages/*cutlass*.pth'):
    for pth in glob.glob(pattern):
        try:
            os.unlink(pth)
        except OSError:
            pass
tmp_whl = Path('/tmp/gemma4-wheelhouse')
tmp_whl.mkdir(parents=True, exist_ok=True)
for wheel in WHEELHOUSE_DIR.glob('*.whl'):
    if 'cutlass' in wheel.name.lower():
        continue
    renamed = wheel.name.replace('cu128', '+cu128') if ('cu128' in wheel.name and '+' not in wheel.name) else wheel.name
    link = tmp_whl / renamed
    if not link.exists():
        link.symlink_to(wheel)
wheels = sorted(str(p) for p in tmp_whl.glob('*.whl'))
assert wheels, f'No official wheels found in {WHEELHOUSE_DIR}'
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index',
                '--no-deps', '--force-reinstall', *wheels], check=True)
importlib.invalidate_caches()
print('Installed official wheelhouse:', len(wheels), 'wheels')
"""),
        cell("markdown", "## Agent arms\n\nThe first pair isolates the per-task time cap. The lean candidate also removes graph and skill calls, since public inspection reports missing graph/embedding files for the 129 public tasks, and it directs Gemma to attempt an evidence-based source edit earlier. The public scored bundle is included as a reference arm.\n"),
        cell("code", f"""VARIANT_FILES = {pformat(variants, width=100, sort_dicts=True)}
EXPECTED_MODEL = {MODEL!r}
VARIANTS_DIR = WORK / 'agent_variants'
if VARIANTS_DIR.exists():
    shutil.rmtree(VARIANTS_DIR)
VARIANTS_DIR.mkdir(parents=True, exist_ok=True)
for label, files in VARIANT_FILES.items():
    root = VARIANTS_DIR / label
    root.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        path = Path(rel)
        assert not path.is_absolute() and '..' not in path.parts, rel
        dest = root.joinpath(*path.parts)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding='utf-8')
    assert (root / 'agent.yaml').is_file()
    print(label, 'files=', len(files), 'root model=', EXPECTED_MODEL)
"""),
        cell("markdown", "## Start the official local Gemma runtime\n\nAll candidates use the competition's required model alias; vLLM uses tensor parallelism across all four L4 GPUs.\n"),
        cell("code", """import litellm, torch
from adk_submission import VllmConfig, VllmServer, discover_adapters
from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS
from swegemma.models.discovery import validate_single_declared_model

litellm.drop_params = True
MODEL_ALIAS = 'gemma-4-31b-it-qat-w4a16-ct'
for label in VARIANT_FILES:
    assert validate_single_declared_model(VARIANTS_DIR / label) == MODEL_ALIAS, label
    discovered = discover_adapters(str(VARIANTS_DIR / label),
                                   adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    assert not discovered.adapters, (label, sorted(discovered.adapters))

vllm_cfg = VllmConfig(
    model=str(MODEL_PATH), port=8000, host='127.0.0.1',
    tool_call_parser='gemma4', reasoning_parser='gemma4',
    max_model_len=32768,
    dtype='bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'auto',
    gpu_memory_utilization=0.90, enable_auto_tool_choice=True,
    enable_lora=False, tensor_parallel_size=4, startup_timeout=60 * 20,
)
server_instance = VllmServer(vllm_cfg)
server_instance.start()
models = server_instance.create_model_registry(
    aliases=[MODEL_ALIAS], model_prefix='openai/', api_key='EMPTY')
print('Official model ready on four L4s:', server_instance.base_url)
"""),
        cell("markdown", "## Paired public-task evaluation\n\nEight fixed public issues are sampled deterministically across repositories. Each arm gets the same issue IDs and official two-phase patch verification. The result table reports verified resolutions, infrastructure failures, accidental test edits, tool calls, and elapsed time.\n"),
        cell("code", """import asyncio, concurrent.futures, random, re, time, yaml
from collections import defaultdict
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps._configs import EventsCompactionConfig
from swegemma.config import EvalConfig, build_submission_limits
from swegemma.evaluate import Evaluator
from swegemma.models import load_tasks

TASKS_PATH = DATA_DIR / 'tasks.jsonl'
tasks = load_tasks(TASKS_PATH)
def choose_public_panel(all_tasks, n=8, seed=4207):
    buckets = defaultdict(list)
    for task in all_tasks:
        buckets[task.repo].append(task)
    repos = sorted(buckets)
    shuffled = {}
    for repo in repos:
        items = sorted(buckets[repo], key=lambda t: t.instance_id)
        rng = random.Random(f'{seed}:{repo}')
        shuffled[repo] = rng.sample(items, len(items))
    selected, offsets = [], {repo: 0 for repo in repos}
    for repo in repos:
        if len(selected) < n and shuffled[repo]:
            selected.append(shuffled[repo][0]); offsets[repo] = 1
    while len(selected) < min(n, len(all_tasks)):
        advanced = False
        for repo in repos:
            if offsets[repo] < len(shuffled[repo]) and len(selected) < n:
                selected.append(shuffled[repo][offsets[repo]])
                offsets[repo] += 1; advanced = True
        if not advanced:
            break
    return selected

PANEL = choose_public_panel(tasks, n=8)
PANEL_IDS = [t.instance_id for t in PANEL]
print('Fixed public panel:', [(t.instance_id, t.repo) for t in PANEL])
assert len(PANEL_IDS) == len(set(PANEL_IDS))
limits, gen_constraints = build_submission_limits()
(WORK / 'agent_eval').mkdir(parents=True, exist_ok=True)
(WORK / 'agent_eval' / 'partial_results.jsonl').write_text('', encoding='utf-8')

def run_sync(coro_or_fn, *args, **kwargs):
    fn = (lambda: coro_or_fn(*args, **kwargs)) if callable(coro_or_fn) else (lambda: coro_or_fn)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(fn())).result()
    return asyncio.run(fn())

def patch_paths(patch):
    return [b for a, b in re.findall(r'^diff --git a/(.*?) b/(.*?)$', patch or '', re.MULTILINE)]

def is_test_path(name):
    p = Path(name)
    return ('tests' in p.parts or p.name.startswith('test_') or
            p.name.endswith('_test.py') or p.name in {'conftest.py', 'pytest.ini'})

def is_infrastructure_error(message):
    return bool(re.search(
        r'SubmissionValidationError|MissingRootConfigError|MultipleRootConfigsError|'
        r'Failed to load skill|ModuleNotFoundError|ConnectionRefusedError|'
        r'CUDA out of memory|Sandbox initialization failed|model (?:was )?not found',
        message or '', re.IGNORECASE))

def evaluate_arm(label):
    root = VARIANTS_DIR / label
    section = yaml.safe_load((root / 'eval_config.yaml').read_text())['evaluation']
    cfg = EvalConfig(
        tasks_path=TASKS_PATH, snapshots_dir=DATA_DIR / 'snapshots',
        results_dir=WORK / 'agent_eval' / label, submission_dir=root,
        models=models, sandbox='subprocess', timeout_seconds=int(section['timeout_seconds']),
        max_time_minutes=float(section['max_time_minutes']),
        max_tool_calls=int(section['max_tool_calls']), max_turns=int(section['max_turns']),
        limits=limits, generation_constraints=gen_constraints,
        context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
        events_compaction_config=EventsCompactionConfig(compaction_interval=15, overlap_size=2,
            token_threshold=14336, event_retention_size=5),
        graph_dir=str(DATA_DIR / 'graphs'), embeddings_dir=str(DATA_DIR / 'embeddings'),
        wheels_dir=DATA_DIR / 'wheels', verbose=False, display_mode='quiet', concurrency=1,
    )
    evaluator = Evaluator(cfg)
    records = []
    for i, task in enumerate(PANEL, 1):
        start = time.monotonic()
        try:
            result = run_sync(evaluator.evaluate_task, task=task, task_index=i, total_tasks=len(PANEL))
            patch = result.agent_patch or ''
            changed = patch_paths(patch)
            problem = (str(getattr(result, 'agent_error', '') or '') + ' ' +
                       str(getattr(result, 'error', '') or '')).strip()[:600]
            row = {
                'arm': label, 'id': task.instance_id, 'repo': task.repo,
                'resolved': bool(result.resolved), 'patch_chars': len(patch),
                'tool_calls': int(getattr(result, 'tool_calls', 0) or 0),
                'duration_seconds': round(float(getattr(result, 'duration_seconds', time.monotonic()-start) or 0), 1),
                'test_files_touched': int(any(is_test_path(p) for p in changed)),
                'test_exit_code': getattr(result, 'test_exit_code', None),
                'error': problem,
                'infra_error': is_infrastructure_error(problem),
            }
        except Exception as exc:
            problem = f'{type(exc).__name__}: {exc}'[:600]
            row = {'arm': label, 'id': task.instance_id, 'repo': task.repo,
                   'resolved': False, 'patch_chars': 0, 'tool_calls': 0,
                   'duration_seconds': round(time.monotonic()-start, 1),
                   'test_files_touched': 0, 'test_exit_code': None,
                   'error': problem, 'infra_error': True}
        records.append(row)
        print(label, i, '/', len(PANEL), row, flush=True)
        log = WORK / 'agent_eval' / 'partial_results.jsonl'
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + '\\n')
    return records

ARMS = ['lean_v3_8m', 'lean_v4_6m']
RESULTS = {}
for arm in ARMS:
    RESULTS[arm] = evaluate_arm(arm)

def summarize(rows):
    return {
        'resolved': sum(r['resolved'] for r in rows),
        'tasks': len(rows),
        'rate': round(sum(r['resolved'] for r in rows) / max(len(rows), 1), 3),
        'test_files_touched': sum(r['test_files_touched'] for r in rows),
        'infra_errors': sum(r['infra_error'] for r in rows),
        'mean_seconds': round(sum(r['duration_seconds'] for r in rows) / max(len(rows), 1), 1),
        'mean_tool_calls': round(sum(r['tool_calls'] for r in rows) / max(len(rows), 1), 1),
    }

SUMMARY = {arm: summarize(rows) for arm, rows in RESULTS.items()}
print('SUMMARY:', json.dumps(SUMMARY, indent=2))
"""),
        cell("markdown", "## Choose and package the measured winner\n\nThe package selector uses only local public-task verification outcomes. It excludes genuine infrastructure/configuration errors and test-file edits, but treats model context or turn-budget exhaustion as agent failures. It then chooses the highest resolved count; ties prefer the lean agent, then the shorter runtime. This panel is a development comparison, not Kaggle's hidden public score.\n"),
        cell("code", """eligible = [arm for arm in ARMS
            if SUMMARY[arm]['infra_errors'] == 0 and SUMMARY[arm]['test_files_touched'] == 0]
if not eligible:
    raise RuntimeError('All agent arms had infrastructure errors or touched test files; inspect the per-task records.')
tie_order = {'lean_v4_6m': 0, 'lean_v3_8m': 1}
SELECTED = max(eligible, key=lambda arm: (
    SUMMARY[arm]['resolved'], -SUMMARY[arm]['mean_seconds'], -tie_order[arm]))
print('Selected arm:', SELECTED, SUMMARY[SELECTED])

import hashlib, zipfile
selected_files = VARIANT_FILES[SELECTED]
zip_path = WORK / 'submission.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for rel, content in sorted(selected_files.items()):
        archive.writestr(rel, content.encode('utf-8'))
with zipfile.ZipFile(zip_path) as archive:
    assert 'agent.yaml' in archive.namelist() and archive.testzip() is None
    assert len([p for p in archive.namelist() if p in
                {'agent.yaml','agent.yml','root_agent.yaml','root_agent.yml'}]) == 1
experiment = {
    'model': MODEL_ALIAS, 'machine': '4x NVIDIA L4',
    'public_panel_ids': PANEL_IDS, 'arms': SUMMARY, 'selected_arm': SELECTED,
    'submission_zip_sha256': hashlib.sha256(zip_path.read_bytes()).hexdigest(),
    'caveat': 'Eight public development issues are a small local comparison, not Kaggle hidden/public leaderboard score.',
}
(WORK / 'experiment_card.json').write_text(json.dumps(experiment, indent=2) + '\\n', encoding='utf-8')
print('Created:', zip_path, 'bytes=', zip_path.stat().st_size)
print('SHA256:', experiment['submission_zip_sha256'])
"""),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
            "kaggle": {"accelerator": "NvidiaL4", "isInternetEnabled": False},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / CODE_FILE).write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    metadata = {
        "id": "shreyanshsingh7777/gemma-4-l4x4-agent-ablation-public-dev-pilot",
        "title": "Gemma 4 L4x4 Agent Ablation - Public Dev Pilot",
        "code_file": CODE_FILE,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": ["metric/gemma-4-developer-agent-wheelhouse"],
        "kernel_sources": [],
        "competition_sources": ["gemma-4-developer-agent"],
        "model_sources": ["google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2"],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
        "machine_shape": "NvidiaL4",
    }
    (OUT / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("Wrote private L4x4 evaluation notebook:", OUT)


if __name__ == "__main__":
    main()
