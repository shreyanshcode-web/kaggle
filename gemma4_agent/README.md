# Gemma 4 Developer Agent starter

The Kaggle API check succeeded. The account is entered in `gemma-4-developer-agent` (`userHasEntered=True`). The folder contains an adapter-free v1 submission, a Kaggle notebook that builds its ZIP, the official `HARNESS_README.md`, and research-only rollout/training helpers.

## Build locally

```powershell
python build_submission.py --check-only
python build_submission.py
```

The ZIP puts `agent.yaml` at its root. Research scripts and inspection wheels are outside `submission/` and are never packaged.

## Initial settings

- Four agent minutes/task, 28 tool calls, 24 turns, and 120 seconds/command. Recompute the task cap using actual task count and a measured p95 sandbox setup time before a full 12-hour run.
- The root agent calls its read-only localizer only on ambiguous multi-file issues.
- No LoRA is included. A Kaggle discussion posted around October 1 reported adapters loading in patched vLLM but producing identical output; prove a nonzero logit delta on the exact W4A16 model before training or packaging an adapter.
- The rollout script admits only explicitly labeled public-train results and excludes supplied holdout IDs. The training script refuses to start until `--gate-passed` is set and currently supports bf16 training only.

## Before a full scored run

Attach the competition data and wheelhouse to a competition notebook, keep internet disabled, and use the competition's L4x4 runtime. Confirm the official compiler accepts this ZIP, run a small fixed public-dev holdout, and measure setup time before committing to a 12-hour evaluation.

The competition README documents nine workspace tools but not the runtime invocation schema for `run_skill_script`/`load_skill_resource`; the bundled skill helper is optional until that integration is confirmed.
