# Gemma 4 Developer Agent starter

The Kaggle API check succeeded. The account is entered in `gemma-4-developer-agent` (`userHasEntered=True`). The folder contains adapter-free submission candidates, the official `HARNESS_README.md`, and research-only rollout/training helpers.

## Build locally

```powershell
python build_submission.py --check-only
python build_submission.py
```

The ZIP puts `agent.yaml` at its root. Research scripts and inspection wheels are outside `submission/` and are never packaged.

## Score diagnosis and GPU evaluation

The original baseline scored `0.00` because the skill manifest failed compilation before Gemma inference. The competition requires a lowercase kebab-case skill name that also matches the skill directory. Both the skill folder and `agent.yaml` reference are now `repo-navigation`, and the ZIP validator checks the directory/name match. The V3 public submission scored `0.08`.

A private Kaggle notebook compares the scored V3 agent with a focused V4 prompt/tool revision on the same eight public development tasks. It attaches the official wheelhouse and the required `gemma-4-31b-it-qat-w4a16-ct` model, uses four L4 GPUs, and disables internet. The V4 run is queued here: [Kaggle notebook](https://www.kaggle.com/code/shreyanshsingh7777/gemma-4-l4x4-agent-ablation-public-dev-pilot). The public-task panel is a development signal; the competition score remains the final measure.

The V3 candidate removes skill and graph-tool configuration and uses the required Gemma 4 QAT model. V4 removes the subagent, status, and file-writing tools, prohibits duplicate reads/searches and scratch files, and forces an earlier edit and one-time patch submission. No adapter is packaged.

## Initial settings

- Four agent minutes/task, 28 tool calls, 24 turns, and 120 seconds/command. Recompute the task cap using actual task count and a measured p95 sandbox setup time before a full 12-hour run.
- The root agent calls its read-only localizer only on ambiguous multi-file issues.
- No LoRA is included. A Kaggle discussion posted around October 1 reported adapters loading in patched vLLM but producing identical output; prove a nonzero logit delta on the exact W4A16 model before training or packaging an adapter.
- The rollout script admits only explicitly labeled public-train results and excludes supplied holdout IDs. The training script refuses to start until `--gate-passed` is set and currently supports bf16 training only.

## Before a full scored run

Attach the competition data and wheelhouse to a competition notebook, keep internet disabled, and use the competition's L4x4 runtime. Confirm the official compiler accepts this ZIP, run a small fixed public-dev holdout, and measure setup time before committing to a 12-hour evaluation.

The competition README documents nine workspace tools but not the runtime invocation schema for `run_skill_script`/`load_skill_resource`; the bundled skill helper is optional until that integration is confirmed.
