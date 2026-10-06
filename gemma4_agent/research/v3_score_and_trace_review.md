# V3 score and L4 trace review

## Public result

Kaggle submission `56785318` completed on 2026-10-03 with a public score of `0.08`, up from `0.00` on V1. V3 used the required `gemma-4-31b-it-qat-w4a16-ct` model and has no skill, graph, or adapter dependency.

## Four-L4 public development panel

V3 resolved 1/8 tasks (12.5%), touched no test files, averaged 15.1 tool calls, and averaged 59 seconds per task. The public reference arm also resolved 1/8, averaging 28.2 tool calls and 92.3 seconds.

V3's only resolved task was `requests_6644`, which trims repeated leading slashes from a URL. The other seven outcomes were:

- `rich_3470`, `fastapi_14986`, and `fastapi_14371`: no submitted patch before turn exhaustion.
- `requests_7502` and `rich_3506`: completed without submitting a patch.
- `httpx_3672`: produced a 2,148-character patch; official verification failed with exit code 2. It used repeated `submit_patch()` calls after the first successful submit.
- `rich_3953`: produced a 620-character patch that failed verification and exhausted turns. It made malformed shell calls for a tool, ran an analyzer request that failed, and wrote a scratch reproduction file into the repository.

The traces show repeated identical file reads and searches on several tasks. One FastAPI run reread the same broad source range repeatedly and never edited code. The V4 comparison removes `write_file`, `get_status`, and the analyzer subagent; it forbids duplicate reads/searches and repository scratch files; it asks for the first edit by tool call 5 and submission by call 10. It compares V3 and V4 on the same eight issues using four L4 GPUs.

Kaggle version 5 compares V3 with the exact V4 settings used in the public package: edit by call 4, submit by call 8, and cap at 12 turns.

## Baseline manifest fix

The original and budget-only agents failed to compile because their skill manifest name `repo-navigation` did not match the directory `repo_navigation`. The skill folder and `agent.yaml` paths now both use `repo-navigation`, and `build_submission.py` validates the equality.
