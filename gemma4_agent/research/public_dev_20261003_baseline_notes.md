# Public development panel notes

This is a diagnostic record from the earlier L4x4 notebook version. It is not the repaired-agent result or a leaderboard score.

The eight fixed issues were `rich_3470`, `httpx_3672`, `fastapi_14986`, `requests_7502`, `rich_3506`, `fastapi_14371`, `requests_6644`, and `rich_3953`.

| Arm | Resolved | Main failure pattern |
| --- | ---: | --- |
| Original baseline | 0/8 | Every task failed before model inference because `repo_navigation` used an underscore in its skill name. |
| Budget-only variant | 0/8 | Same skill-manifest compile error. |
| Public reference agent | 0/8 | Four patches failed official tests, three exhausted the 40-turn budget, and one exceeded the model context window. |
| Lean Gemma v2 | 1/8 | One task passed; several exhausted the turn budget, two produced no patch, one edited a test file, and one patch failed tests. |

The follow-up lean candidate addresses observed failure modes: it has no skill or graph-tool dependency, a 4,096-token output cap to leave room under the 32,768-token context limit, a 24-turn/28-tool cap, explicit early production edits, read-only tests, test-path restoration instructions, and an early `submit_patch()` deadline. The repaired baseline skill uses the required kebab-case name. A new four-L4 run is needed before selecting a scored submission.
