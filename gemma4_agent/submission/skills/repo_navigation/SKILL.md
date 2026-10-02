---
name: repo_navigation
description: Efficient, low-noise repository navigation for software issue repair.
---

# Repository navigation

Use the supplied workspace layout first. Search for the issue's distinctive terms and likely API names with `rg`, inspect the smallest relevant file ranges, and follow the call path only as far as needed.

For a symbol-level question, use graph tools only after resolving a candidate symbol from repository text:

- `search_similar_code(query, k=5)`: pass a class, function, or module symbol such as `parse_header`, never a prose issue description.
- `get_code_neighbors(node, max_neighbors=12)`: inspect the direct callers/callees/definitions of a likely symbol.
- `get_code_subgraph(nodes)`: use only for 2–5 known symbols when their connections affect the bug.

Treat graph results as navigation hints. Confirm every conclusion in source code and tests. If graph data is missing, use `rg` and `read_file` without retrying the same graph query.

The bundled `scripts/repo_map.py` is a read-only helper for local exploration. [VERIFY: the competition's runtime skill-script invocation is not specified in HARNESS_README.md; do not spend task budget trying an unknown invocation.]
