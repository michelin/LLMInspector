#!/usr/bin/env bash
# Stop — run the suite before Claude hands the turn back.
#
# 552 tests in ~7s, so this is cheap enough to run unconditionally on any turn
# that touched source or tests. Passing is silent: the model never spends a tool
# call on `pytest` and never reads 552 dots. Failing blocks the stop and hands
# back only the failure tail.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 0

input=$(cat)
# Set once Claude is already responding to this hook — without the guard a
# persistently failing suite would loop forever.
[[ "$(jq -r '.stop_hook_active // false' <<<"$input")" == "true" ]] && exit 0

[[ -x "$ROOT/.venv/bin/python" ]] || exit 0

# Nothing to check if this turn left llminspector/ and tests/ untouched.
changed=$(git status --porcelain -- llminspector tests 2>/dev/null)
[[ -z "$changed" ]] && exit 0

out=$(.venv/bin/python -m pytest -q --no-cov --tb=short -p no:cacheprovider 2>&1) && exit 0

echo "Test suite FAILED (Stop hook). Fix before finishing:" >&2
# Drop the progress-dot lines so the 40-line budget is all signal.
grep -vE '^[.sFEx]+ *\[ *[0-9]+%\]$' <<<"$out" | tail -n 40 >&2
exit 2
