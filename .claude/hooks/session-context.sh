#!/usr/bin/env bash
# SessionStart — emit the orientation facts Claude would otherwise spend three
# tool calls discovering. stdout is added to the session context.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 0

echo "branch: $(git branch --show-current 2>/dev/null || echo '?')"

dirty=$(git status --porcelain 2>/dev/null)
if [[ -z "$dirty" ]]; then
    echo "worktree: clean"
else
    echo "worktree: dirty"
    head -n 15 <<<"$dirty"
fi

if [[ -x "$ROOT/.venv/bin/python" ]]; then
    echo "venv: .venv ($("$ROOT/.venv/bin/python" -V 2>&1))"
else
    echo "venv: MISSING — create with 'uv venv --python 3.12 .venv'"
fi
exit 0
