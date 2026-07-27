#!/usr/bin/env bash
# PreToolUse(Bash) — keep every Python invocation inside the project venv.
#
# The venv (.venv, Python 3.12, built with uv) has NO pip, and the system
# interpreter is 3.14 where this package does not install. Both mistakes fail
# slowly and confusingly; catching them here costs nothing.
set -uo pipefail

input=$(cat)
cmd=$(jq -r '.tool_input.command // empty' <<<"$input" 2>/dev/null) || exit 0

if grep -qE '(^|[;&|[:space:]])(python3?[[:space:]]+-m[[:space:]]+)?pip[[:space:]]+install' <<<"$cmd" \
   && ! grep -qE '(^|[;&|[:space:]])uv[[:space:]]+pip' <<<"$cmd"; then
    echo "BLOCKED: .venv has no pip. Use:" \
         "  uv pip install --python .venv/bin/python <pkg>" \
         "For an editable install of this package add --no-build-isolation" \
         "(the pydnx_packaging build backend is internal and unreachable here)." >&2
    exit 2
fi

if grep -qE '(^|[;&|[:space:]])(pytest|python3?[[:space:]]+-m[[:space:]]+pytest)' <<<"$cmd" \
   && ! grep -qF '.venv/bin/' <<<"$cmd"; then
    echo "BLOCKED: run tests through the project venv:" \
         "  .venv/bin/python -m pytest -q" >&2
    exit 2
fi
exit 0
