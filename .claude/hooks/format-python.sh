#!/usr/bin/env bash
# PostToolUse(Edit|Write) — format the file Claude just touched.
#
# Runs the same black + isort that .pre-commit-config.yaml gates on, so the
# edit->lint->fix->re-lint loop never reaches the model. Silent on success;
# only a formatter *error* (syntax error in the file) is reported back.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$ROOT/.venv/bin"

input=$(cat)
file=$(jq -r '.tool_input.file_path // empty' <<<"$input")

[[ -z "$file" || "$file" != *.py || ! -f "$file" ]] && exit 0
[[ ! -x "$PY/black" ]] && exit 0

out=$("$PY/black" -q "$file" 2>&1) || {
    printf 'black failed on %s:\n%s\n' "$file" "$out" >&2
    exit 2
}
out=$("$PY/isort" -q "$file" 2>&1) || {
    printf 'isort failed on %s:\n%s\n' "$file" "$out" >&2
    exit 2
}
exit 0
