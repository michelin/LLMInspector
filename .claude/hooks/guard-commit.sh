#!/usr/bin/env bash
# PreToolUse(Bash) — enforce the two standing commit rules statically.
#
#   1. No `Co-Authored-By` trailer in this repo (overrides the global default).
#   2. No direct commits to a protected branch; feature work gets its own
#      branch off code-refactor (see .claude/skills/feature-work).
#
# Exit 2 blocks the call and hands stderr back to Claude as the reason.
set -uo pipefail

input=$(cat)
# Fail open: a guard that wedges the session on unexpected input is worse than
# a guard that misses one call.
cmd=$(jq -r '.tool_input.command // empty' <<<"$input" 2>/dev/null) || exit 0

grep -qE '(^|[;&|[:space:]])git[[:space:]]+commit' <<<"$cmd" || exit 0

# A real trailer only: start of a line (or an escaped \n inside a -m string)
# followed by the key and its colon. Prose that merely names the trailer -- as
# this repo's own docs and commit messages do -- must not trip the guard.
if grep -qiE '(^|\\n)[[:space:]]*co-authored-by:' <<<"$cmd"; then
    echo "BLOCKED: this repo's commit messages carry no Co-Authored-By trailer." \
         "Rewrite the message with the summary/body only." >&2
    exit 2
fi

branch=$(git branch --show-current 2>/dev/null)
case "$branch" in
    main|master|code-refactor)
        echo "BLOCKED: direct commit to protected branch '$branch'." \
             "Create a feature branch off code-refactor first" \
             "(git checkout -b <name>), then commit there. Merging back into" \
             "$branch happens only on the user's explicit go-ahead." >&2
        exit 2
        ;;
esac
exit 0
