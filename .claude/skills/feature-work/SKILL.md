---
name: feature-work
description: The branch-and-approval workflow for any change to LLMInspector — new features, bug fixes, refactors. Use whenever starting a unit of work that will touch llminspector/ or tests/, and before creating a branch, committing, or merging.
---

# Feature workflow

Five steps. Two of them are stops where you wait for the user.

## 1. Propose, then stop

Before writing any code, present:

- **What you will change**, as a short task list.
- **Which files** it touches, by path.
- **Anything ambiguous**, with your recommendation.

Then **stop and wait for approval.** Do not open an editor on the strength of an
implied yes. If the request is a one-line fix and the path is obvious, say so and
propose it in a sentence — but still wait.

## 2. Branch

```bash
git checkout -b <short-kebab-name>
```

Branch off the current integration branch. A PreToolUse hook blocks commits made
directly on `main` or `code-refactor`, so this step is not optional.

Name it for the change, not the phase or ticket: `retry-jitter`,
`policy-metric-thresholds`, `excel-context-parsing`.

## 3. Implement fully

- Code **and** tests in the same branch. Every behaviour change gets a test.
- Docs move with the code: the matching `docs/guides/0N_*.md` and any affected
  `examples/*.ipynb`. A public-API change is not done without them.
- Read the `CLAUDE.md` in each directory you touch before editing it.
- Formatting is handled by a hook; don't spend turns on black/isort.
- The suite runs on a Stop hook. If it fails you will be told — fix it rather
  than reporting a failure you could have fixed.

Finish the whole scope. If part of it turns out to be blocked, complete
everything else and say plainly what you left and why.

## 4. Hand back, then stop

Report:

- What changed, grouped by file.
- Test results (counts, and any that you had to change and why).
- Anything you decided that the user might have decided differently.

Then **stop.** The user verifies manually. **Do not commit at this point.**

## 5. Commit and merge — only on explicit go-ahead

```bash
git add <paths>          # never `git add -A`; the tree has untracked scratch
git commit -m "..."
git checkout code-refactor && git merge --no-ff <branch>
```

Commit message rules:

- Imperative summary under ~72 chars, body explaining *why* where it isn't obvious.
- **No `Co-Authored-By` trailer.** This applies to merge commits too. A hook
  blocks it.
- One commit per coherent change; don't squash unrelated work together.

"Looks good" on the diff is approval to commit. It is not, by itself, approval to
merge or push — ask if unsure.
