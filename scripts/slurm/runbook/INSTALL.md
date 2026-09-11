# Wire this runbook into your agent

`SKILL.md` and `references/` in this directory are plain Markdown with no
executables and no tool-specific syntax. Any agent, and any human, can just read
them. Nothing below is required to *use* the runbook - it only makes an agent pick it
up on its own, without being pointed at it.

Each vendor scans a different path. Pick yours, and prefer a stub file over a symlink:
symlinks do not survive a Windows checkout unless `core.symlinks` is enabled, and this
repository's pre-commit config already goes out of its way to work on both platforms.

## Claude Code

Already wired: `.claude/skills/plgrid-run/SKILL.md` is a committed stub that points
here, so a fresh clone works with no setup.

## Codex

Per-user, outside the repository:

```bash
skills_dir="${CODEX_HOME:-$HOME/.codex}/skills"
mkdir -p "$skills_dir"
ln -s "$PWD/scripts/slurm/runbook" "$skills_dir/plgrid-run"
```

The destination must not already exist. Restart Codex or refresh skill discovery
afterwards.

## Cursor

Add a rule file under `.cursor/rules/` whose body is one line pointing at
`scripts/slurm/runbook/SKILL.md`.

## GitHub Copilot

Reference `scripts/slurm/runbook/SKILL.md` from `.github/copilot-instructions.md`.

## Gemini CLI and the `AGENTS.md` convention

Both read a file at the repository root - `GEMINI.md` and `AGENTS.md` respectively.
Note that this repository's `.gitignore` currently treats both as private local
context, so a committed `AGENTS.md` would need that entry removed first. Until then,
Gemini users should add their own local `GEMINI.md` line pointing at
`scripts/slurm/runbook/SKILL.md`.

## Anything else

Point the tool at `scripts/slurm/runbook/SKILL.md`. That file routes to the three references
and states the operating rules; it is the only entry point.
