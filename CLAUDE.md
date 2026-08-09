# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working in this repository.
Full project context (what this repo is, the three systems, commands, architecture, conventions,
and path-portability rules) lives in `AGENTS.md` — imported below so it's shared with any other
agent tool (e.g. Codex) that reads `AGENTS.md` directly, with no duplication to keep in sync.

@AGENTS.md

## Claude Code specifics

- Prefer the `/aircc-status` and `/slurm-status` skills for read-only campaign checks instead of
  re-deriving queries by hand.
- See `/my-workflow` skill for the full Botero → Slurm/AIRCC development flow.
