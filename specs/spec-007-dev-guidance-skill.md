# Spec 007: Dev guidance skill (`whitebox-dev`)

- **Status:** approved
- **Phase:** 7
- **Depends on:** spec-001..006

## Problem
We want the spec-driven workflow, MCP usage, and model-tiering guidance captured so any
subagent the user spins up is steered correctly. Named tiered agent files are deferred to keep
token cost zero until the workflow proves repetitive.

## Requirements
1. `.claude/skills/whitebox-dev/SKILL.md` documenting:
   - The spec-driven flow (write/approve a spec before implementing; one PR per spec;
     conventional commits; no Claude co-author line).
   - Which model tier to use for what: Opus for advanced design/refactors, Sonnet for
     operative implementation, Haiku for commits/chores.
   - How to use the `whitebox_mcp` tools during development.
   - The variable-governance rules and the collaborative-registry invariant.
2. No named agent files in this scope (explicitly deferred; documented as a future option).

## Acceptance criteria
- `SKILL.md` exists with valid frontmatter (`name`, `description`) and the sections above.

## Test plan
- Lightweight: a presence/lint check that the skill file exists with required frontmatter.

## Out of scope
Building `.claude/agents/whitebox-{architect,engineer,committer}.md` (deferred).
