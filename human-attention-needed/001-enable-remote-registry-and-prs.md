# 001: Enable remote registry sync + open PRs (needs your authorization)

These are outward-facing actions I intentionally did NOT run autonomously. The
whole build is committed locally on branch `feature/internal-encoding-governance`.
Everything works offline; the items below activate the collaborative + remote parts.

## What is blocked / waiting on you

1. **Push the feature branch + open per-spec PRs.** The remote is
   `origin = https://github.com/Trajano81/whitebox.git`. I have not pushed anything.
   - Option A (recommended): push `feature/internal-encoding-governance` and open ONE
     PR for the whole build.
   - Option B: split into 7 stacked PRs (one per spec) per the plan.

2. **Create the `variables-registry` ref.** `registry.json` is designed to live on a
   dedicated orphan branch, not in feature branches. Creating it is a deliberate git
   operation (e.g. `git switch --orphan variables-registry` + an empty
   `{"variables": []}` + push). I have not created it.

3. **Enable the CI fan-out.** `.github/workflows/registry-propagate.yml` and
   `registry-check.yml` are committed but only run once the repo has the ref and
   Actions enabled. They are token-free (Actions `GITHUB_TOKEN`).

4. **Turn on live sync.** `Whitebox.sync_variable` / `whitebox.registry.sync` only push
   when `WHITEBOX_REGISTRY_SYNC=1` and a remote is configured. Off by default.

## Recommended default if you do not specify

Push the branch and open a single PR (Option A); leave the `variables-registry` ref and
live sync for a follow-up once you confirm the remote/Actions setup.

## Answer

<!-- Write your choice here (e.g. "Option A, also create the ref"), then I will proceed. -->
