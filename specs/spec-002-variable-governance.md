# Spec 002: Variable governance (ingestion, profiling, proposals, validation gate)

- **Status:** approved
- **Phase:** 2
- **Depends on:** spec-001

## Problem
First ingest produces many raw variables with messy values (blanks, NA, ".", odd chars).
None may go straight to training. Everything must pass a first validation, with a
profiling/proposal step the reviewer can edit or accept.

## Requirements
1. `whitebox/governance/` module with: schema snapshot + `ingest(new_data)` revision diff;
   a profiler; an editable cleaning/banding proposal engine; status-gate transitions.
2. Statuses: `pending_review` (default for raw + derived), `ready_to_model` (only via
   explicit transition), `needs_cleaning`, `excluded`. `set_status` blocks
   `ready_to_model` while `needs_cleaning` flags are unresolved.
3. `ingest(new_data)` returns a revision report: new variables (added pending_review +
   profiled), missing variables (flagged), changed variables (dtype/cardinality drift ->
   back to pending_review).
4. `profile_variable(name)` -> `{dtype, n_missing, missing_pct, n_unique, top_levels,
   odd_tokens, flags}`.
5. `propose_cleaning(name, method="by_bins", n_bins=10)`:
   - categorical: normalize default missing tokens `["", "NA", "N/A", "na", ".", "null",
     "NaN"]` to a `Missing` level (code -1); optionally fold rare levels into `Other`;
     return an editable `level_map` + per-level counts.
   - numeric: propose banding via `create_bandings`; method `by_bins` (default 10 bins) or
     `by_magnitude` (order-of-magnitude / log-scale edges); return editable cut points + labels.
6. `apply_cleaning(name, edited_proposal)` materializes the accepted mapping/banding, updates
   `data_quality`, clears `needs_cleaning` if resolved. Accepting does NOT auto-approve.
7. `trainable_variables()` returns only `ready_to_model` variables.

## Design
- `whitebox/governance/profiler.py`: `profile_variable`, missing-token normalization, odd-token
  detection, rare-level folding.
- `whitebox/governance/proposals.py`: `propose_cleaning`, `apply_cleaning`; numeric banding
  reuses `whitebox/data_prep.py:create_bandings` (and adds `by_magnitude` edge computation).
- `whitebox/governance/ingest.py`: `SchemaSnapshot`, `ingest(new_data)` diff.
- Status transitions + `data_quality`/`proposal` fields live on the Encoder registry (spec-001).
- `Whitebox` public methods: `set_status`, `profile`, `propose_cleaning`, `apply_cleaning`,
  `ingest`.

## Acceptance criteria
- `auto_encode` registers raw features as `pending_review`, never `ready_to_model`.
- `profile_variable` reports missingness + odd tokens.
- `propose_cleaning` normalizes missing tokens to `Missing`; numeric defaults to 10 bins and
  supports `by_magnitude`.
- `apply_cleaning` materializes the edited proposal.
- `ingest(new_data)` detects new / missing / changed variables.
- `set_status` blocks approval while `needs_cleaning`.

## Test plan
- `tests/unit/test_governance.py`, `tests/unit/test_variable_manager.py`.

## Out of scope
Derived group/combine creation (spec-003), UI (spec-005), cross-modeler sync (spec-004).
