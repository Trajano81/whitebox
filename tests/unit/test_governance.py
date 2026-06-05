"""Variable governance: profiling, proposals, ingest diff, status gate (spec-002)."""
import numpy as np
import pandas as pd

from whitebox.encoding import Encoder
from whitebox.governance import (
    MISSING_LABEL,
    apply_cleaning,
    diff_schema,
    profile_variable,
    propose_cleaning,
    schema_snapshot,
)


def test_profile_reports_missing_and_odd_tokens(messy_df):
    prof = profile_variable(messy_df["region"], name="region")
    assert prof["dtype"] == "categorical"
    assert prof["n_missing"] > 0  # "", "NA", "." are missing tokens
    # " North " (whitespace) is detected as an odd token.
    assert any(tok != tok.strip() for tok in prof["odd_tokens"])


def test_propose_categorical_normalizes_missing_and_trims(messy_df):
    proposal = propose_cleaning(messy_df["region"], name="region")
    assert proposal["kind"] == "categorical_cleaning"
    lm = proposal["level_map"]
    # missing tokens map to the Missing level
    assert lm.get("") == MISSING_LABEL
    assert lm.get("NA") == MISSING_LABEL
    # whitespace-padded label is trimmed to the clean label
    assert lm.get(" North ") == "North"


def test_apply_categorical_cleaning(messy_df):
    proposal = propose_cleaning(messy_df["region"], name="region")
    cleaned = apply_cleaning(messy_df, "region", proposal)
    assert (cleaned == MISSING_LABEL).sum() > 0
    assert " North " not in set(cleaned.unique())
    assert "North" in set(cleaned.unique())


def test_propose_numeric_defaults_to_10_bins(messy_df):
    proposal = propose_cleaning(messy_df["vehicle_value"], name="vehicle_value")
    assert proposal["kind"] == "numeric_banding"
    assert proposal["method"] == "by_bins"
    assert proposal["n_bins"] == 10
    # up to 11 edges for 10 bins (fewer only if duplicate quantiles)
    assert 2 <= len(proposal["edges"]) <= 11


def test_propose_numeric_by_magnitude(messy_df):
    proposal = propose_cleaning(
        messy_df["vehicle_value"], name="vehicle_value", method="by_magnitude"
    )
    assert proposal["method"] == "by_magnitude"
    assert len(proposal["edges"]) >= 2
    # magnitude edges are increasing
    assert proposal["edges"] == sorted(proposal["edges"])


def test_ingest_detects_new_missing_changed(messy_df, feature_names):
    old = schema_snapshot(messy_df, columns=feature_names)
    new_df = messy_df.drop(columns=["mexican_states"]).copy()
    new_df["new_var"] = "x"
    new = schema_snapshot(new_df, columns=list(new_df.columns))
    report = diff_schema(old, new)
    assert "new_var" in report["new"]
    assert "mexican_states" in report["missing"]


def test_whitebox_governance_methods(wb):
    # profile -> propose -> apply -> approve via the Whitebox facade.
    prof = wb.profile("region")
    assert prof["flags"]  # messy region has flags
    assert wb.encoder.registry["region"]["review_status"] == "needs_cleaning"

    proposal = wb.propose_cleaning("region")
    wb.apply_cleaning("region", proposal)
    # cleaning resolved the flags and reset status; encoded column rebuilt.
    assert wb.encoder.registry["region"]["review_status"] == "pending_review"
    assert "region_encoded" in wb.data.columns

    wb.set_status("region", "ready_to_model")
    assert "region" in wb.trainable_variables()

    report = wb.ingest(wb.data.drop(columns=["mexican_states"]))
    assert "mexican_states" in report["missing"]


def test_status_gate_blocks_ready_while_needs_cleaning(messy_df, feature_names):
    enc = Encoder(messy_df, feature_names, verbose=False)
    enc.auto_encode()
    # Profile region -> flags (odd tokens / missingness) -> needs_cleaning.
    prof = profile_variable(messy_df["region"], name="region")
    enc.set_profile("region", prof)
    assert enc.registry["region"]["review_status"] == "needs_cleaning"
    try:
        enc.set_status("region", "ready_to_model")
        assert False, "expected ValueError"
    except ValueError:
        pass
    # After clearing flags it can be promoted.
    enc.registry["region"]["data_quality"]["flags"] = []
    enc.set_status("region", "pending_review")
    enc.set_status("region", "ready_to_model")
    assert enc.trainable_variables() == ["region"]
