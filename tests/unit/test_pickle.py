"""Whitebox pickles and restores its link functions (spec-001)."""
import pickle

import numpy as np


def test_whitebox_pickle_roundtrip(wb):
    blob = pickle.dumps(wb)
    restored = pickle.loads(blob)
    # Link function rebuilt on load (poisson -> np.exp).
    assert restored.link_fn is np.exp
    assert restored.link_fn_str == "poisson"
    # Encoder + category mappings survive and stay aliased.
    assert restored.category_mappings is restored.encoder.category_mappings
    assert "region" in restored.category_mappings


def test_pickle_state_drops_unpicklable(wb):
    state = wb.__getstate__()
    assert "link_fn" not in state
    assert "_ci_fn" not in state
    assert "explainer" not in state
