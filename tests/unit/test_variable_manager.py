"""Name primary-key validation + status lifecycle (spec-002 / spec-003)."""
import pytest

from whitebox.encoding import Encoder


def _encoder(messy_df, feature_names):
    enc = Encoder(messy_df, feature_names, verbose=False)
    enc.auto_encode()
    return enc


def test_slug_rule_rejects_bad_names(messy_df, feature_names):
    enc = _encoder(messy_df, feature_names)
    for bad in ["has space", "weird!", "1leading", "ends_encoded"]:
        with pytest.raises(ValueError):
            enc.validate_name(bad)


def test_uniqueness_rejects_collisions(messy_df, feature_names):
    enc = _encoder(messy_df, feature_names)
    # Collides with an existing raw feature.
    with pytest.raises(ValueError):
        enc.validate_name("region")
    # Collides with an existing encoded column name target.
    with pytest.raises(ValueError):
        enc.validate_name("age")


def test_valid_unique_name_passes(messy_df, feature_names):
    enc = _encoder(messy_df, feature_names)
    assert enc.validate_name("region_grouped_v1") is True


def test_default_status_pending_and_transitions(messy_df, feature_names):
    enc = _encoder(messy_df, feature_names)
    assert enc.registry["region"]["review_status"] == "pending_review"
    enc.set_status("region", "ready_to_model")
    assert enc.trainable_variables() == ["region"]
    enc.set_status("region", "excluded")
    assert "region" not in enc.trainable_variables()
    with pytest.raises(ValueError):
        enc.set_status("region", "not_a_status")
    with pytest.raises(ValueError):
        enc.set_status("unknown_var", "excluded")
