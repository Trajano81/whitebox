"""Smoke test: the package imports and the test harness + fixtures work."""


def test_package_imports():
    import whitebox

    assert hasattr(whitebox, "Whitebox")


def test_messy_fixture(messy_df, feature_names):
    assert len(messy_df) == 600
    assert set(feature_names).issubset(messy_df.columns)
    # The messy region column contains injected missing tokens.
    assert (messy_df["region"] == "").any()
    assert messy_df["mexican_states"].nunique() > 20
