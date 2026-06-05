"""MCP tool logic: structured outputs + actionable errors (spec-006)."""
import pytest

from whitebox.mcp import tools as T
from whitebox.mcp.tools import Session, ToolError


@pytest.fixture
def session(wb):
    s = Session()
    s.wb = wb
    return s


def test_require_without_session_errors():
    s = Session()
    with pytest.raises(ToolError) as e:
        T.list_variables(s)
    assert "whitebox_load_session" in str(e.value)


def test_list_variables_paginates(session):
    res = T.list_variables(session, limit=2, offset=0)
    assert res["total_count"] == 4
    assert res["count"] == 2
    assert res["has_more"] is True
    assert res["next_offset"] == 2
    res2 = T.list_variables(session, limit=2, offset=2)
    assert res2["has_more"] is False
    assert res2["next_offset"] is None


def test_describe_unknown_variable_actionable(session):
    with pytest.raises(ToolError) as e:
        T.describe_variable(session, "nope")
    assert "whitebox_list_variables" in str(e.value)


def test_add_group_and_status_flow(session):
    meta = T.add_group(session, "region_grp", "region", {"North": "NS", "South": "NS"},
                       default="Other")
    assert meta["origin"] == "derived_group"
    assert meta["review_status"] == "pending_review"
    # duplicate name -> actionable error
    with pytest.raises(ToolError):
        T.add_group(session, "region_grp", "region", {"North": "NS"})
    # promote
    T.set_variable_status(session, "region_grp", "ready_to_model")
    assert "region_grp" in T.sync_variables(session)["trainable"]
    # remove frees it
    T.remove_variable(session, "region_grp")
    assert "region_grp" not in [v["name"] for v in T.list_variables(session)["items"]]


def test_profile_and_propose(session):
    prof = T.profile_variable(session, "region")
    assert prof["dtype"] == "categorical"
    proposal = T.propose_cleaning(session, "vehicle_value")
    assert proposal["n_bins"] == 10


def test_univariate_plot_writes_html(session, tmp_path):
    res = T.univariate_plot(session, "region", out_dir=str(tmp_path))
    assert res["html_path"].endswith("univariate_region.html")
    import os
    assert os.path.exists(res["html_path"])


def test_load_session_roundtrip(tmp_path, messy_df, feature_names):
    import xgboost as xgb

    data_path = str(tmp_path / "data.csv")
    messy_df.to_csv(data_path, index=False)
    # train + save a model the loader can read
    from tests.conftest import _encode_for_training

    X = _encode_for_training(messy_df, feature_names)
    dtrain = xgb.DMatrix(X, label=messy_df["claim_count"], weight=messy_df["exposure"],
                         feature_names=feature_names)
    model = xgb.train({"objective": "count:poisson", "verbosity": 0}, dtrain, num_boost_round=5)
    model_path = str(tmp_path / "model.json")
    model.save_model(model_path)

    s = Session()
    res = T.load_session(s, data_path, model_path, feature_names, "exposure",
                         actuals_col="claim_count")
    assert res["loaded"] is True
    assert res["n_rows"] == len(messy_df)
    assert set(feature_names).issubset(set(res["variables"]))


def test_load_session_missing_file_actionable():
    s = Session()
    with pytest.raises(ToolError) as e:
        T.load_session(s, "/no/such/data.csv", "/no/model.json", ["a"], "w")
    assert "file not found" in str(e.value)
