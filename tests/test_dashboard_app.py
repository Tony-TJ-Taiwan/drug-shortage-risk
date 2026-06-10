import pandas as pd

from drug_shortage.dashboard.app import (
    available_component_columns,
    filter_scores,
    sorted_filter_options,
    top_risk_scores,
)


def test_filter_scores_uses_available_dashboard_filters() -> None:
    scores = pd.DataFrame(
        [
            {
                "shortage_group_key": "A",
                "risk_level": "High",
                "ingredient": "Drug A",
                "atc_code": "A01",
                "manufacturer": "Maker 1",
                "total_risk_score": 90,
            },
            {
                "shortage_group_key": "B",
                "risk_level": "Low",
                "ingredient": "Drug B",
                "atc_code": "B01",
                "manufacturer": "Maker 2",
                "total_risk_score": 10,
            },
        ]
    )

    filtered = filter_scores(
        scores,
        risk_levels=["High"],
        ingredient="Drug A",
        atc_code="All",
        manufacturer="Maker 1",
    )

    assert filtered["shortage_group_key"].tolist() == ["A"]


def test_filter_scores_ignores_missing_optional_filter_columns() -> None:
    scores = pd.DataFrame([{"shortage_group_key": "A", "risk_level": "High"}])

    filtered = filter_scores(
        scores,
        risk_levels=["High"],
        ingredient="Missing",
        atc_code="All",
        manufacturer="All",
    )

    assert filtered["shortage_group_key"].tolist() == ["A"]


def test_top_risk_scores_orders_by_total_risk_score() -> None:
    scores = pd.DataFrame(
        [
            {"shortage_group_key": "A", "total_risk_score": 20},
            {"shortage_group_key": "B", "total_risk_score": 80},
        ]
    )

    top_scores = top_risk_scores(scores, row_count=1)

    assert top_scores["shortage_group_key"].tolist() == ["B"]


def test_sorted_filter_options_returns_all_for_missing_column() -> None:
    assert sorted_filter_options(pd.DataFrame(), "ingredient") == ["All"]


def test_available_component_columns_only_returns_present_columns() -> None:
    scores = pd.DataFrame(columns=["supply_score", "recall_score", "other"])

    assert available_component_columns(scores) == ["supply_score", "recall_score"]
