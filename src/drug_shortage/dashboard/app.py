from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCORE_PATH = PROJECT_ROOT / "data" / "processed" / "shortage_risk_scores.parquet"

DISPLAY_COLUMNS = [
    "shortage_group_key",
    "risk_level",
    "total_risk_score",
    "supply_score",
    "demand_score",
    "recall_score",
    "global_shortage_score",
    "risk_explanation",
]
FILTER_COLUMNS = {
    "ingredient": "Ingredient",
    "atc_code": "ATC code",
    "manufacturer": "Manufacturer",
}
COMPONENT_COLUMNS = [
    "supply_score",
    "demand_score",
    "recall_score",
    "global_shortage_score",
]


@st.cache_data(show_spinner=False)
def load_scores(path: Path = DEFAULT_SCORE_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def available_display_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in DISPLAY_COLUMNS if column in frame.columns]


def available_component_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in COMPONENT_COLUMNS if column in frame.columns]


def filter_scores(
    frame: pd.DataFrame,
    risk_levels: list[str],
    ingredient: str,
    atc_code: str,
    manufacturer: str,
) -> pd.DataFrame:
    filtered = frame.copy()
    if risk_levels and "risk_level" in filtered.columns:
        filtered = filtered[filtered["risk_level"].isin(risk_levels)]
    filter_values = {
        "ingredient": ingredient,
        "atc_code": atc_code,
        "manufacturer": manufacturer,
    }
    for column, value in filter_values.items():
        if column in filtered.columns and value != "All":
            filtered = filtered[filtered[column].astype("string") == value]
    return filtered


def sorted_filter_options(frame: pd.DataFrame, column: str) -> list[str]:
    if column not in frame.columns:
        return ["All"]
    values = frame[column].dropna().astype("string").unique().tolist()
    return ["All", *sorted(values)]


def top_risk_scores(frame: pd.DataFrame, row_count: int = 50) -> pd.DataFrame:
    if "total_risk_score" not in frame.columns:
        return frame.head(row_count)
    return frame.sort_values("total_risk_score", ascending=False).head(row_count)


def render_missing_output(path: Path) -> None:
    st.title("Drug Shortage Risk Dashboard")
    st.warning(
        "Risk score output was not found. Run the scoring command first, then refresh this page."
    )
    st.code("uv run drug-shortage score", language="powershell")
    st.caption(f"Expected file: {path}")


def render_data_freshness(score_path: Path, frame: pd.DataFrame) -> None:
    st.subheader("Data Freshness")
    modified_at = pd.Timestamp(score_path.stat().st_mtime, unit="s").strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    st.write(f"Score file last modified: {modified_at}")
    st.write(f"Rows loaded: {len(frame):,}")
    if "latest_recall_date" in frame.columns:
        latest_recall = frame["latest_recall_date"].dropna().max()
        st.write(f"Latest recall date in scored data: {latest_recall}")


def render_missing_data_limitations(frame: pd.DataFrame) -> None:
    st.subheader("Missing Data Limitations")
    missing_filter_columns = [
        label for column, label in FILTER_COLUMNS.items() if column not in frame.columns
    ]
    missing_components = [
        column for column in COMPONENT_COLUMNS if column not in frame.columns
    ]
    limitations = []
    if missing_filter_columns:
        limitations.append(
            "These filter columns are not present in the score file: "
            + ", ".join(missing_filter_columns)
            + "."
        )
    if missing_components:
        limitations.append(
            "These component scores are not present in the score file: "
            + ", ".join(missing_components)
            + "."
        )
    if "risk_explanation" not in frame.columns:
        limitations.append("Risk explanations are not present in the score file.")
    if not limitations:
        limitations.append("No missing dashboard fields were detected in the score file.")
    for limitation in limitations:
        st.write(f"- {limitation}")


def main() -> None:
    st.set_page_config(page_title="Drug Shortage Risk", layout="wide")
    st.title("Drug Shortage Risk Dashboard")

    score_path = DEFAULT_SCORE_PATH
    if not score_path.exists():
        render_missing_output(score_path)
        return

    scores = load_scores(score_path)
    filtered_scores = scores

    st.sidebar.header("Filters")
    risk_level_options = (
        sorted(scores["risk_level"].dropna().astype("string").unique().tolist())
        if "risk_level" in scores.columns
        else []
    )
    selected_risk_levels = st.sidebar.multiselect(
        "Risk level",
        options=risk_level_options,
        default=risk_level_options,
    )
    selected_ingredient = st.sidebar.selectbox(
        "Ingredient", sorted_filter_options(scores, "ingredient")
    )
    selected_atc_code = st.sidebar.selectbox("ATC code", sorted_filter_options(scores, "atc_code"))
    selected_manufacturer = st.sidebar.selectbox(
        "Manufacturer", sorted_filter_options(scores, "manufacturer")
    )
    filtered_scores = filter_scores(
        filtered_scores,
        selected_risk_levels,
        selected_ingredient,
        selected_atc_code,
        selected_manufacturer,
    )

    high_risk = top_risk_scores(filtered_scores, row_count=50)
    metric_columns = st.columns(3)
    metric_columns[0].metric("Scored groups", f"{len(filtered_scores):,}")
    if "total_risk_score" in filtered_scores.columns and not filtered_scores.empty:
        metric_columns[1].metric(
            "Highest total risk score",
            f"{filtered_scores['total_risk_score'].max():.1f}",
        )
        metric_columns[2].metric(
            "Average total risk score",
            f"{filtered_scores['total_risk_score'].mean():.1f}",
        )
    else:
        metric_columns[1].metric("Highest total risk score", "N/A")
        metric_columns[2].metric("Average total risk score", "N/A")

    st.subheader("Top 50 High-Risk Drugs")
    st.dataframe(
        high_risk[available_display_columns(high_risk)],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Component Score Breakdown")
    component_columns = available_component_columns(high_risk)
    if component_columns:
        chart_frame = high_risk.set_index("shortage_group_key")[component_columns]
        st.bar_chart(chart_frame)
    else:
        st.info("No component score columns are available for the breakdown.")

    st.subheader("Risk Explanation")
    if "risk_explanation" in high_risk.columns and not high_risk.empty:
        selected_group = st.selectbox(
            "Shortage group",
            high_risk["shortage_group_key"].astype("string").tolist(),
        )
        explanation = high_risk.loc[
            high_risk["shortage_group_key"].astype("string") == selected_group,
            "risk_explanation",
        ].iloc[0]
        st.write(explanation)
    else:
        st.info("No risk explanations are available in the score file.")

    render_data_freshness(score_path, scores)
    render_missing_data_limitations(scores)


if __name__ == "__main__":
    main()
