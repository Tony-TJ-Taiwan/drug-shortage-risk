import argparse
from pathlib import Path

from drug_shortage.profiling import create_sample_data
from drug_shortage.profiling.profile_data import print_summary, profile_all
from drug_shortage.transform import (
    build_demand_features,
    build_master,
    build_recall_features,
    build_supply_features,
    score_risk,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRS = (
    PROJECT_ROOT / "data" / "interim",
    PROJECT_ROOT / "data" / "processed",
)


def ensure_working_dirs() -> None:
    for directory in DATA_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


def run_placeholder(command_name: str) -> int:
    ensure_working_dirs()
    print(
        f"{command_name}: command is scaffolded. "
        "No raw CSV files were read or processed."
    )
    return 0


def run_profile_data() -> int:
    profiles = profile_all()
    print_summary(profiles)
    return 0


def run_create_sample_data() -> int:
    results, limitations = create_sample_data.create_sample_data()
    create_sample_data.print_summary(results, limitations)
    return 0


def run_build_master() -> int:
    result = build_master.build_drug_master()
    build_master.print_summary(result)
    return 0


def run_build_features() -> int:
    supply_result = build_supply_features.build_supply_features()
    build_supply_features.print_summary(supply_result)
    demand_result = build_demand_features.build_demand_features()
    build_demand_features.print_summary(demand_result)
    recall_result = build_recall_features.build_recall_features()
    build_recall_features.print_summary(recall_result)
    return 0


def run_score() -> int:
    result = score_risk.build_risk_scores()
    score_risk.print_summary(result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drug-shortage",
        description="Rule-based drug shortage risk pipeline commands.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in (
        "profile-data",
        "create-sample-data",
        "build-master",
        "build-features",
        "score",
    ):
        subparsers.add_parser(command)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "profile-data":
        return run_profile_data()
    if args.command == "create-sample-data":
        return run_create_sample_data()
    if args.command == "build-master":
        return run_build_master()
    if args.command == "build-features":
        return run_build_features()
    if args.command == "score":
        return run_score()
    return run_placeholder(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
