import argparse
from pathlib import Path

from drug_shortage.profiling.profile_data import print_summary, profile_all

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drug-shortage",
        description="Rule-based drug shortage risk pipeline commands.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("profile-data", "build-master", "build-features", "score"):
        subparsers.add_parser(command)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "profile-data":
        return run_profile_data()
    return run_placeholder(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
