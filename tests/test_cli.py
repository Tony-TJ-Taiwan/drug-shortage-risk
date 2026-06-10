from drug_shortage.cli import build_parser


def test_cli_accepts_scaffolded_commands() -> None:
    parser = build_parser()

    args = parser.parse_args(["score"])

    assert args.command == "score"
