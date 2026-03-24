"""
Entry point for the ACBC survey CLI and web server.

Usage:
    python main.py                                # Run CLI survey (default config)
    python main.py --config path/to.yaml          # Run with custom config
    python main.py --seed 42 --participant P001   # Fixed seed + participant ID
    python main.py --output-dir ./my_data         # Custom output directory
    python main.py aggregate                      # Aggregate all participants
    python main.py aggregate --data-dir ./my_data # Aggregate from custom dir
    python main.py serve                          # Start web survey interface
    python main.py serve --port 9000              # Web interface on custom port
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "development.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ACBC — Adaptive Choice-Based Conjoint Analysis Survey",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── Default: run a survey ───────────────────────────────────────
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to YAML survey config (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible scenario generation",
    )
    parser.add_argument(
        "--participant",
        type=str,
        default=None,
        help="Participant ID (prompted interactively if omitted)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory to save raw data and analysis results (default: ./data)",
    )

    # ── Subcommand: aggregate ───────────────────────────────────────
    agg_parser = subparsers.add_parser(
        "aggregate",
        help="Aggregate results from all participants",
    )
    agg_parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing raw/ and analysis/ subdirs (default: ./data)",
    )
    agg_parser.add_argument(
        "--method",
        choices=["counts", "monotone", "bayesian_logit", "hb", "all"],
        default="all",
        help=(
            "Analysis method: counts, monotone, bayesian_logit "
            "(per-participant), hb (joint Hierarchical Bayes), "
            "or all (default: all)"
        ),
    )
    agg_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for HB analysis during aggregation",
    )

    # ── Subcommand: serve (web interface) ─────────────────────────
    serve_parser = subparsers.add_parser(
        "serve",
        help="Run web survey interface",
    )
    serve_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to YAML survey config (default: {DEFAULT_CONFIG})",
    )
    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)",
    )
    serve_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for scenario generation",
    )
    serve_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory to save raw data (default: ./data)",
    )

    args = parser.parse_args()

    if args.command == "aggregate":
        from cli.aggregate import run_aggregate
        run_aggregate(
            data_dir=args.data_dir,
            method=args.method,
            seed=args.seed,
        )
    elif args.command == "serve":
        config_path = args.config
        if not config_path.exists():
            print(f"Error: config file not found: {config_path}", file=sys.stderr)
            sys.exit(1)

        import uvicorn
        from acbc.models import SurveyConfig
        from web.app import create_app

        config = SurveyConfig.from_yaml(config_path)
        app = create_app(config, seed=args.seed, output_dir=args.output_dir)
        print(f"Starting ACBC web survey at http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        if not args.config.exists():
            print(f"Error: config file not found: {args.config}", file=sys.stderr)
            sys.exit(1)

        from cli.survey import run_survey
        run_survey(
            args.config,
            seed=args.seed,
            participant_id=args.participant,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
