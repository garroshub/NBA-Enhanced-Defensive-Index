#!/usr/bin/env python
"""Run EDI model evaluation against All-Defensive Team ground truth.

Usage:
    python src/run_evaluation.py                    # Default: show help
    python src/run_evaluation.py 2022-23            # Specific season
    python src/run_evaluation.py --all              # 5 seasons (2019-20 to 2023-24)
    python src/run_evaluation.py --external 2021-22 # With external validation
"""

import io
import sys
from pathlib import Path

# Set UTF-8 encoding for stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation import (
    evaluate_season,
    generate_season_report,
    generate_multi_season_report,
    run_multi_season_evaluation,
    calculate_stability_metrics,
)

import pandas as pd

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Default seasons for --all flag
ALL_SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]


def evaluate_single_season(season: str, with_external: bool = False) -> None:
    """Run evaluation for a single season and print report.

    Args:
        season: Season string (e.g., "2023-24")
        with_external: Whether to include external metric validation
    """
    csv_path = DATA_DIR / f"nba_defensive_all_players_{season}.csv"

    if not csv_path.exists():
        print(f"Error: Data file not found: {csv_path}")
        print(f"Run 'python src/nba_defense_mvp.py {season}' first to generate data.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} players from {csv_path.name}")
    print()

    # Run evaluation
    try:
        eval_result = evaluate_season(df, season)
        report = generate_season_report(eval_result)
        print(report)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # External validation (if requested and available)
    if with_external:
        print()
        try:
            from external_metrics import generate_external_validation_report

            ext_report = generate_external_validation_report(df, season)
            print(ext_report)
        except ImportError:
            print("External metrics module not available")
        except Exception as e:
            print(f"External validation error: {e}")


def evaluate_all_seasons(with_external: bool = False) -> None:
    """Run evaluation for all 5 seasons and print comprehensive report."""
    print("Running 5-season evaluation...")
    print(f"Seasons: {', '.join(ALL_SEASONS)}")
    print()

    # Check which data files exist
    missing = []
    for season in ALL_SEASONS:
        csv_path = DATA_DIR / f"nba_defensive_all_players_{season}.csv"
        if not csv_path.exists():
            missing.append(season)

    if missing:
        print(f"Warning: Missing data for seasons: {', '.join(missing)}")
        print("Run 'python src/nba_defense_mvp.py <season>' to generate missing data.")
        print()

    # Run evaluation
    results, stability = run_multi_season_evaluation(ALL_SEASONS, str(DATA_DIR))

    # Print report
    report = generate_multi_season_report(results, stability)
    print(report)

    # External validation summary (if requested)
    if with_external and results:
        print()
        print("=" * 70)
        print("EXTERNAL VALIDATION SUMMARY")
        print("=" * 70)
        print()

        try:
            from external_metrics import (
                merge_external_metrics,
                calculate_external_correlation,
            )

            # Only for seasons with RAPTOR data (2019-20 to 2021-22)
            raptor_seasons = ["2019-20", "2020-21", "2021-22"]

            print("Note: RAPTOR data available for 2019-20 to 2021-22 only")
            print()
            print(f"{'Season':<12} {'D-RAPTOR Corr':<15} {'N Players':<12}")
            print("-" * 40)

            for season in raptor_seasons:
                csv_path = DATA_DIR / f"nba_defensive_all_players_{season}.csv"
                if not csv_path.exists():
                    continue

                df = pd.read_csv(csv_path)
                merged = merge_external_metrics(df, season)
                corr = calculate_external_correlation(merged)

                if "D_RAPTOR" in corr and "error" not in corr["D_RAPTOR"]:
                    r = corr["D_RAPTOR"]["spearman"]
                    n = corr["D_RAPTOR"]["n_players"]
                    print(f"{season:<12} {r:<15.3f} {n:<12}")
                else:
                    print(f"{season:<12} {'N/A':<15} {'-':<12}")

        except ImportError:
            print("External metrics module not available")
        except Exception as e:
            print(f"External validation error: {e}")


def main():
    """Main entry point."""
    print("=" * 70)
    print("EDI Model Three-Dimensional Evaluation")
    print("=" * 70)
    print()

    # Parse arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        with_external = "--external" in sys.argv

        if arg == "--all":
            evaluate_all_seasons(with_external)
        elif arg == "--external":
            # If --external is first arg, check for season
            if len(sys.argv) > 2 and sys.argv[2] != "--all":
                evaluate_single_season(sys.argv[2], with_external=True)
            else:
                evaluate_all_seasons(with_external=True)
        elif arg == "--help" or arg == "-h":
            show_help()
        else:
            evaluate_single_season(arg, with_external)
    else:
        # Default: show help
        show_help()


def show_help():
    """Show usage help."""
    print("Evaluates EDI model against NBA All-Defensive Team selections")
    print()
    print("Three-Dimensional Evaluation:")
    print("  1. Tier Alignment - Average rank of All-Defense players in model")
    print("  2. Candidate Pool Quality - Recall@K metrics (K=10,15,20,30)")
    print("  3. Miss Analysis - Categorizes misses by severity")
    print()
    print("Usage:")
    print("  python src/run_evaluation.py <season>       # Single season")
    print("  python src/run_evaluation.py --all          # All 5 seasons")
    print(
        "  python src/run_evaluation.py --external <season>  # With D-RAPTOR validation"
    )
    print("  python src/run_evaluation.py --all --external     # All with validation")
    print()
    print("Examples:")
    print("  python src/run_evaluation.py 2023-24")
    print("  python src/run_evaluation.py --all")
    print("  python src/run_evaluation.py --external 2021-22")
    print()
    print("Note: External validation (D-RAPTOR) available for 2019-20 to 2021-22")


if __name__ == "__main__":
    main()
