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
    evaluate_dpoy_alignment,
    benchmark_against_raptor,
    generate_dpoy_report,
    generate_benchmark_report,
    DPOYEvaluation,
    BenchmarkResult,
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

        # DPOY alignment report
        dpoy_eval = evaluate_dpoy_alignment(df, season)
        dpoy_report = generate_dpoy_report(dpoy_eval)
        print(dpoy_report)

    except ValueError as e:
        print(f"Error: {e}")
        return

    # External validation (if requested and available)
    if with_external:
        print()
        try:
            from external_metrics import (
                generate_external_validation_report,
                fetch_raptor_data,
            )

            ext_report = generate_external_validation_report(df, season)
            print(ext_report)

            # EDI vs D-RAPTOR benchmark (for seasons with RAPTOR data)
            # Note: fetch_raptor_data returns ALL seasons, need to filter
            all_raptor_df = fetch_raptor_data()
            if all_raptor_df is not None and not all_raptor_df.empty:
                # Convert season string to year (e.g., "2021-22" -> 2022)
                season_year = int(season.split("-")[0]) + 1
                raptor_df = pd.DataFrame(
                    all_raptor_df[all_raptor_df["season"] == season_year]
                )

                if not raptor_df.empty:
                    benchmark = benchmark_against_raptor(
                        df,
                        season,
                        raptor_df,
                        raptor_score_col="raptor_defense",
                        raptor_name_col="player_name",
                    )
                    if benchmark:
                        benchmark_report = generate_benchmark_report([benchmark])
                        print(benchmark_report)

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

    # DPOY Alignment Summary
    print("=" * 70)
    print("🏆 DPOY ALIGNMENT SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Season':<12} {'Actual DPOY':<25} {'EDI Rank':<12} {'Grade':<8}")
    print("-" * 60)

    dpoy_evals: list[DPOYEvaluation] = []
    for season in ALL_SEASONS:
        csv_path = DATA_DIR / f"nba_defensive_all_players_{season}.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        dpoy_eval = evaluate_dpoy_alignment(df, season)
        dpoy_evals.append(dpoy_eval)

        dpoy_name = dpoy_eval.actual_dpoy_name or "N/A"
        dpoy_rank = (
            f"#{dpoy_eval.actual_dpoy_rank}" if dpoy_eval.actual_dpoy_rank else "N/A"
        )
        hit_mark = " ✓" if dpoy_eval.is_hit else ""
        print(
            f"{season:<12} {dpoy_name:<25} {dpoy_rank:<12} {dpoy_eval.grade:<8}{hit_mark}"
        )

    # DPOY Summary stats
    ranks = [d.actual_dpoy_rank for d in dpoy_evals if d.actual_dpoy_rank is not None]
    hits = sum(1 for d in dpoy_evals if d.is_hit)
    if ranks:
        print("-" * 60)
        print(
            f"{'Average':<12} {'':<25} {'#' + str(round(sum(ranks) / len(ranks), 1)):<12}"
        )
        print(f"{'Hits':<12} {'':<25} {f'{hits}/{len(dpoy_evals)}':<12}")
    print()

    # EDI vs D-RAPTOR Benchmark (for seasons with RAPTOR data)
    if with_external:
        print()
        print("=" * 70)
        print("⚔️ EDI vs D-RAPTOR BENCHMARK")
        print("=" * 70)
        print()
        print("Note: RAPTOR data available for 2019-20 to 2021-22 only")
        print()

        try:
            from external_metrics import (
                merge_external_metrics,
                calculate_external_correlation,
                fetch_raptor_data,
            )

            raptor_seasons = ["2019-20", "2020-21", "2021-22"]
            benchmarks: list[BenchmarkResult] = []

            # Fetch all RAPTOR data once (cached)
            all_raptor_df = fetch_raptor_data()

            for season in raptor_seasons:
                csv_path = DATA_DIR / f"nba_defensive_all_players_{season}.csv"
                if not csv_path.exists():
                    continue

                df = pd.read_csv(csv_path)

                # Filter RAPTOR data for this season
                if all_raptor_df is not None and not all_raptor_df.empty:
                    season_year = int(season.split("-")[0]) + 1
                    raptor_df = pd.DataFrame(
                        all_raptor_df[all_raptor_df["season"] == season_year]
                    )

                    if not raptor_df.empty:
                        benchmark = benchmark_against_raptor(
                            df,
                            season,
                            raptor_df,
                            raptor_score_col="raptor_defense",
                            raptor_name_col="player_name",
                        )
                        if benchmark:
                            benchmarks.append(benchmark)

            if benchmarks:
                benchmark_report = generate_benchmark_report(benchmarks)
                print(benchmark_report)
            else:
                print("No benchmark data available (RAPTOR data fetch failed)")

            # Also show correlation summary
            print("-" * 70)
            print("External Correlation Summary:")
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
    print("Evaluation Dimensions:")
    print("  1. Tier Alignment - Average rank of All-Defense players in model")
    print("  2. Candidate Pool Quality - Recall@K metrics (K=10,15,20,30)")
    print("  3. Miss Analysis - Categorizes misses by severity")
    print("  4. DPOY Alignment - How well model predicts Defensive Player of Year")
    print("  5. Benchmark (with --external) - EDI vs D-RAPTOR head-to-head")
    print()
    print("Usage:")
    print("  python src/run_evaluation.py <season>       # Single season")
    print("  python src/run_evaluation.py --all          # All 5 seasons")
    print(
        "  python src/run_evaluation.py --external <season>  # With D-RAPTOR benchmark"
    )
    print("  python src/run_evaluation.py --all --external     # All with benchmark")
    print()
    print("Examples:")
    print("  python src/run_evaluation.py 2023-24")
    print("  python src/run_evaluation.py --all")
    print("  python src/run_evaluation.py --external 2021-22")
    print()
    print(
        "Note: D-RAPTOR data available for 2019-20 to 2021-22 (FiveThirtyEight discontinued)"
    )


if __name__ == "__main__":
    main()
