#!/usr/bin/env python
"""Run EDI model evaluation against ground truth data.

Usage:
    python src/run_evaluation.py                    # Default: 2023-24 season
    python src/run_evaluation.py 2022-23            # Specific season
    python src/run_evaluation.py --all              # All available seasons
"""

import sys
from pathlib import Path

import pandas as pd

from data_fetcher import get_all_defensive_teams, get_all_defensive_player_ids
from evaluation import (
    calculate_coverage,
    calculate_correlations,
    generate_evaluation_report,
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Reference columns for correlation analysis
# DEF_RATING: lower is better (points allowed per 100 possessions)
# Other stats: higher is better
CORRELATION_COLS = [
    "D1_Score",
    "D2_Score",
    "D3_Score",
    "D4_Score",
    "D5_Score",
    "Efficiency",
]


def load_model_output(season: str) -> pd.DataFrame:
    """Load EDI model output for a season.

    Args:
        season: Season string (e.g., "2023-24")

    Returns:
        DataFrame with model predictions.
    """
    filepath = DATA_DIR / f"nba_defensive_all_players_{season}.csv"
    if not filepath.exists():
        print(f"Warning: Model output not found: {filepath}")
        return pd.DataFrame()

    return pd.read_csv(filepath)


def add_ground_truth_labels(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """Add Is_All_Defense column based on ground truth.

    Args:
        df: DataFrame with PLAYER_ID column
        season: Season string

    Returns:
        DataFrame with Is_All_Defense column added.
    """
    all_defense_ids = get_all_defensive_player_ids(season)
    df = df.copy()
    df["Is_All_Defense"] = df["PLAYER_ID"].isin(all_defense_ids).astype(int)
    return df


def evaluate_season(season: str) -> dict:
    """Run full evaluation for a single season.

    Args:
        season: Season string (e.g., "2023-24")

    Returns:
        Dictionary with evaluation results.
    """
    print(f"\nEvaluating season: {season}")
    print("-" * 40)

    # Load model output
    df = load_model_output(season)
    if df.empty:
        print(f"  No data available for {season}")
        return {}

    # Add ground truth labels
    df = add_ground_truth_labels(df, season)

    # Check if we have ground truth for this season
    ground_truth = get_all_defensive_teams(season)
    if ground_truth.empty:
        print(f"  No ground truth available for {season}")
        return {}

    print(f"  Loaded {len(df)} players, {df['Is_All_Defense'].sum()} All-Defense")

    # Layer 1: Coverage analysis
    coverage_10 = calculate_coverage(df, top_n=10)
    coverage_15 = calculate_coverage(df, top_n=15)
    coverage_20 = calculate_coverage(df, top_n=20)

    # Layer 2: Correlation analysis
    available_cols = [c for c in CORRELATION_COLS if c in df.columns]
    correlations = calculate_correlations(df, "EDI_Total", available_cols)

    # Generate report
    report = generate_evaluation_report(coverage_10, correlations, season)
    print(report)

    # Additional coverage at different thresholds
    print("\n## Coverage at Different Thresholds")
    print("-" * 40)
    print(f"Top 10: Precision={coverage_10['precision@10']:.1%}, Recall={coverage_10['recall@10']:.1%}")
    print(f"Top 15: Precision={coverage_15['precision@15']:.1%}, Recall={coverage_15['recall@15']:.1%}")
    print(f"Top 20: Precision={coverage_20['precision@20']:.1%}, Recall={coverage_20['recall@20']:.1%}")

    # Show which All-Defense players we found/missed
    print("\n## All-Defensive Team Analysis")
    print("-" * 40)
    
    all_defense_df = df[df["Is_All_Defense"] == 1].sort_values("EDI_Total", ascending=False)
    print("Found All-Defensive players and their EDI ranks:")
    
    df_sorted = df.sort_values("EDI_Total", ascending=False).reset_index(drop=True)
    df_sorted["Rank"] = df_sorted.index + 1
    
    for _, row in all_defense_df.iterrows():
        player_rank = df_sorted[df_sorted["PLAYER_ID"] == row["PLAYER_ID"]]["Rank"].values
        if len(player_rank) > 0:
            print(f"  #{player_rank[0]:3d}: {row['PLAYER_NAME']:<25} EDI={row['EDI_Total']:.1f}")

    return {
        "season": season,
        "coverage_10": coverage_10,
        "coverage_15": coverage_15,
        "coverage_20": coverage_20,
        "correlations": correlations,
    }


def main():
    """Main entry point."""
    # Parse arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            seasons = ["2023-24", "2022-23", "2021-22"]
        else:
            seasons = [sys.argv[1]]
    else:
        seasons = ["2023-24"]

    print("=" * 60)
    print("EDI Model Evaluation")
    print("=" * 60)

    all_results = []
    for season in seasons:
        result = evaluate_season(season)
        if result:
            all_results.append(result)

    # Summary across seasons
    if len(all_results) > 1:
        print("\n" + "=" * 60)
        print("Summary Across Seasons")
        print("=" * 60)
        for r in all_results:
            s = r["season"]
            p10 = r["coverage_10"]["precision@10"]
            r10 = r["coverage_10"]["recall@10"]
            print(f"{s}: Precision@10={p10:.1%}, Recall@10={r10:.1%}")


if __name__ == "__main__":
    main()
