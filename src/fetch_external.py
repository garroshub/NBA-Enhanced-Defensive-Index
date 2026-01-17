"""External defensive metrics fetcher for EDI model validation.

This module provides utilities to fetch official NBA defensive metrics
(DEF_RATING, DEF_WS) for external validation of the EDI model.

Data Sources:
- NBA Official API (nba.com/stats): DEF_RATING, DEF_WS
  - DEF_RATING: Defensive Rating (points allowed per 100 possessions)
  - DEF_WS: Defensive Win Shares (cumulative defensive contribution)

Usage:
    # Fetch official NBA metrics for a season
    df = fetch_official_defensive_metrics("2023-24")

    # Compare with EDI model
    report = generate_external_comparison_report(edi_df, "2023-24")
"""

import time
from pathlib import Path
from functools import lru_cache

import pandas as pd
import numpy as np
from scipy import stats

# Data directory
DATA_DIR = Path(__file__).parent.parent / "data"

# Supported seasons
SUPPORTED_SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]


# =============================================================================
# NBA Official API Data Fetching
# =============================================================================


@lru_cache(maxsize=10)
def fetch_official_defensive_metrics(
    season: str, min_gp: int = 40, min_mpg: float = 20.0
) -> pd.DataFrame:
    """Fetch DEF_RATING and DEF_WS from NBA official API.

    Args:
        season: Season string (e.g., '2023-24')
        min_gp: Minimum games played filter
        min_mpg: Minimum minutes per game filter

    Returns:
        DataFrame with PLAYER_NAME, DEF_RATING, DEF_WS, and ranks
    """
    from nba_api.stats.endpoints import leaguedashplayerstats

    print(f"  Fetching official NBA metrics for {season}...")

    # Get Advanced stats (contains DEF_RATING)
    advanced = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star="Regular Season",
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="PerGame",
    )
    time.sleep(0.6)
    df_adv = advanced.get_data_frames()[0]

    # Get Defense stats (contains DEF_WS)
    defense = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star="Regular Season",
        measure_type_detailed_defense="Defense",
        per_mode_detailed="Totals",  # DEF_WS is a cumulative stat
    )
    time.sleep(0.6)
    df_def = defense.get_data_frames()[0]

    # Filter by GP and MIN
    df_adv_filtered = df_adv[(df_adv["GP"] >= min_gp) & (df_adv["MIN"] >= min_mpg)]

    # Merge DEF_WS from defense stats
    df_merged = df_adv_filtered.merge(
        df_def[["PLAYER_ID", "DEF_WS"]], on="PLAYER_ID", how="left"
    )

    # Calculate ranks
    # DEF_RATING: lower is better (rank ascending)
    # DEF_WS: higher is better (rank descending)
    df_merged["DEF_RATING_RANK"] = df_merged["DEF_RATING"].rank(ascending=True)
    df_merged["DEF_WS_RANK"] = df_merged["DEF_WS"].rank(ascending=False)

    # Sort by DEF_RATING (primary)
    df_merged = df_merged.sort_values("DEF_RATING", ascending=True).reset_index(
        drop=True
    )

    print(
        f"    Found {len(df_merged)} qualified players (GP>={min_gp}, MPG>={min_mpg})"
    )

    return df_merged[
        [
            "PLAYER_NAME",
            "PLAYER_ID",
            "GP",
            "MIN",
            "DEF_RATING",
            "DEF_WS",
            "DEF_RATING_RANK",
            "DEF_WS_RANK",
        ]
    ]


def get_official_top_n(
    season: str, metric: str = "DEF_RATING", top_n: int = 30
) -> list[tuple[str, float, int]]:
    """Get top N players by official defensive metric.

    Args:
        season: Season string
        metric: 'DEF_RATING' or 'DEF_WS'
        top_n: Number of top players to return

    Returns:
        List of (player_name, metric_value, rank) tuples
    """
    df = fetch_official_defensive_metrics(season)

    if metric == "DEF_RATING":
        # Lower is better, already sorted ascending
        df_sorted = df.head(top_n)
    elif metric == "DEF_WS":
        # Higher is better, sort descending
        df_sorted = df.sort_values("DEF_WS", ascending=False).head(top_n)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    result = []
    for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
        result.append((row["PLAYER_NAME"], row[metric], i))

    return result


def save_official_metrics_to_csv(season: str) -> Path:
    """Save official metrics to CSV for caching.

    Args:
        season: Season string

    Returns:
        Path to saved CSV file
    """
    df = fetch_official_defensive_metrics(season)
    output_path = DATA_DIR / f"external_official_{season}.csv"
    df.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")
    return output_path


# =============================================================================
# Comparison and Correlation Analysis
# =============================================================================


def calculate_correlation_with_edi(
    edi_df: pd.DataFrame, season: str, metric: str = "DEF_RATING"
) -> dict:
    """Calculate Spearman correlation between EDI and official metric.

    Args:
        edi_df: EDI results DataFrame with PLAYER_NAME and EDI_Total
        season: Season string
        metric: 'DEF_RATING' or 'DEF_WS'

    Returns:
        Dict with correlation coefficient, p-value, and sample size
    """
    official_df = fetch_official_defensive_metrics(season)

    # Merge on player name
    merged = edi_df.merge(
        official_df[["PLAYER_NAME", metric]],
        on="PLAYER_NAME",
        how="inner",
    )

    if len(merged) < 10:
        return {"r": None, "p": None, "n": len(merged), "error": "Insufficient overlap"}

    # Calculate Spearman correlation
    # Note: For DEF_RATING, lower is better, so we expect negative correlation with EDI
    # For DEF_WS, higher is better, so we expect positive correlation with EDI
    if metric == "DEF_RATING":
        # Negate DEF_RATING so higher = better (like EDI)
        r, p = stats.spearmanr(merged["EDI_Total"], -merged[metric])
    else:
        r, p = stats.spearmanr(merged["EDI_Total"], merged[metric])

    return {"r": r, "p": p, "n": len(merged)}


def calculate_recall_at_k(
    edi_df: pd.DataFrame,
    season: str,
    all_defense_names: list[str],
    metric: str = "DEF_RATING",
    k_values: list[int] = [10, 20, 30],
) -> dict[int, dict]:
    """Calculate recall of All-Defense players at various K values.

    Args:
        edi_df: EDI results DataFrame
        season: Season string
        all_defense_names: List of All-Defensive team player names
        metric: 'DEF_RATING' or 'DEF_WS'
        k_values: List of K values to evaluate

    Returns:
        Dict mapping K to recall metrics for both EDI and official metric
    """
    official_df = fetch_official_defensive_metrics(season)

    results = {}
    for k in k_values:
        # EDI top K
        edi_top_k = set(
            edi_df.sort_values("EDI_Total", ascending=False).head(k)["PLAYER_NAME"]
        )
        edi_recall = len(edi_top_k & set(all_defense_names))

        # Official metric top K
        if metric == "DEF_RATING":
            official_top_k = set(official_df.head(k)["PLAYER_NAME"])
        else:  # DEF_WS
            official_top_k = set(
                official_df.sort_values("DEF_WS", ascending=False).head(k)[
                    "PLAYER_NAME"
                ]
            )
        official_recall = len(official_top_k & set(all_defense_names))

        results[k] = {
            "edi_recall": edi_recall,
            "edi_top_k": list(edi_top_k & set(all_defense_names)),
            f"{metric.lower()}_recall": official_recall,
            f"{metric.lower()}_top_k": list(official_top_k & set(all_defense_names)),
        }

    return results


def get_dpoy_ranks(
    edi_df: pd.DataFrame, season: str, dpoy_name: str
) -> dict[str, int | None]:
    """Get DPOY's rank in EDI and official metrics.

    Args:
        edi_df: EDI results DataFrame
        season: Season string
        dpoy_name: DPOY player name

    Returns:
        Dict with ranks in EDI, DEF_RATING, and DEF_WS
    """
    official_df = fetch_official_defensive_metrics(season)

    # EDI rank
    edi_sorted = edi_df.sort_values("EDI_Total", ascending=False).reset_index(drop=True)
    edi_match = edi_sorted[
        edi_sorted["PLAYER_NAME"].str.contains(dpoy_name, case=False, na=False)
    ]
    edi_rank = int(edi_match.index[0]) + 1 if len(edi_match) > 0 else None

    # DEF_RATING rank (already sorted ascending = best first)
    rating_match = official_df[
        official_df["PLAYER_NAME"].str.contains(dpoy_name, case=False, na=False)
    ]
    rating_rank = (
        int(rating_match["DEF_RATING_RANK"].iloc[0]) if len(rating_match) > 0 else None
    )

    # DEF_WS rank
    ws_match = official_df[
        official_df["PLAYER_NAME"].str.contains(dpoy_name, case=False, na=False)
    ]
    ws_rank = int(ws_match["DEF_WS_RANK"].iloc[0]) if len(ws_match) > 0 else None

    return {
        "edi_rank": edi_rank,
        "def_rating_rank": rating_rank,
        "def_ws_rank": ws_rank,
    }


# =============================================================================
# Report Generation
# =============================================================================


def generate_external_comparison_report(edi_df: pd.DataFrame, season: str) -> str:
    """Generate comparison report between EDI and official metrics.

    Args:
        edi_df: EDI results DataFrame
        season: Season string

    Returns:
        Formatted report string
    """
    lines = [
        f"\n{'=' * 60}",
        f"EDI vs Official NBA Metrics Comparison: {season}",
        f"{'=' * 60}",
    ]

    # Correlation analysis
    for metric in ["DEF_RATING", "DEF_WS"]:
        corr = calculate_correlation_with_edi(edi_df, season, metric)
        lines.append(f"\n{metric} Correlation:")
        if corr.get("r") is not None:
            lines.append(f"  Spearman r: {corr['r']:.3f} (p={corr['p']:.4f})")
            lines.append(f"  Sample size: {corr['n']} players")
        else:
            lines.append(f"  Error: {corr.get('error', 'Unknown')}")

    # Top 10 comparison
    lines.append(f"\nTop 10 Comparison:")
    lines.append("-" * 50)

    # EDI top 10
    edi_top10 = edi_df.sort_values("EDI_Total", ascending=False).head(10)
    lines.append("\nEDI Top 10:")
    for i, (_, row) in enumerate(edi_top10.iterrows(), 1):
        lines.append(f"  {i:2d}. {row['PLAYER_NAME']}: {row['EDI_Total']:.2f}")

    # DEF_RATING top 10
    official_df = fetch_official_defensive_metrics(season)
    lines.append("\nDEF_RATING Top 10 (lower is better):")
    for i, (_, row) in enumerate(official_df.head(10).iterrows(), 1):
        lines.append(f"  {i:2d}. {row['PLAYER_NAME']}: {row['DEF_RATING']:.1f}")

    # DEF_WS top 10
    ws_top10 = official_df.sort_values("DEF_WS", ascending=False).head(10)
    lines.append("\nDEF_WS Top 10 (higher is better):")
    for i, (_, row) in enumerate(ws_top10.iterrows(), 1):
        lines.append(f"  {i:2d}. {row['PLAYER_NAME']}: {row['DEF_WS']:.2f}")

    return "\n".join(lines)


def generate_multi_season_report() -> str:
    """Generate comparison report across all supported seasons.

    Returns:
        Formatted multi-season report string
    """
    lines = [
        "\n" + "=" * 70,
        "EDI vs Official NBA Metrics: Multi-Season Summary",
        "=" * 70,
    ]

    all_correlations = {"DEF_RATING": [], "DEF_WS": []}

    for season in SUPPORTED_SEASONS:
        edi_path = DATA_DIR / f"nba_defensive_all_players_{season}.csv"
        if not edi_path.exists():
            lines.append(f"\n{season}: EDI data not found, skipping...")
            continue

        edi_df = pd.read_csv(edi_path)
        lines.append(f"\n{season}:")
        lines.append("-" * 40)

        for metric in ["DEF_RATING", "DEF_WS"]:
            corr = calculate_correlation_with_edi(edi_df, season, metric)
            if corr.get("r") is not None:
                all_correlations[metric].append(corr["r"])
                lines.append(f"  {metric}: r={corr['r']:.3f} (n={corr['n']})")
            else:
                lines.append(f"  {metric}: {corr.get('error', 'Error')}")

    # Summary
    lines.append(f"\n{'=' * 70}")
    lines.append("Overall Summary:")
    lines.append("-" * 40)

    for metric in ["DEF_RATING", "DEF_WS"]:
        if all_correlations[metric]:
            avg_r = np.mean(all_correlations[metric])
            lines.append(
                f"  {metric}: Average r = {avg_r:.3f} ({len(all_correlations[metric])} seasons)"
            )

    return "\n".join(lines)


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import sys

    print("NBA Official Defensive Metrics Fetcher")
    print("=" * 50)

    # Check for --all flag for multi-season report
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        report = generate_multi_season_report()
        print(report)
    elif len(sys.argv) > 1 and sys.argv[1] == "--save":
        # Save all seasons to CSV
        for season in SUPPORTED_SEASONS:
            save_official_metrics_to_csv(season)
    else:
        # Demo: Get official data for 2023-24
        season = "2023-24"
        print(f"\nDEF_RATING Top 10 for {season}:")
        for name, value, rank in get_official_top_n(season, "DEF_RATING", 10):
            print(f"  {rank:2d}. {name}: {value:.1f}")

        print(f"\nDEF_WS Top 10 for {season}:")
        for name, value, rank in get_official_top_n(season, "DEF_WS", 10):
            print(f"  {rank:2d}. {name}: {value:.2f}")

        # Try to load EDI data and compare
        edi_path = DATA_DIR / f"nba_defensive_all_players_{season}.csv"
        if edi_path.exists():
            print(f"\nLoading EDI data from: {edi_path}")
            edi_df = pd.read_csv(edi_path)
            report = generate_external_comparison_report(edi_df, season)
            print(report)
        else:
            print(f"\nEDI data not found at: {edi_path}")
            print("Run the main analysis first to generate EDI data.")
