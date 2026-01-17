"""5-Season Cross-Comparison Benchmark: EDI vs NBA Official Metrics

This module provides comprehensive benchmarking of the EDI model against
official NBA defensive metrics (DEF_RATING, DEF_WS) using All-Defensive Team
and DPOY selections as ground truth.

External Metrics:
- DEF_RATING: Defensive Rating (points allowed per 100 possessions, lower is better)
- DEF_WS: Defensive Win Shares (cumulative defensive contribution, higher is better)

Key Metrics:
1. All-Defense Recall Count @10/20/30 (多少名 All-Defense 球员被捕获)
2. DPOY Average Rank across 5 seasons (越低越好)
3. Spearman Rank Correlation (评估指标间的逻辑一致性)

Usage:
    python src/benchmark_evaluation.py --all
    python src/benchmark_evaluation.py 2023-24
"""

import io
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Set UTF-8 encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Local imports
from constants import (
    EVALUATION_SEASONS,
    POOL_SIZE,
    classify_role_3cat,
)
from data_fetcher import get_all_defensive_teams, get_dpoy_winner
from fetch_external import fetch_official_defensive_metrics

# Data directory
DATA_DIR = Path(__file__).parent.parent / "data"
REPORTS_DIR = Path(__file__).parent.parent / "reports"


@dataclass
class MetricResult:
    """Evaluation results for a single metric in a single season."""

    metric_name: str
    season: str

    # Recall counts (absolute numbers, not percentages)
    recall_10: int  # How many All-Defense in Top 10
    recall_20: int  # How many All-Defense in Top 20
    recall_30: int  # How many All-Defense in Top 30

    # DPOY ranking
    dpoy_name: str | None
    dpoy_rank: int | None  # Rank within this metric's list (1-30, or 31 if not found)

    # Pool-restricted average rank of All-Defense players
    all_defense_avg_rank: float  # Average rank of All-Defense players within Top 30


@dataclass
class SeasonBenchmark:
    """Complete benchmark for one season across all metrics."""

    season: str
    dpoy_name: str | None
    dpoy_pos_category: str  # "Backcourt" or "Frontcourt"
    edi: MetricResult
    def_rating: MetricResult | None
    def_ws: MetricResult | None


@dataclass
class AggregateStats:
    """Aggregate statistics across all seasons."""

    # DPOY average ranks (key metric)
    edi_dpoy_avg_rank: float
    def_rating_dpoy_avg_rank: float
    def_ws_dpoy_avg_rank: float

    # Total recall counts across 5 seasons
    edi_total_recall_10: int
    edi_total_recall_20: int
    edi_total_recall_30: int
    def_rating_total_recall_10: int
    def_rating_total_recall_20: int
    def_rating_total_recall_30: int
    def_ws_total_recall_10: int
    def_ws_total_recall_20: int
    def_ws_total_recall_30: int

    # Spearman correlations
    edi_def_rating_correlations: list[float] = field(default_factory=list)
    edi_def_ws_correlations: list[float] = field(default_factory=list)


def get_player_rank_in_list(player_name: str, ranked_list: list[str]) -> int | None:
    """Get 1-indexed rank of a player in a list.

    Returns None if player not found.
    """
    # Try exact match first
    try:
        return ranked_list.index(player_name) + 1
    except ValueError:
        pass

    # Try partial match (case-insensitive)
    player_lower = player_name.lower()
    for i, name in enumerate(ranked_list):
        if player_lower in name.lower() or name.lower() in player_lower:
            return i + 1

    return None


def evaluate_metric(
    player_list: list[str],  # Ordered list of player names (rank 1 = first)
    season: str,
    metric_name: str,
) -> MetricResult:
    """Evaluate a metric against All-Defense ground truth.

    Args:
        player_list: Top 30 players in order (index 0 = rank 1)
        season: Season string
        metric_name: Name of the metric

    Returns:
        MetricResult with recall counts and DPOY rank
    """
    # Get ground truth
    ground_truth = get_all_defensive_teams(season)
    if ground_truth.empty:
        raise ValueError(f"No ground truth available for {season}")

    all_defense_names = set(ground_truth["PLAYER_NAME"])
    dpoy_info = get_dpoy_winner(season)
    dpoy_name = dpoy_info[0] if dpoy_info else None

    # Limit to Top 30
    top_30 = player_list[:POOL_SIZE]
    top_20 = player_list[:20]
    top_10 = player_list[:10]

    # Calculate recall counts
    recall_10 = len(all_defense_names & set(top_10))
    recall_20 = len(all_defense_names & set(top_20))
    recall_30 = len(all_defense_names & set(top_30))

    # DPOY rank (31 if not in Top 30)
    dpoy_rank = None
    if dpoy_name:
        dpoy_rank = get_player_rank_in_list(dpoy_name, top_30)
        if dpoy_rank is None:
            dpoy_rank = 31  # Not in Top 30

    # Calculate average rank of All-Defense players within Top 30
    ranks_in_pool = []
    for name in all_defense_names:
        rank = get_player_rank_in_list(name, top_30)
        if rank is not None:
            ranks_in_pool.append(rank)
        else:
            ranks_in_pool.append(31)  # Not found = 31

    all_defense_avg_rank = float(np.mean(ranks_in_pool)) if ranks_in_pool else 31.0

    return MetricResult(
        metric_name=metric_name,
        season=season,
        recall_10=recall_10,
        recall_20=recall_20,
        recall_30=recall_30,
        dpoy_name=dpoy_name,
        dpoy_rank=dpoy_rank,
        all_defense_avg_rank=all_defense_avg_rank,
    )


def create_unified_pool(
    season: str, min_gp: int = 40, min_mpg: float = 20.0
) -> pd.DataFrame | None:
    """Create a unified player pool with EDI and official metrics.

    This ensures all metrics are compared on the same sample of players.
    Uses 3-category position classification: Backcourt, Roamer, Frontcourt.

    Args:
        season: Season string
        min_gp: Minimum games played filter
        min_mpg: Minimum minutes per game filter

    Returns:
        DataFrame with PLAYER_NAME, PLAYER_POSITION, Roamer_Pct, EDI_Total, DEF_RATING, DEF_WS, and ranks
    """
    # Load EDI data
    edi_path = DATA_DIR / f"nba_defensive_all_players_{season}.csv"
    if not edi_path.exists():
        return None

    edi_df = pd.read_csv(edi_path)

    # Ensure Roamer_Pct column exists
    if "Roamer_Pct" not in edi_df.columns:
        edi_df["Roamer_Pct"] = 0.0

    # Load official metrics (already filtered by GP/MPG)
    try:
        official_df = fetch_official_defensive_metrics(season, min_gp, min_mpg)
    except Exception as e:
        print(f"  Error fetching official metrics: {e}")
        return None

    # Merge on player name - only keep players in BOTH datasets
    # Keep PLAYER_POSITION and Roamer_Pct from EDI data
    merged = edi_df[
        ["PLAYER_NAME", "PLAYER_POSITION", "Roamer_Pct", "EDI_Total"]
    ].merge(
        official_df[["PLAYER_NAME", "DEF_RATING", "DEF_WS"]],
        on="PLAYER_NAME",
        how="inner",
    )

    if len(merged) < 30:
        print(f"  Warning: Only {len(merged)} players in unified pool for {season}")

    # Add 3-category position classification (Backcourt, Roamer, Frontcourt)
    # Uses Roamer_Pct from EDI data to determine if swing position is a Roamer
    merged["POS_CATEGORY"] = merged.apply(
        lambda r: classify_role_3cat(
            r["PLAYER_POSITION"], r.get("Roamer_Pct", 0.0), threshold=0.15
        ),
        axis=1,
    )

    # Calculate ranks within the unified pool (overall)
    # EDI: higher is better (rank descending)
    merged["EDI_RANK"] = merged["EDI_Total"].rank(ascending=False, method="min")
    # DEF_RATING: lower is better (rank ascending)
    merged["DEF_RATING_RANK"] = merged["DEF_RATING"].rank(ascending=True, method="min")
    # DEF_WS: higher is better (rank descending)
    merged["DEF_WS_RANK"] = merged["DEF_WS"].rank(ascending=False, method="min")

    print(f"    Unified pool size: {len(merged)} players")

    return merged


def get_positional_rank(
    pool_df: pd.DataFrame, player_name: str, metric: str
) -> tuple[int | None, int, str]:
    """Get a player's rank within their position category.

    Args:
        pool_df: Unified pool DataFrame with POS_CATEGORY
        player_name: Player name to look up
        metric: 'EDI', 'DEF_RATING', or 'DEF_WS'

    Returns:
        Tuple of (positional_rank, pool_size, position_category)
        Returns (None, 0, '') if player not found
    """
    # Find the player
    match = pool_df[pool_df["PLAYER_NAME"] == player_name]

    # Try partial match if exact match fails
    if len(match) == 0:
        player_lower = player_name.lower()
        for _, row in pool_df.iterrows():
            if (
                player_lower in row["PLAYER_NAME"].lower()
                or row["PLAYER_NAME"].lower() in player_lower
            ):
                match = pool_df[pool_df["PLAYER_NAME"] == row["PLAYER_NAME"]]
                break

    if len(match) == 0:
        return None, 0, ""

    player_row = match.iloc[0]
    pos_category = player_row["POS_CATEGORY"]

    # Filter pool to same position category
    pos_pool = pool_df[pool_df["POS_CATEGORY"] == pos_category].copy()

    # Calculate positional ranks
    if metric == "EDI":
        pos_pool["POS_RANK"] = pos_pool["EDI_Total"].rank(ascending=False, method="min")
    elif metric == "DEF_RATING":
        pos_pool["POS_RANK"] = pos_pool["DEF_RATING"].rank(ascending=True, method="min")
    elif metric == "DEF_WS":
        pos_pool["POS_RANK"] = pos_pool["DEF_WS"].rank(ascending=False, method="min")
    else:
        return None, 0, ""

    # Get the player's positional rank
    player_pos_rank = pos_pool[pos_pool["PLAYER_NAME"] == player_row["PLAYER_NAME"]][
        "POS_RANK"
    ]

    if len(player_pos_rank) == 0:
        return None, len(pos_pool), pos_category

    return int(player_pos_rank.iloc[0]), len(pos_pool), pos_category


def get_top_n_from_pool(pool_df: pd.DataFrame, metric: str, n: int = 30) -> list[str]:
    """Get top N player names from unified pool by a specific metric.

    Args:
        pool_df: Unified pool DataFrame
        metric: 'EDI', 'DEF_RATING', or 'DEF_WS'
        n: Number of top players to return

    Returns:
        List of player names in rank order
    """
    if metric == "EDI":
        df_sorted = pool_df.sort_values("EDI_Total", ascending=False)
    elif metric == "DEF_RATING":
        df_sorted = pool_df.sort_values("DEF_RATING", ascending=True)  # Lower is better
    elif metric == "DEF_WS":
        df_sorted = pool_df.sort_values("DEF_WS", ascending=False)  # Higher is better
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return list(df_sorted.head(n)["PLAYER_NAME"])


def get_player_rank_from_pool(
    pool_df: pd.DataFrame, player_name: str, metric: str
) -> int | None:
    """Get a player's rank in the unified pool for a specific metric.

    Args:
        pool_df: Unified pool DataFrame
        player_name: Player name to look up
        metric: 'EDI', 'DEF_RATING', or 'DEF_WS'

    Returns:
        Player's rank (1-indexed), or None if not found
    """
    rank_col = f"{metric}_RANK"

    # Try exact match
    match = pool_df[pool_df["PLAYER_NAME"] == player_name]
    if len(match) > 0:
        return int(match[rank_col].iloc[0])

    # Try partial match
    player_lower = player_name.lower()
    for _, row in pool_df.iterrows():
        if (
            player_lower in row["PLAYER_NAME"].lower()
            or row["PLAYER_NAME"].lower() in player_lower
        ):
            return int(row[rank_col])

    return None


def get_edi_top_30(season: str) -> list[str] | None:
    """Get EDI Top 30 player names for a season (legacy function for compatibility)."""
    edi_path = DATA_DIR / f"nba_defensive_all_players_{season}.csv"
    if not edi_path.exists():
        return None

    df = pd.read_csv(edi_path)
    df = df.sort_values("EDI_Total", ascending=False).reset_index(drop=True)
    return list(df.head(POOL_SIZE)["PLAYER_NAME"])


def get_official_top_30(season: str, metric: str) -> list[str] | None:
    """Get official NBA metric Top 30 player names for a season.

    Args:
        season: Season string
        metric: 'DEF_RATING' or 'DEF_WS'

    Returns:
        List of top 30 player names, or None if data unavailable
    """
    try:
        df = fetch_official_defensive_metrics(season)

        if metric == "DEF_RATING":
            # Already sorted by DEF_RATING ascending (lower is better)
            return list(df.head(POOL_SIZE)["PLAYER_NAME"])
        elif metric == "DEF_WS":
            # Sort by DEF_WS descending (higher is better)
            df_sorted = df.sort_values("DEF_WS", ascending=False)
            return list(df_sorted.head(POOL_SIZE)["PLAYER_NAME"])
        else:
            return None
    except Exception as e:
        print(f"  Error fetching {metric} for {season}: {e}")
        return None


def calculate_spearman_correlation(
    edi_list: list[str],
    ext_list: list[str],
) -> float | None:
    """Calculate Spearman correlation between two ranked lists.

    Uses common players between the two lists.
    """
    # Find common players
    common = set(edi_list) & set(ext_list)

    if len(common) < 5:
        return None

    # Get ranks for common players
    edi_ranks = []
    ext_ranks = []

    for name in common:
        edi_r = get_player_rank_in_list(name, edi_list)
        ext_r = get_player_rank_in_list(name, ext_list)
        if edi_r and ext_r:
            edi_ranks.append(edi_r)
            ext_ranks.append(ext_r)

    if len(edi_ranks) < 5:
        return None

    r, _ = stats.spearmanr(edi_ranks, ext_ranks)
    return float(r)


def benchmark_season(season: str) -> SeasonBenchmark | None:
    """Run complete benchmark for a single season using unified pool with positional ranking."""
    print(f"\nBenchmarking {season}...")

    # Create unified pool (same players, same sample size for all metrics)
    pool = create_unified_pool(season)
    if pool is None or len(pool) < 30:
        print(f"  Warning: Could not create unified pool for {season}")
        return None

    # Get top 30 from unified pool for each metric
    edi_list = get_top_n_from_pool(pool, "EDI", POOL_SIZE)
    def_rating_list = get_top_n_from_pool(pool, "DEF_RATING", POOL_SIZE)
    def_ws_list = get_top_n_from_pool(pool, "DEF_WS", POOL_SIZE)

    # Evaluate each metric against All-Defense ground truth
    edi_result = evaluate_metric(edi_list, season, "EDI")
    def_rating_result = evaluate_metric(def_rating_list, season, "DEF_RATING")
    def_ws_result = evaluate_metric(def_ws_list, season, "DEF_WS")

    # Get DPOY info and calculate POSITIONAL ranks from unified pool
    dpoy_info = get_dpoy_winner(season)
    dpoy_name = dpoy_info[0] if dpoy_info else None

    # Calculate DPOY positional ranks (within same position category)
    if dpoy_name:
        edi_pos_rank, edi_pos_size, pos_cat = get_positional_rank(
            pool, dpoy_name, "EDI"
        )
        def_rating_pos_rank, _, _ = get_positional_rank(pool, dpoy_name, "DEF_RATING")
        def_ws_pos_rank, _, _ = get_positional_rank(pool, dpoy_name, "DEF_WS")

        # Update results with positional ranks
        edi_result.dpoy_rank = edi_pos_rank if edi_pos_rank else edi_pos_size + 1
        def_rating_result.dpoy_rank = (
            def_rating_pos_rank if def_rating_pos_rank else edi_pos_size + 1
        )
        def_ws_result.dpoy_rank = (
            def_ws_pos_rank if def_ws_pos_rank else edi_pos_size + 1
        )

        print(
            f"    DPOY {dpoy_name} ({pos_cat}, n={edi_pos_size}): "
            f"EDI #{edi_pos_rank}, DEF_RATING #{def_rating_pos_rank}, DEF_WS #{def_ws_pos_rank}"
        )
    else:
        pos_cat = ""

    return SeasonBenchmark(
        season=season,
        dpoy_name=dpoy_name,
        dpoy_pos_category=pos_cat,
        edi=edi_result,
        def_rating=def_rating_result,
        def_ws=def_ws_result,
    )


def calculate_aggregate_stats(
    results: list[SeasonBenchmark],
) -> AggregateStats:
    """Calculate aggregate statistics across all seasons."""
    # DPOY ranks
    edi_dpoy_ranks = [r.edi.dpoy_rank for r in results if r.edi.dpoy_rank]
    def_rating_dpoy_ranks = [
        r.def_rating.dpoy_rank
        for r in results
        if r.def_rating and r.def_rating.dpoy_rank
    ]
    def_ws_dpoy_ranks = [
        r.def_ws.dpoy_rank for r in results if r.def_ws and r.def_ws.dpoy_rank
    ]

    # Total recall counts
    edi_r10 = sum(r.edi.recall_10 for r in results)
    edi_r20 = sum(r.edi.recall_20 for r in results)
    edi_r30 = sum(r.edi.recall_30 for r in results)

    dr_r10 = sum(r.def_rating.recall_10 for r in results if r.def_rating)
    dr_r20 = sum(r.def_rating.recall_20 for r in results if r.def_rating)
    dr_r30 = sum(r.def_rating.recall_30 for r in results if r.def_rating)

    dw_r10 = sum(r.def_ws.recall_10 for r in results if r.def_ws)
    dw_r20 = sum(r.def_ws.recall_20 for r in results if r.def_ws)
    dw_r30 = sum(r.def_ws.recall_30 for r in results if r.def_ws)

    # Spearman correlations using unified pool (same sample for all metrics)
    edi_def_rating_corrs = []
    edi_def_ws_corrs = []

    for r in results:
        pool = create_unified_pool(r.season)
        if pool is not None and len(pool) >= 30:
            # Calculate correlation on the full unified pool
            # EDI vs DEF_RATING (note: DEF_RATING lower is better, so negate)
            corr_rating, _ = stats.spearmanr(pool["EDI_Total"], -pool["DEF_RATING"])
            edi_def_rating_corrs.append(float(corr_rating))

            # EDI vs DEF_WS (both higher is better)
            corr_ws, _ = stats.spearmanr(pool["EDI_Total"], pool["DEF_WS"])
            edi_def_ws_corrs.append(float(corr_ws))

    return AggregateStats(
        edi_dpoy_avg_rank=float(np.mean(edi_dpoy_ranks)) if edi_dpoy_ranks else 999.0,
        def_rating_dpoy_avg_rank=float(np.mean(def_rating_dpoy_ranks))
        if def_rating_dpoy_ranks
        else 999.0,
        def_ws_dpoy_avg_rank=float(np.mean(def_ws_dpoy_ranks))
        if def_ws_dpoy_ranks
        else 999.0,
        edi_total_recall_10=edi_r10,
        edi_total_recall_20=edi_r20,
        edi_total_recall_30=edi_r30,
        def_rating_total_recall_10=dr_r10,
        def_rating_total_recall_20=dr_r20,
        def_rating_total_recall_30=dr_r30,
        def_ws_total_recall_10=dw_r10,
        def_ws_total_recall_20=dw_r20,
        def_ws_total_recall_30=dw_r30,
        edi_def_rating_correlations=edi_def_rating_corrs,
        edi_def_ws_correlations=edi_def_ws_corrs,
    )


def generate_report(
    results: list[SeasonBenchmark],
    agg: AggregateStats,
) -> str:
    """Generate Markdown benchmark report."""
    lines = []

    # Header
    lines.append("# EDI Model Benchmark Report (V0.65)")
    lines.append("")
    lines.append("## Cross-Comparison: EDI vs NBA Official Metrics")
    lines.append("")
    lines.append("### External Metrics Used")
    lines.append(
        "- **DEF_RATING**: Defensive Rating (points allowed per 100 possessions, lower is better)"
    )
    lines.append(
        "- **DEF_WS**: Defensive Win Shares (cumulative defensive contribution, higher is better)"
    )
    lines.append("")
    lines.append("### Data Source")
    lines.append("- NBA Official API (nba.com/stats)")
    lines.append("- Filter: GP >= 40, MPG >= 20")
    lines.append("")
    lines.append("### Methodology Notes")
    lines.append(
        "- **DPOY Ranking**: Uses **3-category positional ranking** (Backcourt, Roamer, Frontcourt) "
        "for fair comparison across different player archetypes."
    )
    lines.append(
        "- **Roamer**: Swing positions (F, F-C, C-F) with Roamer_Pct >= 0.15 (sweep/help defenders like JJJ, Giannis)."
    )
    lines.append(
        "- **Frontcourt**: Pure centers (C) like Gobert stay Frontcourt regardless of Roamer_Pct."
    )
    lines.append(
        "- **Recall@K**: Uses overall Top-K from unified pool (same sample for all metrics)."
    )
    lines.append(
        "- **Spearman Correlation**: Calculated on full unified pool (~220 players per season)."
    )
    lines.append("")
    lines.append(f"**Seasons Evaluated:** {', '.join([r.season for r in results])}")
    lines.append(f"**Pool Size:** Top {POOL_SIZE} players per metric")
    lines.append("")

    # Executive Summary
    lines.append("---")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("### DPOY Average Rank (5-Season Mean)")
    lines.append("")
    lines.append("| Metric | DPOY Avg Rank | Interpretation |")
    lines.append("|--------|---------------|----------------|")

    # Sort by DPOY rank (lower is better)
    dpoy_ranks = [
        ("EDI", agg.edi_dpoy_avg_rank),
        ("DEF_RATING", agg.def_rating_dpoy_avg_rank),
        ("DEF_WS", agg.def_ws_dpoy_avg_rank),
    ]
    dpoy_ranks.sort(key=lambda x: x[1])

    for i, (name, rank) in enumerate(dpoy_ranks):
        interp = "Best" if i == 0 else ""
        lines.append(f"| {name} | {rank:.1f} | {interp} |")
    lines.append("")

    # Recall Summary
    lines.append("### All-Defense Recall (5-Season Total, 50 players)")
    lines.append("")
    lines.append("| Metric | Recall@10 | Recall@20 | Recall@30 |")
    lines.append("|--------|-----------|-----------|-----------|")
    lines.append(
        f"| EDI | {agg.edi_total_recall_10}/50 | {agg.edi_total_recall_20}/50 | {agg.edi_total_recall_30}/50 |"
    )
    lines.append(
        f"| DEF_RATING | {agg.def_rating_total_recall_10}/50 | {agg.def_rating_total_recall_20}/50 | {agg.def_rating_total_recall_30}/50 |"
    )
    lines.append(
        f"| DEF_WS | {agg.def_ws_total_recall_10}/50 | {agg.def_ws_total_recall_20}/50 | {agg.def_ws_total_recall_30}/50 |"
    )
    lines.append("")

    # Per-Season Details
    lines.append("---")
    lines.append("## Per-Season Results")
    lines.append("")

    # DPOY Table
    lines.append("### DPOY Ranking by Season (Positional Rank)")
    lines.append("")
    lines.append(
        "*Ranks are calculated within the player's position category "
        "(Backcourt, Roamer, or Frontcourt).*"
    )
    lines.append("")
    lines.append(
        "| Season | DPOY | Category | EDI Rank | DEF_RATING Rank | DEF_WS Rank |"
    )
    lines.append(
        "|--------|------|----------|----------|-----------------|-------------|"
    )

    for r in results:
        dpoy = r.dpoy_name or "N/A"
        # Show full category name: B=Backcourt, R=Roamer, F=Frontcourt
        pos_map = {
            "Backcourt": "Backcourt",
            "Roamer": "Roamer",
            "Frontcourt": "Frontcourt",
        }
        pos = (
            pos_map.get(r.dpoy_pos_category, r.dpoy_pos_category)
            if r.dpoy_pos_category
            else "?"
        )
        edi_r = (
            f"#{r.edi.dpoy_rank}"
            if r.edi.dpoy_rank and r.edi.dpoy_rank <= 30
            else ">30"
        )
        dr_r = (
            f"#{r.def_rating.dpoy_rank}"
            if r.def_rating and r.def_rating.dpoy_rank and r.def_rating.dpoy_rank <= 30
            else ">30"
            if r.def_rating
            else "N/A"
        )
        dw_r = (
            f"#{r.def_ws.dpoy_rank}"
            if r.def_ws and r.def_ws.dpoy_rank and r.def_ws.dpoy_rank <= 30
            else ">30"
            if r.def_ws
            else "N/A"
        )
        lines.append(f"| {r.season} | {dpoy} | {pos} | {edi_r} | {dr_r} | {dw_r} |")
    lines.append("")

    # Recall Table
    lines.append("### All-Defense Recall by Season")
    lines.append("")
    lines.append(
        "| Season | EDI @10 | EDI @20 | EDI @30 | DEF_RATING @30 | DEF_WS @30 |"
    )
    lines.append(
        "|--------|---------|---------|---------|----------------|------------|"
    )

    for r in results:
        dr_30 = str(r.def_rating.recall_30) if r.def_rating else "N/A"
        dw_30 = str(r.def_ws.recall_30) if r.def_ws else "N/A"
        lines.append(
            f"| {r.season} | {r.edi.recall_10} | {r.edi.recall_20} | {r.edi.recall_30} | {dr_30} | {dw_30} |"
        )
    lines.append("")

    # Spearman Correlation
    lines.append("---")
    lines.append("## Spearman Rank Correlation")
    lines.append("")
    lines.append("| Season | EDI vs DEF_RATING | EDI vs DEF_WS |")
    lines.append("|--------|-------------------|---------------|")

    for i, r in enumerate(results):
        dr_corr = (
            f"{agg.edi_def_rating_correlations[i]:.3f}"
            if i < len(agg.edi_def_rating_correlations)
            else "N/A"
        )
        dw_corr = (
            f"{agg.edi_def_ws_correlations[i]:.3f}"
            if i < len(agg.edi_def_ws_correlations)
            else "N/A"
        )
        lines.append(f"| {r.season} | {dr_corr} | {dw_corr} |")

    # Average correlation
    if agg.edi_def_rating_correlations:
        avg_dr = np.mean(agg.edi_def_rating_correlations)
        avg_dw = (
            np.mean(agg.edi_def_ws_correlations) if agg.edi_def_ws_correlations else 0
        )
        lines.append(f"| **Average** | **{avg_dr:.3f}** | **{avg_dw:.3f}** |")
    lines.append("")

    # Interpretation
    lines.append("---")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("### Correlation Strength Guide")
    lines.append("- 0.8+ : Very Strong (metrics measure similar constructs)")
    lines.append("- 0.6-0.8 : Strong")
    lines.append("- 0.4-0.6 : Moderate")
    lines.append("- <0.4 : Weak (metrics diverge significantly)")
    lines.append("")

    # Key Findings
    lines.append("### Key Findings")
    lines.append("")

    # Determine winner for DPOY
    if agg.edi_dpoy_avg_rank <= min(
        agg.def_rating_dpoy_avg_rank, agg.def_ws_dpoy_avg_rank
    ):
        lines.append(
            f"1. **EDI achieves the best DPOY average rank ({agg.edi_dpoy_avg_rank:.1f})**, "
            f"outperforming DEF_RATING ({agg.def_rating_dpoy_avg_rank:.1f}) and DEF_WS ({agg.def_ws_dpoy_avg_rank:.1f})."
        )
    else:
        best_metric = (
            "DEF_RATING"
            if agg.def_rating_dpoy_avg_rank < agg.def_ws_dpoy_avg_rank
            else "DEF_WS"
        )
        best_rank = min(agg.def_rating_dpoy_avg_rank, agg.def_ws_dpoy_avg_rank)
        lines.append(
            f"1. **{best_metric} achieves the best DPOY average rank ({best_rank:.1f})**. "
            f"EDI ranks at {agg.edi_dpoy_avg_rank:.1f}."
        )

    # Recall comparison
    edi_r30 = agg.edi_total_recall_30
    dr_r30 = agg.def_rating_total_recall_30
    dw_r30 = agg.def_ws_total_recall_30

    if edi_r30 >= max(dr_r30, dw_r30):
        lines.append(
            f"2. **EDI captures {edi_r30}/50 All-Defense players in Top 30**, "
            f"matching or exceeding official metrics."
        )
    else:
        best = "DEF_RATING" if dr_r30 > dw_r30 else "DEF_WS"
        best_count = max(dr_r30, dw_r30)
        lines.append(
            f"2. {best} captures {best_count}/50 All-Defense players in Top 30, "
            f"while EDI captures {edi_r30}/50."
        )

    # Correlation insight
    if agg.edi_def_ws_correlations:
        avg_corr = np.mean(agg.edi_def_ws_correlations)
        if avg_corr >= 0.6:
            lines.append(
                f"3. **EDI shows strong correlation (r={avg_corr:.2f}) with DEF_WS**, "
                "indicating aligned defensive evaluation logic."
            )
        elif avg_corr >= 0.4:
            lines.append(
                f"3. EDI shows moderate correlation (r={avg_corr:.2f}) with DEF_WS, "
                "suggesting some divergence in evaluation criteria."
            )
        else:
            lines.append(
                f"3. EDI shows weak correlation (r={avg_corr:.2f}) with DEF_WS, "
                "indicating fundamentally different evaluation approaches."
            )

    lines.append("")
    lines.append("---")
    lines.append("*Generated by benchmark_evaluation.py*")

    return "\n".join(lines)


def run_full_benchmark(
    seasons: list[str] | None = None,
) -> tuple[list[SeasonBenchmark], AggregateStats]:
    """Run benchmark across all specified seasons."""
    if seasons is None:
        seasons = EVALUATION_SEASONS

    results = []
    for season in seasons:
        result = benchmark_season(season)
        if result:
            results.append(result)

    agg = calculate_aggregate_stats(results)
    return results, agg


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark EDI against official NBA defensive metrics"
    )
    parser.add_argument(
        "season",
        nargs="?",
        default=None,
        help="Season to evaluate (e.g., 2023-24). Use --all for all seasons.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all 5 seasons (2019-20 to 2023-24)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path for Markdown report",
    )

    args = parser.parse_args()

    # Determine seasons to evaluate
    if args.all:
        seasons = EVALUATION_SEASONS
    elif args.season:
        seasons = [args.season]
    else:
        seasons = EVALUATION_SEASONS

    print("=" * 60)
    print("EDI Benchmark: Cross-Comparison with NBA Official Metrics")
    print("=" * 60)
    print(f"Seasons: {', '.join(seasons)}")
    print(f"Pool Size: Top {POOL_SIZE}")
    print("")

    # Run benchmark
    results, agg = run_full_benchmark(seasons)

    if not results:
        print("No results generated. Check that EDI data files exist.")
        return

    # Generate report
    report = generate_report(results, agg)

    # Output
    if args.output:
        output_path = Path(args.output)
    else:
        REPORTS_DIR.mkdir(exist_ok=True)
        output_path = REPORTS_DIR / "benchmark_edi_vs_official.md"

    output_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {output_path}")

    # Console summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\n### DPOY Average Rank (5-Season Mean)")
    print(f"  EDI:        {agg.edi_dpoy_avg_rank:.1f}")
    print(f"  DEF_RATING: {agg.def_rating_dpoy_avg_rank:.1f}")
    print(f"  DEF_WS:     {agg.def_ws_dpoy_avg_rank:.1f}")

    print("\n### All-Defense Recall@30 (5-Season Total)")
    print(f"  EDI:        {agg.edi_total_recall_30}/50")
    print(f"  DEF_RATING: {agg.def_rating_total_recall_30}/50")
    print(f"  DEF_WS:     {agg.def_ws_total_recall_30}/50")

    if agg.edi_def_rating_correlations:
        print("\n### Avg Spearman Correlation with EDI")
        print(f"  DEF_RATING: {np.mean(agg.edi_def_rating_correlations):.3f}")
        print(f"  DEF_WS:     {np.mean(agg.edi_def_ws_correlations):.3f}")


if __name__ == "__main__":
    main()
