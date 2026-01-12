"""Evaluation metrics for defensive analysis model.

This module provides comprehensive evaluation of the EDI model against
All-Defensive Team selections using three dimensions:

1. Tier Alignment (梯队重合度)
   - Average rank of All-Defense players in model rankings
   - Lower is better (ideal: ~5.5 for top 10)

2. Candidate Pool Quality (候选池质量)
   - Recall@K: What % of All-Defense players appear in model's top K?
   - Higher is better (measures if model identifies the right pool)

3. Miss Analysis (遗漏偏差分析)
   - Average rank of missed players
   - Classifies misses as "near-miss" (rank 11-20) vs "model blind spot" (rank 30+)

Additional validations:
- Correlation with advanced stats (D-EPM if available)
- Cross-season stability analysis

Evaluation Logic by Era:
- Pre-2023-24: Position-based (4G + 4F + 2C)
- 2023-24+: Positionless top 10, 65-game minimum
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd
import numpy as np

from data_fetcher import (
    get_all_defensive_teams,
    get_dpoy_winner,
    get_min_games_for_awards,
)

# Minimum games required for 2023-24+ seasons
MIN_GAMES_2024 = 65


@dataclass
class TierAlignment:
    """Tier alignment metrics for All-Defense players."""

    avg_rank: float  # Average rank of All-Defense players in model
    median_rank: float  # Median rank (less sensitive to outliers)
    all_in_top20_pct: float  # % of All-Defense in model's top 20
    all_in_top30_pct: float  # % of All-Defense in model's top 30
    player_ranks: list[dict]  # Individual player ranks

    @property
    def grade(self) -> str:
        """Letter grade based on average rank."""
        if self.avg_rank <= 8:
            return "A"
        elif self.avg_rank <= 12:
            return "B"
        elif self.avg_rank <= 18:
            return "C"
        elif self.avg_rank <= 25:
            return "D"
        return "F"


@dataclass
class CandidatePoolQuality:
    """Candidate pool quality metrics (Recall@K)."""

    recall_at_10: float  # Core coverage (same as old metric)
    recall_at_15: float  # Near-miss tolerance
    recall_at_20: float  # Extended candidate pool
    recall_at_30: float  # Broad pool quality
    hits_at_10: list[str]  # Players hit in top 10
    hits_at_20: list[str]  # Players hit in top 20

    @property
    def grade(self) -> str:
        """Letter grade based on Recall@20."""
        if self.recall_at_20 >= 90:
            return "A"
        elif self.recall_at_20 >= 80:
            return "B"
        elif self.recall_at_20 >= 70:
            return "C"
        elif self.recall_at_20 >= 60:
            return "D"
        return "F"


@dataclass
class MissAnalysis:
    """Analysis of missed All-Defense players."""

    near_misses: list[dict]  # Rank 11-20 (understandable)
    moderate_misses: list[dict]  # Rank 21-40 (concerning)
    blind_spots: list[dict]  # Rank 41+ or missing (model failure)
    not_in_data: list[str]  # Players not in dataset
    avg_miss_rank: float  # Average rank of missed players

    @property
    def miss_severity(self) -> str:
        """Categorize overall miss severity."""
        if not self.moderate_misses and not self.blind_spots:
            return "Excellent - all misses are near-misses"
        elif not self.blind_spots:
            return "Good - no blind spots, some moderate misses"
        elif len(self.blind_spots) <= 2:
            return "Fair - few blind spots"
        return "Poor - significant blind spots"


@dataclass
class DPOYEvaluation:
    """DPOY prediction evaluation metrics."""

    season: str
    actual_dpoy_name: str | None  # Actual DPOY winner
    actual_dpoy_id: int | None
    actual_dpoy_rank: int | None  # Where actual DPOY ranks in model
    predicted_dpoy_name: str  # Model's top eligible player
    predicted_dpoy_rank: int  # Should be 1 if top player is eligible
    is_hit: bool  # Did model's prediction match actual?
    eligible_players_checked: int  # How many players checked for eligibility
    min_games_required: int  # Games threshold used

    @property
    def grade(self) -> str:
        """Letter grade based on actual DPOY's rank in model."""
        if self.actual_dpoy_rank is None:
            return "N/A"
        if self.actual_dpoy_rank == 1:
            return "A+"
        elif self.actual_dpoy_rank <= 3:
            return "A"
        elif self.actual_dpoy_rank <= 5:
            return "B"
        elif self.actual_dpoy_rank <= 10:
            return "C"
        elif self.actual_dpoy_rank <= 20:
            return "D"
        return "F"


@dataclass
class BenchmarkResult:
    """Head-to-head comparison between EDI and a competitor metric."""

    season: str
    competitor_name: str  # e.g., "D-RAPTOR"

    # All-Defense average rank comparison (lower is better)
    edi_avg_rank: float
    competitor_avg_rank: float

    # DPOY rank comparison (lower is better)
    edi_dpoy_rank: int | None
    competitor_dpoy_rank: int | None

    # Winner determination
    avg_rank_winner: str  # "EDI", competitor_name, or "TIE"
    dpoy_winner: str  # "EDI", competitor_name, or "TIE"
    overall_winner: str  # "EDI", competitor_name, or "TIE"


@dataclass
class SeasonEvaluation:
    """Complete evaluation for a single season."""

    season: str
    tier_alignment: TierAlignment
    candidate_pool: CandidatePoolQuality
    miss_analysis: MissAnalysis
    is_positionless: bool
    games_filter_applied: bool
    filtered_count: int

    # Position breakdown (only for pre-2023-24)
    position_breakdown: dict[str, dict] | None = None

    # DPOY evaluation
    dpoy_evaluation: DPOYEvaluation | None = None

    # Benchmark against competitors
    benchmark_results: list[BenchmarkResult] | None = None


def classify_positions(pos: str) -> list[str]:
    """Classify player position into eligible positions for evaluation.

    Swing players can be eligible for multiple position pools.

    Args:
        pos: Position string (e.g., "G", "F-C", "C-F", "G-F")

    Returns:
        List of eligible positions ("G", "F", "C")
    """
    if pd.isna(pos) or pos == "":
        return ["F"]  # Default to Forward

    pos = str(pos).upper()

    # Pure Guard (no F or C)
    if "G" in pos and "F" not in pos and "C" not in pos:
        return ["G"]

    # Pure Center
    if pos == "C":
        return ["C"]

    # Pure Forward
    if pos == "F":
        return ["F"]

    # Swing players - can count in multiple pools
    positions = []

    # Guard-Forward swing (G-F, F-G)
    if "G" in pos and "F" in pos and "C" not in pos:
        positions = ["G", "F"]

    # Forward-Center swing (F-C, C-F)
    elif "F" in pos and "C" in pos:
        positions = ["F", "C"]

    # Guard-Center (rare, but handle it)
    elif "G" in pos and "C" in pos:
        positions = ["G", "C"]

    return positions if positions else ["F"]


def classify_position(pos: str) -> str:
    """Classify player position into single G/F/C for backward compatibility."""
    positions = classify_positions(pos)
    return positions[0]


def _get_player_rank(
    model_df: pd.DataFrame, player_name: str, score_col: str, name_col: str
) -> int | None:
    """Get a player's rank in the model (1-indexed)."""
    player_row = model_df[model_df[name_col] == player_name]
    if len(player_row) == 0:
        return None
    scores = player_row[score_col]
    player_score = float(scores.iloc[0])  # type: ignore[union-attr]
    rank = int((model_df[score_col] > player_score).sum()) + 1
    return rank


def calculate_tier_alignment(
    model_df: pd.DataFrame,
    ground_truth_names: set[str],
    score_col: str = "EDI_Total",
    name_col: str = "PLAYER_NAME",
) -> TierAlignment:
    """Calculate tier alignment metrics.

    Measures how well the model ranks actual All-Defense players.
    """
    model_df = model_df.sort_values(score_col, ascending=False).reset_index(drop=True)

    player_ranks = []
    valid_ranks = []

    for name in ground_truth_names:
        rank = _get_player_rank(model_df, name, score_col, name_col)
        player_ranks.append({"name": name, "rank": rank if rank else "N/A"})
        if rank is not None:
            valid_ranks.append(rank)

    # Sort by rank for display
    player_ranks.sort(key=lambda x: x["rank"] if isinstance(x["rank"], int) else 9999)

    if not valid_ranks:
        return TierAlignment(
            avg_rank=999,
            median_rank=999,
            all_in_top20_pct=0,
            all_in_top30_pct=0,
            player_ranks=player_ranks,
        )

    return TierAlignment(
        avg_rank=float(np.mean(valid_ranks)),
        median_rank=float(np.median(valid_ranks)),
        all_in_top20_pct=sum(1 for r in valid_ranks if r <= 20)
        / len(ground_truth_names)
        * 100,
        all_in_top30_pct=sum(1 for r in valid_ranks if r <= 30)
        / len(ground_truth_names)
        * 100,
        player_ranks=player_ranks,
    )


def calculate_candidate_pool_quality(
    model_df: pd.DataFrame,
    ground_truth_names: set[str],
    score_col: str = "EDI_Total",
    name_col: str = "PLAYER_NAME",
) -> CandidatePoolQuality:
    """Calculate candidate pool quality (Recall@K metrics).

    Measures what fraction of All-Defense players appear in model's top K.
    """
    model_df = model_df.sort_values(score_col, ascending=False).reset_index(drop=True)

    def recall_at_k(k: int) -> tuple[float, list[str]]:
        top_k_names = set(model_df.head(k)[name_col])
        hits = ground_truth_names & top_k_names
        return len(hits) / len(ground_truth_names) * 100, list(hits)

    recall_10, hits_10 = recall_at_k(10)
    recall_15, _ = recall_at_k(15)
    recall_20, hits_20 = recall_at_k(20)
    recall_30, _ = recall_at_k(30)

    return CandidatePoolQuality(
        recall_at_10=recall_10,
        recall_at_15=recall_15,
        recall_at_20=recall_20,
        recall_at_30=recall_30,
        hits_at_10=hits_10,
        hits_at_20=hits_20,
    )


def calculate_miss_analysis(
    model_df: pd.DataFrame,
    ground_truth_names: set[str],
    score_col: str = "EDI_Total",
    name_col: str = "PLAYER_NAME",
) -> MissAnalysis:
    """Analyze missed All-Defense players.

    Categorizes misses by severity based on where they rank.
    """
    model_df = model_df.sort_values(score_col, ascending=False).reset_index(drop=True)
    top_10_names = set(model_df.head(10)[name_col])

    missed_names = ground_truth_names - top_10_names

    near_misses = []  # 11-20
    moderate_misses = []  # 21-40
    blind_spots = []  # 41+
    not_in_data = []
    valid_miss_ranks = []

    for name in missed_names:
        rank = _get_player_rank(model_df, name, score_col, name_col)

        if rank is None:
            not_in_data.append(name)
            continue

        valid_miss_ranks.append(rank)
        player_info = {"name": name, "rank": rank}

        if rank <= 20:
            near_misses.append(player_info)
        elif rank <= 40:
            moderate_misses.append(player_info)
        else:
            blind_spots.append(player_info)

    # Sort each category by rank
    near_misses.sort(key=lambda x: x["rank"])
    moderate_misses.sort(key=lambda x: x["rank"])
    blind_spots.sort(key=lambda x: x["rank"])

    avg_miss_rank = float(np.mean(valid_miss_ranks)) if valid_miss_ranks else 0.0

    return MissAnalysis(
        near_misses=near_misses,
        moderate_misses=moderate_misses,
        blind_spots=blind_spots,
        not_in_data=not_in_data,
        avg_miss_rank=avg_miss_rank,
    )


def evaluate_season(
    model_df: pd.DataFrame,
    season: str,
    score_col: str = "EDI_Total",
    position_col: str = "PLAYER_POSITION",
    name_col: str = "PLAYER_NAME",
    games_col: str = "GP",
) -> SeasonEvaluation:
    """Comprehensive evaluation for a single season.

    Uses three-dimensional evaluation:
    1. Tier Alignment - How well All-Defense players rank
    2. Candidate Pool Quality - Recall@K metrics
    3. Miss Analysis - Understanding model blind spots

    Args:
        model_df: DataFrame with model predictions
        season: Season string (e.g., "2023-24")
        score_col: Column name for EDI scores
        position_col: Column name for player positions
        name_col: Column name for player names
        games_col: Column name for games played

    Returns:
        SeasonEvaluation with comprehensive metrics
    """
    # Get ground truth
    ground_truth = get_all_defensive_teams(season)
    if ground_truth.empty:
        raise ValueError(f"No ground truth available for {season}")

    gt_all = set(ground_truth["PLAYER_NAME"])

    # Prepare model data
    model_df = model_df.copy()

    # Determine era and apply filters
    season_year = int(season.split("-")[0])
    is_positionless = season_year >= 2023
    games_filter_applied = False
    filtered_count = 0

    if is_positionless and games_col in model_df.columns:
        original_count = len(model_df)
        model_df = pd.DataFrame(model_df[model_df[games_col] >= MIN_GAMES_2024])
        filtered_count = original_count - len(model_df)
        games_filter_applied = True

    # Sort by score
    model_df = model_df.sort_values(score_col, ascending=False).reset_index(drop=True)

    # Calculate three dimensions
    tier_alignment = calculate_tier_alignment(model_df, gt_all, score_col, name_col)
    candidate_pool = calculate_candidate_pool_quality(
        model_df, gt_all, score_col, name_col
    )
    miss_analysis = calculate_miss_analysis(model_df, gt_all, score_col, name_col)

    # Position breakdown for pre-2023-24
    position_breakdown = None
    if not is_positionless:
        position_breakdown = _calculate_position_breakdown(
            model_df, ground_truth, score_col, position_col, name_col
        )

    return SeasonEvaluation(
        season=season,
        tier_alignment=tier_alignment,
        candidate_pool=candidate_pool,
        miss_analysis=miss_analysis,
        is_positionless=is_positionless,
        games_filter_applied=games_filter_applied,
        filtered_count=filtered_count,
        position_breakdown=position_breakdown,
    )


def _calculate_position_breakdown(
    model_df: pd.DataFrame,
    ground_truth: pd.DataFrame,
    score_col: str,
    position_col: str,
    name_col: str,
) -> dict[str, dict]:
    """Calculate position-specific metrics for pre-2023-24 seasons."""

    # Add position classification
    model_df = model_df.copy()
    model_df["_eval_positions"] = model_df[position_col].apply(classify_positions)

    def get_top_n_for_position(pos: str, n: int) -> list[str]:
        eligible = model_df[model_df["_eval_positions"].apply(lambda x: pos in x)]
        return list(eligible.head(n)[name_col])

    breakdown = {}
    for pos, n, label in [("G", 4, "guard"), ("F", 4, "forward"), ("C", 2, "center")]:
        gt_pos = set(ground_truth[ground_truth["POSITION"] == pos]["PLAYER_NAME"])
        picks = get_top_n_for_position(pos, n)
        hits = [p for p in picks if p in gt_pos]

        breakdown[label] = {
            "picks": picks,
            "hits": hits,
            "recall": len(hits) / n * 100 if n > 0 else 0,
            "target": n,
        }

    return breakdown


def calculate_stability_metrics(
    season_results: list[SeasonEvaluation],
) -> dict[str, Any]:
    """Calculate cross-season stability metrics.

    Measures how consistent the model performs across seasons.
    """
    if len(season_results) < 2:
        return {"note": "Need 2+ seasons for stability analysis"}

    avg_ranks = [r.tier_alignment.avg_rank for r in season_results]
    recall_20s = [r.candidate_pool.recall_at_20 for r in season_results]

    return {
        "avg_rank_mean": float(np.mean(avg_ranks)),
        "avg_rank_std": float(np.std(avg_ranks)),
        "avg_rank_range": (min(avg_ranks), max(avg_ranks)),
        "recall_20_mean": float(np.mean(recall_20s)),
        "recall_20_std": float(np.std(recall_20s)),
        "recall_20_range": (min(recall_20s), max(recall_20s)),
        "consistency_grade": "A"
        if np.std(avg_ranks) < 3
        else "B"
        if np.std(avg_ranks) < 5
        else "C",
    }


def calculate_component_correlation(
    model_df: pd.DataFrame,
    ground_truth_names: set[str],
    name_col: str = "PLAYER_NAME",
) -> dict[str, Any]:
    """Calculate correlation between model components and All-Defense selection.

    This validates whether each component (D1-D5) contributes meaningfully
    to identifying All-Defense players.

    Args:
        model_df: DataFrame with model predictions
        ground_truth_names: Set of All-Defense player names
        name_col: Column name for player names

    Returns:
        Dictionary with component correlation metrics
    """
    model_df = model_df.copy()

    # Create binary target: 1 if All-Defense, 0 otherwise
    gt_list = list(ground_truth_names)  # Convert set to list for pandas
    model_df["is_all_defense"] = model_df[name_col].isin(gt_list).astype(int)

    # Components to analyze
    score_cols = [
        "D1_Score",
        "D2_Score",
        "D3_Score",
        "D4_Score",
        "D5_Score",
        "EDI_Total",
    ]
    available_cols = [c for c in score_cols if c in model_df.columns]

    correlations = {}
    for col in available_cols:
        col_series = model_df[col]
        target_series = model_df["is_all_defense"]
        if col_series.std() > 0:  # Avoid division by zero
            corr = col_series.corr(target_series)  # type: ignore[arg-type]
            correlations[col] = round(float(corr), 3)  # type: ignore[arg-type]

    # Calculate average score for All-Defense vs others
    all_defense_df = model_df[model_df["is_all_defense"] == 1]
    others_df = model_df[model_df["is_all_defense"] == 0]

    score_comparison = {}
    for col in available_cols:
        if col in model_df.columns:
            ad_mean = (
                float(all_defense_df[col].mean()) if len(all_defense_df) > 0 else 0
            )
            other_mean = float(others_df[col].mean()) if len(others_df) > 0 else 0
            score_comparison[col] = {
                "all_defense_avg": round(ad_mean, 3),
                "others_avg": round(other_mean, 3),
                "difference": round(ad_mean - other_mean, 3),
            }

    return {
        "correlations": correlations,
        "score_comparison": score_comparison,
    }


def calculate_model_diagnostics(
    model_df: pd.DataFrame,
    ground_truth_names: set[str],
    score_col: str = "EDI_Total",
    name_col: str = "PLAYER_NAME",
) -> dict[str, Any]:
    """Calculate comprehensive model diagnostics.

    Includes:
    - Component correlation analysis
    - Score distribution analysis
    - Separation metrics (how well model separates All-Defense from others)
    """
    model_df = model_df.sort_values(score_col, ascending=False).reset_index(drop=True)

    # Component correlation
    component_corr = calculate_component_correlation(
        model_df, ground_truth_names, name_col
    )

    # Score distribution
    gt_list = list(ground_truth_names)
    all_defense_df = model_df[model_df[name_col].isin(gt_list)]
    others_df = model_df[~model_df[name_col].isin(gt_list)]

    if len(all_defense_df) > 0 and len(others_df) > 0:
        ad_scores = all_defense_df[score_col]
        other_scores = others_df[score_col]

        # Effect size (Cohen's d) - measures separation
        pooled_std = np.sqrt((ad_scores.std() ** 2 + other_scores.std() ** 2) / 2)
        cohens_d = (
            (ad_scores.mean() - other_scores.mean()) / pooled_std
            if pooled_std > 0
            else 0
        )

        score_distribution = {
            "all_defense": {
                "mean": round(float(ad_scores.mean()), 3),
                "std": round(float(ad_scores.std()), 3),
                "min": round(float(ad_scores.min()), 3),
                "max": round(float(ad_scores.max()), 3),
            },
            "others": {
                "mean": round(float(other_scores.mean()), 3),
                "std": round(float(other_scores.std()), 3),
                "min": round(float(other_scores.min()), 3),
                "max": round(float(other_scores.max()), 3),
            },
            "separation": {
                "cohens_d": round(float(cohens_d), 3),
                "interpretation": "Large"
                if abs(cohens_d) >= 0.8
                else "Medium"
                if abs(cohens_d) >= 0.5
                else "Small",
            },
        }
    else:
        score_distribution = {"error": "Insufficient data for distribution analysis"}

    return {
        "component_correlation": component_corr,
        "score_distribution": score_distribution,
    }


def generate_season_report(eval_result: SeasonEvaluation) -> str:
    """Generate detailed text report for a single season evaluation."""
    lines = []
    season = eval_result.season
    era = "Positionless Era" if eval_result.is_positionless else "Position-Based Era"

    lines.append(f"{'=' * 60}")
    lines.append(f"Season: {season} ({era})")
    lines.append(f"{'=' * 60}")

    if eval_result.games_filter_applied:
        lines.append(
            f"Note: 65-game minimum applied, {eval_result.filtered_count} players filtered out"
        )
    lines.append("")

    # 1. Tier Alignment
    ta = eval_result.tier_alignment
    lines.append("📊 1. TIER ALIGNMENT (梯队重合度)")
    lines.append("-" * 40)
    lines.append(f"  Average Rank of All-Defense Players: {ta.avg_rank:.1f}")
    lines.append(f"  Median Rank: {ta.median_rank:.1f}")
    lines.append(f"  Grade: {ta.grade}")
    lines.append(f"  In Top 20: {ta.all_in_top20_pct:.0f}%")
    lines.append(f"  In Top 30: {ta.all_in_top30_pct:.0f}%")
    lines.append("")
    lines.append("  Individual Rankings:")
    for p in ta.player_ranks:
        rank_str = f"#{p['rank']}" if isinstance(p["rank"], int) else p["rank"]
        marker = "✓" if isinstance(p["rank"], int) and p["rank"] <= 10 else " "
        lines.append(f"    {marker} {p['name']}: {rank_str}")
    lines.append("")

    # 2. Candidate Pool Quality
    cp = eval_result.candidate_pool
    lines.append("🎯 2. CANDIDATE POOL QUALITY (候选池质量)")
    lines.append("-" * 40)
    lines.append(f"  Recall@10 (核心覆盖率): {cp.recall_at_10:.0f}%")
    lines.append(f"  Recall@15: {cp.recall_at_15:.0f}%")
    lines.append(f"  Recall@20 (候选池质量): {cp.recall_at_20:.0f}%")
    lines.append(f"  Recall@30: {cp.recall_at_30:.0f}%")
    lines.append(f"  Grade: {cp.grade}")
    lines.append("")

    # 3. Miss Analysis
    ma = eval_result.miss_analysis
    lines.append("🔍 3. MISS ANALYSIS (遗漏偏差分析)")
    lines.append("-" * 40)
    lines.append(f"  Overall: {ma.miss_severity}")
    if ma.avg_miss_rank > 0:
        lines.append(f"  Average Rank of Missed Players: {ma.avg_miss_rank:.1f}")

    if ma.near_misses:
        lines.append(f"  Near Misses (Rank 11-20): {len(ma.near_misses)}")
        for p in ma.near_misses:
            lines.append(f"    - {p['name']}: #{p['rank']}")

    if ma.moderate_misses:
        lines.append(f"  Moderate Misses (Rank 21-40): {len(ma.moderate_misses)}")
        for p in ma.moderate_misses:
            lines.append(f"    - {p['name']}: #{p['rank']}")

    if ma.blind_spots:
        lines.append(f"  ⚠️ Blind Spots (Rank 41+): {len(ma.blind_spots)}")
        for p in ma.blind_spots:
            lines.append(f"    - {p['name']}: #{p['rank']}")

    if ma.not_in_data:
        lines.append(f"  ❓ Not in Dataset: {len(ma.not_in_data)}")
        for name in ma.not_in_data:
            lines.append(f"    - {name}")
    lines.append("")

    # Position breakdown (if applicable)
    if eval_result.position_breakdown:
        lines.append("📍 POSITION BREAKDOWN")
        lines.append("-" * 40)
        for pos_label, data in eval_result.position_breakdown.items():
            lines.append(
                f"  {pos_label.title()}: {len(data['hits'])}/{data['target']} ({data['recall']:.0f}%)"
            )
            lines.append(f"    Picks: {', '.join(data['picks'])}")
            lines.append(
                f"    Hits: {', '.join(data['hits']) if data['hits'] else 'None'}"
            )
        lines.append("")

    return "\n".join(lines)


def generate_multi_season_report(
    season_results: list[SeasonEvaluation],
    stability: dict[str, Any] | None = None,
) -> str:
    """Generate comprehensive report for multi-season evaluation."""
    lines = []

    lines.append("=" * 70)
    lines.append("EDI MODEL EVALUATION REPORT - Three-Dimensional Analysis")
    lines.append("=" * 70)
    lines.append("")

    # Executive Summary
    lines.append("📋 EXECUTIVE SUMMARY")
    lines.append("-" * 50)

    avg_ranks = [r.tier_alignment.avg_rank for r in season_results]
    recall_20s = [r.candidate_pool.recall_at_20 for r in season_results]
    recall_10s = [r.candidate_pool.recall_at_10 for r in season_results]

    lines.append(f"Seasons Evaluated: {len(season_results)}")
    lines.append(
        f"Tier Alignment (Avg Rank): {np.mean(avg_ranks):.1f} (lower is better)"
    )
    lines.append(f"Candidate Pool Quality (Recall@20): {np.mean(recall_20s):.0f}%")
    lines.append(f"Core Coverage (Recall@10): {np.mean(recall_10s):.0f}%")
    lines.append("")

    # Interpretation guide
    lines.append("📖 INTERPRETATION GUIDE")
    lines.append("-" * 50)
    lines.append("• Tier Alignment: If All-Defense players average rank ~10, model")
    lines.append(
        "  identifies them as top-tier defenders even if exact top-10 differs."
    )
    lines.append("• Recall@20: If 90%+ All-Defense players are in model's top-20,")
    lines.append("  the model's 'candidate pool' is solid - just minor ranking diffs.")
    lines.append("• Miss Analysis: Near-misses (rank 11-20) are acceptable;")
    lines.append("  Blind spots (rank 40+) indicate model systematic weakness.")
    lines.append("")

    # Quick comparison table
    lines.append("📊 SEASON COMPARISON")
    lines.append("-" * 50)
    lines.append(
        f"{'Season':<12} {'Avg Rank':<10} {'R@10':<8} {'R@20':<8} {'Grade':<8}"
    )
    lines.append("-" * 50)
    for r in season_results:
        grade = r.tier_alignment.grade
        lines.append(
            f"{r.season:<12} {r.tier_alignment.avg_rank:<10.1f} "
            f"{r.candidate_pool.recall_at_10:<8.0f}% {r.candidate_pool.recall_at_20:<8.0f}% {grade:<8}"
        )
    lines.append("")

    # Stability metrics
    if stability and "avg_rank_mean" in stability:
        lines.append("📈 STABILITY ANALYSIS")
        lines.append("-" * 50)
        lines.append(
            f"Avg Rank Consistency: {stability['avg_rank_mean']:.1f} ± {stability['avg_rank_std']:.1f}"
        )
        lines.append(
            f"Recall@20 Consistency: {stability['recall_20_mean']:.0f}% ± {stability['recall_20_std']:.1f}%"
        )
        lines.append(f"Consistency Grade: {stability['consistency_grade']}")
        lines.append("")

    # Detailed per-season reports
    lines.append("=" * 70)
    lines.append("DETAILED SEASON REPORTS")
    lines.append("=" * 70)

    for eval_result in season_results:
        lines.append("")
        lines.append(generate_season_report(eval_result))

    return "\n".join(lines)


def run_multi_season_evaluation(
    seasons: list[str],
    data_dir: str = "data",
) -> tuple[list[SeasonEvaluation], dict[str, Any]]:
    """Run evaluation across multiple seasons.

    Uses appropriate evaluation method for each era:
    - Pre-2023-24: Position-based with swing player support
    - 2023-24+: Positionless with 65-game minimum

    Args:
        seasons: List of season strings
        data_dir: Directory containing CSV files

    Returns:
        Tuple of (list of SeasonEvaluation, stability metrics)
    """
    from pathlib import Path

    results = []
    errors = []

    for season in seasons:
        csv_path = Path(data_dir) / f"nba_defensive_all_players_{season}.csv"

        if not csv_path.exists():
            errors.append(f"Data file not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path)

        try:
            eval_result = evaluate_season(df, season)
            results.append(eval_result)
        except ValueError as e:
            errors.append(f"{season}: {e}")

    # Calculate stability metrics
    stability = calculate_stability_metrics(results)

    if errors:
        stability["errors"] = errors

    return results, stability


# Legacy compatibility functions


def evaluate_season_positional(
    model_df: pd.DataFrame,
    season: str,
    score_col: str = "EDI_Total",
    position_col: str = "PLAYER_POSITION",
    name_col: str = "PLAYER_NAME",
    games_col: str = "GP",
) -> dict[str, Any]:
    """Legacy wrapper for positional evaluation.

    Converts new SeasonEvaluation to old dict format for compatibility.
    """
    eval_result = evaluate_season(
        model_df, season, score_col, position_col, name_col, games_col
    )

    # Convert to legacy format
    result = {
        "season": season,
        "overall": {
            "coverage": int(eval_result.candidate_pool.recall_at_10 / 10),
            "target": 10,
            "coverage_pct": eval_result.candidate_pool.recall_at_10,
        },
        "missed_players": [
            {"name": p["name"], "edi_rank": p["rank"]}
            for p in eval_result.miss_analysis.near_misses
            + eval_result.miss_analysis.moderate_misses
            + eval_result.miss_analysis.blind_spots
        ]
        + [
            {"name": name, "edi_rank": "N/A"}
            for name in eval_result.miss_analysis.not_in_data
        ],
    }

    if eval_result.position_breakdown:
        for pos in ["guard", "forward", "center"]:
            pb = eval_result.position_breakdown[pos]
            result[pos] = {
                "picks": pb["picks"],
                "hits": pb["hits"],
                "coverage": len(pb["hits"]),
                "target": pb["target"],
                "coverage_pct": pb["recall"],
            }

    return result


def evaluate_season_positionless(
    model_df: pd.DataFrame,
    season: str,
    score_col: str = "EDI_Total",
    name_col: str = "PLAYER_NAME",
    games_col: str = "GP",
) -> dict[str, Any]:
    """Legacy wrapper for positionless evaluation."""
    eval_result = evaluate_season(
        model_df, season, score_col, "PLAYER_POSITION", name_col, games_col
    )

    model_df_sorted = model_df.sort_values(score_col, ascending=False)
    picks = list(model_df_sorted.head(10)[name_col])

    return {
        "season": season,
        "overall": {
            "picks": picks,
            "hits": eval_result.candidate_pool.hits_at_10,
            "coverage": int(eval_result.candidate_pool.recall_at_10 / 10),
            "target": 10,
            "coverage_pct": eval_result.candidate_pool.recall_at_10,
        },
        "missed_players": [
            {"name": p["name"], "edi_rank": p["rank"]}
            for p in eval_result.miss_analysis.near_misses
            + eval_result.miss_analysis.moderate_misses
            + eval_result.miss_analysis.blind_spots
        ]
        + [
            {"name": name, "edi_rank": "N/A"}
            for name in eval_result.miss_analysis.not_in_data
        ],
        "filtered_players": eval_result.filtered_count
        if eval_result.games_filter_applied
        else 0,
    }


# =============================================================================
# DPOY Evaluation Functions
# =============================================================================


def evaluate_dpoy_alignment(
    model_df: pd.DataFrame,
    season: str,
    score_col: str = "EDI_Total",
    name_col: str = "PLAYER_NAME",
    games_col: str = "GP",
) -> DPOYEvaluation:
    """Evaluate how well the model predicts DPOY.

    This function:
    1. Gets the actual DPOY winner for the season
    2. Finds where the actual DPOY ranks in the model
    3. Finds the model's top eligible player as "predicted DPOY"
    4. Compares prediction vs actual

    Eligibility: Player must meet minimum games threshold (65 for 2024+, 50 otherwise).
    This matters because players like Wembanyama might rank #1 statistically
    but not meet games threshold.

    Args:
        model_df: DataFrame with model predictions
        season: Season string (e.g., "2023-24")
        score_col: Column name for EDI scores
        name_col: Column name for player names
        games_col: Column name for games played

    Returns:
        DPOYEvaluation with prediction metrics
    """
    # Get actual DPOY
    dpoy_info = get_dpoy_winner(season)
    min_games = get_min_games_for_awards(season)

    actual_dpoy_name = dpoy_info[0] if dpoy_info else None
    actual_dpoy_id = dpoy_info[1] if dpoy_info else None

    # Sort model by score
    model_df = model_df.copy()
    model_df = model_df.sort_values(score_col, ascending=False).reset_index(drop=True)

    # Find actual DPOY's rank in model
    actual_dpoy_rank = None
    if actual_dpoy_name:
        actual_dpoy_rank = _get_player_rank(
            model_df, actual_dpoy_name, score_col, name_col
        )

    # Find model's predicted DPOY (first eligible player by score)
    predicted_dpoy_name = ""
    predicted_dpoy_rank = 0
    eligible_checked = 0

    if games_col in model_df.columns:
        for idx, row in model_df.iterrows():
            eligible_checked += 1
            if row[games_col] >= min_games:
                predicted_dpoy_name = row[name_col]
                predicted_dpoy_rank = eligible_checked  # 1-indexed rank
                break
    else:
        # No games column, just use top player
        if len(model_df) > 0:
            predicted_dpoy_name = model_df.iloc[0][name_col]
            predicted_dpoy_rank = 1
            eligible_checked = 1

    # Determine if hit
    is_hit = actual_dpoy_name is not None and predicted_dpoy_name == actual_dpoy_name

    return DPOYEvaluation(
        season=season,
        actual_dpoy_name=actual_dpoy_name,
        actual_dpoy_id=actual_dpoy_id,
        actual_dpoy_rank=actual_dpoy_rank,
        predicted_dpoy_name=predicted_dpoy_name,
        predicted_dpoy_rank=predicted_dpoy_rank,
        is_hit=is_hit,
        eligible_players_checked=eligible_checked,
        min_games_required=min_games,
    )


def benchmark_against_raptor(
    model_df: pd.DataFrame,
    season: str,
    raptor_df: pd.DataFrame,
    score_col: str = "EDI_Total",
    name_col: str = "PLAYER_NAME",
    raptor_score_col: str = "RAPTOR_DEFENSE",
    raptor_name_col: str = "player_name",
) -> BenchmarkResult | None:
    """Compare EDI vs D-RAPTOR on the same evaluation tasks.

    This is a HEAD-TO-HEAD comparison using:
    1. Average Rank of All-Defense players (lower is better)
    2. DPOY prediction rank (lower is better)

    Winner is determined by who achieves lower average rank.

    Args:
        model_df: DataFrame with EDI predictions
        season: Season string
        raptor_df: DataFrame with D-RAPTOR scores
        score_col: EDI score column
        name_col: Player name column in model_df
        raptor_score_col: D-RAPTOR score column
        raptor_name_col: Player name column in raptor_df

    Returns:
        BenchmarkResult or None if RAPTOR data unavailable
    """
    if raptor_df is None or raptor_df.empty:
        return None

    # Get ground truth
    ground_truth = get_all_defensive_teams(season)
    if ground_truth.empty:
        return None

    gt_names = set(ground_truth["PLAYER_NAME"])

    # Get DPOY info
    dpoy_info = get_dpoy_winner(season)
    dpoy_name = dpoy_info[0] if dpoy_info else None

    # === EDI metrics ===
    model_df_sorted = model_df.sort_values(score_col, ascending=False).reset_index(
        drop=True
    )

    # Calculate EDI avg_rank using existing function
    edi_tier = calculate_tier_alignment(model_df_sorted, gt_names, score_col, name_col)
    edi_avg_rank = edi_tier.avg_rank

    edi_dpoy_rank = None
    if dpoy_name:
        edi_dpoy_rank = _get_player_rank(
            model_df_sorted, dpoy_name, score_col, name_col
        )

    # === RAPTOR metrics ===
    raptor_df_sorted = raptor_df.sort_values(
        raptor_score_col, ascending=False
    ).reset_index(drop=True)

    # Calculate RAPTOR avg_rank using existing function
    raptor_tier = calculate_tier_alignment(
        raptor_df_sorted, gt_names, raptor_score_col, raptor_name_col
    )
    competitor_avg_rank = raptor_tier.avg_rank

    raptor_dpoy_rank = None
    if dpoy_name:
        raptor_dpoy_rank = _get_player_rank(
            raptor_df_sorted, dpoy_name, raptor_score_col, raptor_name_col
        )

    # === Determine winners ===
    # AvgRank winner: lower avg_rank wins (better at ranking All-Defense players)
    avg_rank_diff = edi_avg_rank - competitor_avg_rank

    if avg_rank_diff < -1.0:  # EDI is lower by 1+ rank = EDI better
        avg_rank_winner = "EDI"
    elif avg_rank_diff > 1.0:  # RAPTOR is lower by 1+ rank = RAPTOR better
        avg_rank_winner = "D-RAPTOR"
    else:
        avg_rank_winner = "TIE"

    # DPOY winner: lower rank wins (closer to #1)
    if edi_dpoy_rank is None and raptor_dpoy_rank is None:
        dpoy_winner = "TIE"
    elif edi_dpoy_rank is None:
        dpoy_winner = "D-RAPTOR"
    elif raptor_dpoy_rank is None:
        dpoy_winner = "EDI"
    elif edi_dpoy_rank < raptor_dpoy_rank:
        dpoy_winner = "EDI"
    elif raptor_dpoy_rank < edi_dpoy_rank:
        dpoy_winner = "D-RAPTOR"
    else:
        dpoy_winner = "TIE"

    # Overall winner: needs to win both or win one + tie one
    wins = {"EDI": 0, "D-RAPTOR": 0}
    if avg_rank_winner == "EDI":
        wins["EDI"] += 1
    elif avg_rank_winner == "D-RAPTOR":
        wins["D-RAPTOR"] += 1

    if dpoy_winner == "EDI":
        wins["EDI"] += 1
    elif dpoy_winner == "D-RAPTOR":
        wins["D-RAPTOR"] += 1

    if wins["EDI"] > wins["D-RAPTOR"]:
        overall_winner = "EDI"
    elif wins["D-RAPTOR"] > wins["EDI"]:
        overall_winner = "D-RAPTOR"
    else:
        overall_winner = "TIE"

    return BenchmarkResult(
        season=season,
        competitor_name="D-RAPTOR",
        edi_avg_rank=edi_avg_rank,
        competitor_avg_rank=competitor_avg_rank,
        edi_dpoy_rank=edi_dpoy_rank,
        competitor_dpoy_rank=raptor_dpoy_rank,
        avg_rank_winner=avg_rank_winner,
        dpoy_winner=dpoy_winner,
        overall_winner=overall_winner,
    )


def generate_dpoy_report(dpoy_eval: DPOYEvaluation) -> str:
    """Generate report section for DPOY evaluation."""
    lines = []
    lines.append("🏆 DPOY ALIGNMENT")
    lines.append("-" * 40)

    if dpoy_eval.actual_dpoy_name is None:
        lines.append(f"  No DPOY data available for {dpoy_eval.season}")
        return "\n".join(lines)

    lines.append(f"  Actual DPOY: {dpoy_eval.actual_dpoy_name}")
    lines.append(f"  DPOY Rank in Model: #{dpoy_eval.actual_dpoy_rank or 'N/A'}")
    lines.append(
        f"  Model Prediction: {dpoy_eval.predicted_dpoy_name} (#{dpoy_eval.predicted_dpoy_rank})"
    )
    lines.append(f"  Hit: {'✓ YES' if dpoy_eval.is_hit else '✗ NO'}")
    lines.append(f"  Grade: {dpoy_eval.grade}")
    lines.append(f"  Min Games Required: {dpoy_eval.min_games_required}")

    if dpoy_eval.eligible_players_checked > 1:
        lines.append(
            f"  Note: Top {dpoy_eval.eligible_players_checked - 1} player(s) "
            "were ineligible (games threshold)"
        )

    lines.append("")
    return "\n".join(lines)


def generate_benchmark_report(benchmarks: list[BenchmarkResult]) -> str:
    """Generate report for EDI vs competitor benchmark.

    Compares models on:
    1. AvgRank - Average rank of All-Defense players (lower is better)
    2. DPOY Rank - Where actual DPOY ranks in each model (lower is better)
    """
    if not benchmarks:
        return ""

    lines = []
    lines.append("⚔️ EDI vs D-RAPTOR BENCHMARK")
    lines.append("=" * 60)
    lines.append("")
    lines.append(
        "Note: AvgRank = Average rank of All-Defense players (lower is better)"
    )
    lines.append("")

    # Summary table
    lines.append(
        f"{'Season':<12} {'Metric':<15} {'EDI':<12} {'D-RAPTOR':<12} {'Winner':<10}"
    )
    lines.append("-" * 60)

    for b in benchmarks:
        # AvgRank comparison (lower is better)
        lines.append(
            f"{b.season:<12} {'AvgRank':<15} {b.edi_avg_rank:<12.1f} "
            f"{b.competitor_avg_rank:<12.1f} {b.avg_rank_winner:<10}"
        )

        # DPOY rank comparison (lower is better)
        edi_dpoy = f"#{b.edi_dpoy_rank}" if b.edi_dpoy_rank else "N/A"
        raptor_dpoy = f"#{b.competitor_dpoy_rank}" if b.competitor_dpoy_rank else "N/A"
        lines.append(
            f"{'':12} {'DPOY Rank':<15} {edi_dpoy:<12} {raptor_dpoy:<12} {b.dpoy_winner:<10}"
        )

        # Overall winner
        lines.append(f"{'':12} {'OVERALL':<15} {'':12} {'':12} {b.overall_winner:<10}")
        lines.append("-" * 60)

    # Aggregate winner
    edi_wins = sum(1 for b in benchmarks if b.overall_winner == "EDI")
    raptor_wins = sum(1 for b in benchmarks if b.overall_winner == "D-RAPTOR")
    ties = sum(1 for b in benchmarks if b.overall_winner == "TIE")

    lines.append("")
    lines.append(f"Overall: EDI {edi_wins} - D-RAPTOR {raptor_wins} (Ties: {ties})")

    if edi_wins > raptor_wins:
        lines.append("🏅 EDI wins the benchmark!")
    elif raptor_wins > edi_wins:
        lines.append("🏅 D-RAPTOR wins the benchmark!")
    else:
        lines.append("🤝 It's a draw!")

    lines.append("")
    return "\n".join(lines)
