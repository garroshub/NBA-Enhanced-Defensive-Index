"""
EDI Model Parameter Optimization Script
========================================
Explores parameter combinations to optimize DPOY ranking and All-Defense recall.

Search Space:
- SYNERGY_FACTOR: [0.5, 2.0]
- ROAMER_WEIGHT_REDIST_OUTPUT: [0.3, 0.7]
- ROAMER_THRESHOLD: [0.15, 0.35] (percentile for Roamer classification)
- SYNERGY_D1_THRESHOLD: [0.70, 0.80]
- SYNERGY_D2_THRESHOLD: [0.65, 0.75]

Objective:
- Minimize DPOY average positional rank across 5 seasons
- Maintain All-Defense Recall@30 >= 35/50

Usage:
    python src/tune_parameters.py
    python src/tune_parameters.py --quick  # Fast mode with fewer combinations
"""

import io
import sys
import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Set UTF-8 encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Import from constants and data_fetcher
sys.path.insert(0, str(BASE_DIR / "src"))
from constants import (
    EVALUATION_SEASONS,
    DPOY_INFO,
    classify_role_3cat,
)
from data_fetcher import get_all_defensive_teams, get_dpoy_winner


@dataclass
class ParameterSet:
    """A set of parameters to evaluate."""

    synergy_factor: float
    roamer_redist_output: float
    roamer_threshold: float  # Percentile threshold for Roamer classification
    synergy_d1_threshold: float
    synergy_d2_threshold: float


@dataclass
class EvaluationResult:
    """Results from evaluating a parameter set."""

    params: ParameterSet
    dpoy_avg_rank: float
    recall_at_30: int
    jjj_roamer_rank: int  # JJJ's rank within Roamer category (2022-23)
    detail_ranks: dict  # Per-season DPOY ranks


def load_season_data(season: str) -> pd.DataFrame | None:
    """Load pre-computed EDI data for a season."""
    path = DATA_DIR / f"nba_defensive_all_players_{season}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def recalculate_edi_with_params(df: pd.DataFrame, params: ParameterSet) -> pd.DataFrame:
    """
    Recalculate EDI scores with new parameters.

    This is a simplified recalculation that adjusts:
    1. Roamer classification threshold
    2. Weight redistribution
    3. Synergy bonus

    Note: This assumes the base scores (D1-D5, W1-W5_Base) are already computed.
    """
    df = df.copy()

    # Ensure required columns exist
    required_cols = [
        "D1_Score",
        "D2_Score",
        "D3_Score",
        "D4_Score",
        "D5_Score",
        "W1",
        "W2",
        "W3",
        "W4",
        "W5",
        "Roamer_Pct",
        "PLAYER_POSITION",
    ]

    # Fill missing columns with defaults
    for col in required_cols:
        if col not in df.columns:
            if "Score" in col:
                df[col] = 0.5
            elif col == "Roamer_Pct":
                df[col] = 0.0
            elif col == "PLAYER_POSITION":
                df[col] = "F"
            else:
                df[col] = 0.5

    # Step 1: Reclassify roles with new Roamer threshold
    df["ROLE_NEW"] = df.apply(
        lambda r: classify_role_3cat(
            r["PLAYER_POSITION"], r["Roamer_Pct"], params.roamer_threshold
        ),
        axis=1,
    )

    # Step 2: Recalculate weight redistribution
    # Get original W5 before Roamer adjustment (approximate from current W5)
    # W5_original ≈ W5 / (1 - 0.3 * Roamer_Pct) for Frontcourt
    df["W5_Original"] = df["W5"] / (1 - 0.3 * df["Roamer_Pct"] + 1e-6)
    df["W5_Original"] = df["W5_Original"].clip(0, 1)

    # For Roamers, apply weight loss and redistribution
    roamer_mask = df["ROLE_NEW"] == "Roamer"

    # Calculate W5 loss for Roamers
    df["W5_Loss"] = 0.0
    df.loc[roamer_mask, "W5_Loss"] = (
        df.loc[roamer_mask, "W5_Original"] * 0.3 * df.loc[roamer_mask, "Roamer_Pct"]
    )

    # Redistribute to W1/W2 (output) and W3 (hustle)
    w1_w2_total = df["W1"] + df["W2"] + 1e-6

    df["W1_Adj"] = df["W1"] + df["W5_Loss"] * params.roamer_redist_output * (
        df["W1"] / w1_w2_total
    )
    df["W2_Adj"] = df["W2"] + df["W5_Loss"] * params.roamer_redist_output * (
        df["W2"] / w1_w2_total
    )
    df["W3_Adj"] = df["W3"] + df["W5_Loss"] * (1 - params.roamer_redist_output)
    df["W5_Adj"] = df["W5"] - df["W5_Loss"]

    # Step 3: Calculate Synergy Bonus (sqrt formula)
    def calc_synergy(row):
        d1, d2 = row["D1_Score"], row["D2_Score"]
        if d1 >= params.synergy_d1_threshold and d2 >= params.synergy_d2_threshold:
            raw = (d1 - params.synergy_d1_threshold) * (
                d2 - params.synergy_d2_threshold
            )
            return np.sqrt(raw) * params.synergy_factor * 100
        return 0.0

    df["Synergy_Bonus_New"] = df.apply(calc_synergy, axis=1)

    # Step 4: Recalculate EDI_Total
    # Simplified formula (without full efficiency model recalculation)
    total_weight = df["W1_Adj"] + df["W2_Adj"] + df["W3_Adj"] + df["W4"] + df["W5_Adj"]

    weighted_score = (
        df["D1_Score"] * df["W1_Adj"]
        + df["D2_Score"] * df["W2_Adj"]
        + df["D3_Score"] * df["W3_Adj"]
        + df["D4_Score"] * df["W4"]
        + df["D5_Score"] * df["W5_Adj"]
    )

    df["EDI_Recalc"] = np.where(
        total_weight > 0,
        weighted_score / total_weight * 100 + df["Synergy_Bonus_New"],
        50.0,
    )

    return df


def get_positional_rank(
    df: pd.DataFrame, player_name: str, role: str
) -> tuple[int | None, int]:
    """
    Get player's rank within their role category.

    Returns:
        (rank, pool_size) or (None, 0) if not found
    """
    # Filter to role
    role_df = df[df["ROLE_NEW"] == role].copy()
    if len(role_df) == 0:
        return None, 0

    role_df = role_df.sort_values("EDI_Recalc", ascending=False).reset_index(drop=True)

    # Find player
    match = role_df[
        role_df["PLAYER_NAME"].str.contains(player_name, case=False, na=False)
    ]
    if len(match) == 0:
        # Try partial match
        for idx, row in role_df.iterrows():
            if player_name.lower() in row["PLAYER_NAME"].lower():
                return idx + 1, len(role_df)
        return None, len(role_df)

    return match.index[0] + 1, len(role_df)


def evaluate_params(params: ParameterSet) -> EvaluationResult:
    """Evaluate a parameter set across all seasons."""
    dpoy_ranks = {}
    total_recall = 0
    jjj_rank = 999

    for season in EVALUATION_SEASONS:
        df = load_season_data(season)
        if df is None:
            continue

        # Recalculate with new params
        df = recalculate_edi_with_params(df, params)

        # Get DPOY info
        dpoy_name, expected_role = DPOY_INFO.get(season, (None, None))
        if dpoy_name is None:
            continue

        # Get DPOY's positional rank
        rank, pool_size = get_positional_rank(df, dpoy_name, expected_role)
        if rank is None:
            rank = pool_size + 1  # Not found = last

        dpoy_ranks[season] = rank

        # Special case: track JJJ's Roamer rank for 2022-23
        if season == "2022-23":
            jjj_r, _ = get_positional_rank(df, "Jaren Jackson", "Roamer")
            if jjj_r is not None:
                jjj_rank = jjj_r

        # Calculate Recall@30
        ground_truth = get_all_defensive_teams(season)
        if not ground_truth.empty:
            all_defense_names = set(ground_truth["PLAYER_NAME"])
            top_30 = set(df.nlargest(30, "EDI_Recalc")["PLAYER_NAME"])
            recall = len(all_defense_names & top_30)
            total_recall += recall

    # Calculate average DPOY rank
    avg_rank = np.mean(list(dpoy_ranks.values())) if dpoy_ranks else 999.0

    return EvaluationResult(
        params=params,
        dpoy_avg_rank=avg_rank,
        recall_at_30=total_recall,
        jjj_roamer_rank=jjj_rank,
        detail_ranks=dpoy_ranks,
    )


def generate_search_space(quick_mode: bool = False) -> list[ParameterSet]:
    """Generate parameter combinations to explore."""
    if quick_mode:
        # Reduced search space for quick testing
        synergy_factors = [0.5, 1.0, 1.5]
        roamer_redists = [0.4, 0.5, 0.6]
        roamer_thresholds = [0.20, 0.25, 0.30]
        d1_thresholds = [0.75]
        d2_thresholds = [0.70]
    else:
        # Full search space
        synergy_factors = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        roamer_redists = [0.3, 0.4, 0.5, 0.6, 0.7]
        roamer_thresholds = [0.15, 0.20, 0.25, 0.30, 0.35]
        d1_thresholds = [0.70, 0.75, 0.80]
        d2_thresholds = [0.65, 0.70, 0.75]

    params_list = []
    for sf, rr, rt, d1t, d2t in itertools.product(
        synergy_factors, roamer_redists, roamer_thresholds, d1_thresholds, d2_thresholds
    ):
        params_list.append(
            ParameterSet(
                synergy_factor=sf,
                roamer_redist_output=rr,
                roamer_threshold=rt,
                synergy_d1_threshold=d1t,
                synergy_d2_threshold=d2t,
            )
        )

    return params_list


def run_optimization(quick_mode: bool = False):
    """Run parameter optimization."""
    print("=" * 70)
    print("EDI Parameter Optimization")
    print("=" * 70)

    # Generate search space
    params_list = generate_search_space(quick_mode)
    print(f"Search space: {len(params_list)} parameter combinations")
    print()

    # Evaluate all combinations
    results = []
    best_result = None

    for i, params in enumerate(params_list):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"Evaluating {i + 1}/{len(params_list)}...")

        result = evaluate_params(params)
        results.append(result)

        # Track best result (minimize DPOY rank, maximize recall)
        if best_result is None:
            best_result = result
        else:
            # Primary: lower DPOY rank is better
            # Secondary: higher recall is better
            if result.dpoy_avg_rank < best_result.dpoy_avg_rank or (
                result.dpoy_avg_rank == best_result.dpoy_avg_rank
                and result.recall_at_30 > best_result.recall_at_30
            ):
                best_result = result

    # Sort results
    results.sort(key=lambda r: (r.dpoy_avg_rank, -r.recall_at_30))

    # Print top 10 results
    print()
    print("=" * 70)
    print("TOP 10 PARAMETER COMBINATIONS")
    print("=" * 70)
    print()
    print(
        f"{'Rank':4} | {'DPOY Avg':8} | {'Recall':6} | {'JJJ':4} | {'SF':4} | {'RR':4} | {'RT':4} | {'D1T':4} | {'D2T':4}"
    )
    print("-" * 70)

    for i, r in enumerate(results[:10], 1):
        p = r.params
        print(
            f"{i:4} | {r.dpoy_avg_rank:8.2f} | {r.recall_at_30:6}/50 | #{r.jjj_roamer_rank:<3} | {p.synergy_factor:.2f} | {p.roamer_redist_output:.2f} | {p.roamer_threshold:.2f} | {p.synergy_d1_threshold:.2f} | {p.synergy_d2_threshold:.2f}"
        )

    # Print best result details
    print()
    print("=" * 70)
    print("BEST PARAMETERS")
    print("=" * 70)
    best = results[0]
    print(f"SYNERGY_FACTOR = {best.params.synergy_factor}")
    print(f"ROAMER_WEIGHT_REDIST_OUTPUT = {best.params.roamer_redist_output}")
    print(f"ROAMER_THRESHOLD = {best.params.roamer_threshold}")
    print(f"SYNERGY_D1_THRESHOLD = {best.params.synergy_d1_threshold}")
    print(f"SYNERGY_D2_THRESHOLD = {best.params.synergy_d2_threshold}")
    print()
    print(f"DPOY Average Rank: {best.dpoy_avg_rank:.2f}")
    print(f"All-Defense Recall@30: {best.recall_at_30}/50")
    print(f"JJJ Roamer Rank (2022-23): #{best.jjj_roamer_rank}")
    print()
    print("Per-Season DPOY Ranks:")
    for season, rank in best.detail_ranks.items():
        dpoy_name, role = DPOY_INFO.get(season, ("?", "?"))
        print(f"  {season}: {dpoy_name} ({role}) -> #{rank}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="EDI Parameter Optimization")
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode with fewer combinations"
    )
    args = parser.parse_args()

    run_optimization(quick_mode=args.quick)


if __name__ == "__main__":
    main()
