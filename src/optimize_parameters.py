"""Parameter optimization for NBA EDI (Enhanced Defensive Index) model.

This script performs grid search optimization over EDI model hyperparameters
to maximize alignment with All-Defensive Team selections while maintaining
correlation with external metrics (D-RAPTOR, DBPM).

Optimization Framework:
- Training: 2019-20 to 2021-22 (have D-RAPTOR for external validation)
- Validation: 2022-23 to 2023-24 (no D-RAPTOR but have All-Defense)
- Test (Holdout): 2024-25 (final evaluation, never touch during tuning)

Parameters Optimized:
- ROLE_CONFIG weights (D2 interior/exterior, D5 impact)
- BAYES_C (Bayesian shrinkage constant)
- MD_K (Matchup difficulty adjustment coefficient)

Usage:
    python src/optimize_parameters.py           # Run optimization
    python src/optimize_parameters.py --quick   # Quick search (fewer combinations)
    python src/optimize_parameters.py --test    # Evaluate on holdout set
"""

import itertools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data_fetcher import get_all_defensive_teams
from external_metrics import merge_external_metrics, calculate_external_correlation
from evaluation import (
    evaluate_season,
    calculate_tier_alignment,
    calculate_candidate_pool_quality,
    calculate_miss_analysis,
    evaluate_dpoy_alignment,
)


# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Season splits
TRAINING_SEASONS = ["2019-20", "2020-21", "2021-22"]
VALIDATION_SEASONS = ["2022-23", "2023-24"]
TEST_SEASON = "2024-25"

# Current baseline parameters (from nba_defense_mvp.py)
BASELINE_PARAMS = {
    "guard_d2_ext": 0.6,
    "guard_d2_int": 0.4,
    "front_d2_ext": 0.4,
    "front_d2_int": 0.6,
    "guard_d5_impact": 0.5,
    "front_d5_impact": 1.0,
    "bayes_c": 50,
    "md_k": 0.015,
}

# Search space with basketball-logical constraints
# Guards should defend perimeter MORE (ext >= int)
# Frontcourt should protect rim MORE (int >= ext)
SEARCH_SPACE = {
    "guard_d2_ext": [0.55, 0.60, 0.65, 0.70],
    "front_d2_int": [0.50, 0.55, 0.60, 0.65],
    "guard_d5_impact": [0.4, 0.5, 0.6],
    "bayes_c": [35, 45, 55, 65],
    "md_k": [0.010, 0.015, 0.020],
}

QUICK_SEARCH_SPACE = {
    "guard_d2_ext": [0.55, 0.65],
    "front_d2_int": [0.55, 0.65],
    "guard_d5_impact": [0.4, 0.6],
    "bayes_c": [40, 60],
    "md_k": [0.012, 0.018],
}


@dataclass
class ParameterSet:
    """Container for a parameter configuration."""

    guard_d2_ext: float
    guard_d2_int: float  # Derived: 1 - guard_d2_ext
    front_d2_ext: float  # Derived: 1 - front_d2_int
    front_d2_int: float
    guard_d5_impact: float
    front_d5_impact: float  # Always 1.0 (frontcourt full rebound credit)
    bayes_c: int
    md_k: float

    @classmethod
    def from_search(
        cls,
        guard_d2_ext: float,
        front_d2_int: float,
        guard_d5_impact: float,
        bayes_c: int,
        md_k: float,
    ) -> "ParameterSet":
        """Create from search space (derives complementary weights)."""
        return cls(
            guard_d2_ext=guard_d2_ext,
            guard_d2_int=1.0 - guard_d2_ext,
            front_d2_ext=1.0 - front_d2_int,
            front_d2_int=front_d2_int,
            guard_d5_impact=guard_d5_impact,
            front_d5_impact=1.0,
            bayes_c=bayes_c,
            md_k=md_k,
        )


@dataclass
class EvaluationResult:
    """Results from evaluating a parameter set on a season."""

    season: str
    avg_rank: float
    recall_at_10: float
    recall_at_20: float
    blind_spots: int
    d_raptor_corr: float | None
    dbpm_corr: float | None
    dpoy_rank: int | None  # Where actual DPOY ranks in model
    dpoy_hit: bool  # Model's top eligible player is DPOY?


@dataclass
class OptimizationResult:
    """Complete optimization result for a parameter set."""

    params: ParameterSet
    training_score: float
    validation_score: float
    training_results: list[EvaluationResult]
    validation_results: list[EvaluationResult]

    @property
    def combined_score(self) -> float:
        """Weighted combination of training and validation scores."""
        return 0.6 * self.training_score + 0.4 * self.validation_score


# =============================================================================
# Core Functions
# =============================================================================


def bayesian_score(raw_pct: float, n: float, c: int = 50) -> tuple[float, float]:
    """Apply Bayesian shrinkage to raw percentile.

    Same logic as nba_defense_mvp.py to ensure consistency.
    """
    prior = 0.5
    weight = n / (n + c)
    shrunk = (weight * raw_pct) + ((1 - weight) * prior)
    return shrunk, weight


def recalculate_edi_scores(
    df: pd.DataFrame,
    params: ParameterSet,
) -> pd.DataFrame:
    """Recalculate EDI scores with new parameters.

    This is the fast in-memory recalculation that doesn't require API calls.
    Uses pre-computed raw metrics from the CSV files.

    Args:
        df: DataFrame with raw EDI component data
        params: Parameter set to use for calculation

    Returns:
        DataFrame with recalculated EDI_Total
    """
    df = df.copy()

    # Classify roles (same logic as main script)
    def classify_role(position: str) -> str:
        if pd.isna(position) or position == "":
            return "Frontcourt"
        pos = str(position).upper()
        if "F" in pos or "C" in pos:
            return "Frontcourt"
        if "G" in pos:
            return "Guards"
        return "Frontcourt"

    if "ROLE" not in df.columns:
        df["ROLE"] = df["PLAYER_POSITION"].apply(classify_role)

    # Recalculate D1 with new MD_K
    if "PCT_PLUSMINUS" in df.columns and "MD_Zscore" in df.columns:
        df["PCT_PLUSMINUS_ADJ"] = df["PCT_PLUSMINUS"] - (params.md_k * df["MD_Zscore"])
        df["D1_Raw"] = 1 - df["PCT_PLUSMINUS_ADJ"].rank(pct=True)
        df["D1_N"] = df["D_FGA"].fillna(0) * df["GP"]

        d1_result = df.apply(
            lambda r: bayesian_score(
                r["D1_Raw"] if pd.notna(r["D1_Raw"]) else 0.5,
                r["D1_N"],
                params.bayes_c,
            ),
            axis=1,
        )
        df["D1_Score"] = d1_result.apply(lambda x: x[0])
        df["W1"] = d1_result.apply(lambda x: x[1])

    # Recalculate D2 with new role weights
    if "Rim_PLUSMINUS" in df.columns and "3PT_PLUSMINUS" in df.columns:
        # Recalculate adjusted values
        df["Rim_PLUSMINUS_ADJ"] = df["Rim_PLUSMINUS"] - (params.md_k * df["MD_Zscore"])
        df["3PT_PLUSMINUS_ADJ"] = df["3PT_PLUSMINUS"] - (params.md_k * df["MD_Zscore"])

        df["Rim_Raw"] = 1 - df["Rim_PLUSMINUS_ADJ"].rank(pct=True)
        df["3PT_Raw"] = 1 - df["3PT_PLUSMINUS_ADJ"].rank(pct=True)

        rim_raw = df["Rim_Raw"].fillna(0.5)
        pt3_raw = df["3PT_Raw"].fillna(0.5)

        def calc_d2_raw(row):
            if row["ROLE"] == "Guards":
                return (rim_raw[row.name] * params.guard_d2_int) + (
                    pt3_raw[row.name] * params.guard_d2_ext
                )
            else:
                return (rim_raw[row.name] * params.front_d2_int) + (
                    pt3_raw[row.name] * params.front_d2_ext
                )

        df["D2_Raw"] = df.apply(calc_d2_raw, axis=1)

        # Sample size calculation
        rim_fga = df["Rim_FGA"].fillna(0) * df["GP"]
        fg3_fga = df["FG3_FGA"].fillna(0) * df["GP"]

        def calc_d2_n(row):
            if row["ROLE"] == "Guards":
                return (rim_fga[row.name] * params.guard_d2_int) + (
                    fg3_fga[row.name] * params.guard_d2_ext
                )
            else:
                return (rim_fga[row.name] * params.front_d2_int) + (
                    fg3_fga[row.name] * params.front_d2_ext
                )

        df["D2_N"] = df.apply(calc_d2_n, axis=1)

        d2_result = df.apply(
            lambda r: bayesian_score(r["D2_Raw"], r["D2_N"], params.bayes_c), axis=1
        )
        df["D2_Score"] = d2_result.apply(lambda x: x[0])
        df["W2"] = d2_result.apply(lambda x: x[1])

    # Recalculate D3 (Hustle) - only Bayes_C changes
    if "D3_Raw" in df.columns:
        d3_result = df.apply(
            lambda r: bayesian_score(
                r["D3_Raw"] if pd.notna(r["D3_Raw"]) else 0.5,
                r.get("D3_N", 0),
                params.bayes_c,
            ),
            axis=1,
        )
        df["D3_Score"] = d3_result.apply(lambda x: x[0])
        df["W3"] = d3_result.apply(lambda x: x[1])

    # D4 (Stocks) - only Bayes_C changes
    if "D4_Raw" in df.columns:
        d4_result = df.apply(
            lambda r: bayesian_score(
                r["D4_Raw"] if pd.notna(r["D4_Raw"]) else 0.5,
                r.get("D4_N", 0),
                params.bayes_c,
            ),
            axis=1,
        )
        df["D4_Score"] = d4_result.apply(lambda x: x[0])
        df["W4"] = d4_result.apply(lambda x: x[1])

    # Recalculate D5 with new role impact
    if "D5_Raw" in df.columns:
        d5_result = df.apply(
            lambda r: bayesian_score(
                r["D5_Raw"] if pd.notna(r["D5_Raw"]) else 0.5,
                r.get("D5_N", 0),
                params.bayes_c,
            ),
            axis=1,
        )
        df["D5_Score"] = d5_result.apply(lambda x: x[0])
        df["W5_Base"] = d5_result.apply(lambda x: x[1])

        # Apply role-based impact
        df["W5"] = df.apply(
            lambda r: r["W5_Base"] * params.guard_d5_impact
            if r["ROLE"] == "Guards"
            else r["W5_Base"] * params.front_d5_impact,
            axis=1,
        )

    # Fill NaN scores with 0.5
    score_cols = ["D1_Score", "D2_Score", "D3_Score", "D4_Score", "D5_Score"]
    weight_cols = ["W1", "W2", "W3", "W4", "W5"]

    for col in score_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.5)
    for col in weight_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # Calculate EDI_Total (weighted average of dimensions)
    df["Actual_Output"] = sum(
        df[s] * df[w] for s, w in zip(score_cols, weight_cols) if s in df.columns
    )
    df["Input_Score"] = sum(df[w] for w in weight_cols if w in df.columns)

    # Avoid division by zero
    df["Expected_Output"] = df["Input_Score"] * 0.5
    df["Efficiency"] = df["Actual_Output"] / df["Expected_Output"].replace(0, 1)

    # Final EDI score (efficiency-weighted)
    df["EDI_Total"] = df["Actual_Output"] * df["Efficiency"].clip(lower=0.5, upper=1.5)

    return df


def evaluate_params_on_season(
    df: pd.DataFrame,
    params: ParameterSet,
    season: str,
    include_external: bool = False,
) -> EvaluationResult:
    """Evaluate a parameter set on a single season.

    Args:
        df: Raw data DataFrame
        params: Parameter set to evaluate
        season: Season string
        include_external: Whether to calculate external correlations

    Returns:
        EvaluationResult with metrics
    """
    # Recalculate with new parameters
    recalc_df = recalculate_edi_scores(df, params)

    # Get ground truth
    gt = get_all_defensive_teams(season)
    if gt.empty:
        return EvaluationResult(
            season=season,
            avg_rank=999,
            recall_at_10=0,
            recall_at_20=0,
            blind_spots=10,
            d_raptor_corr=None,
            dbpm_corr=None,
            dpoy_rank=None,
            dpoy_hit=False,
        )

    gt_names = set(gt["PLAYER_NAME"])

    # Calculate metrics
    tier = calculate_tier_alignment(recalc_df, gt_names)
    pool = calculate_candidate_pool_quality(recalc_df, gt_names)
    miss = calculate_miss_analysis(recalc_df, gt_names)

    # DPOY alignment
    dpoy_eval = evaluate_dpoy_alignment(recalc_df, season)
    dpoy_rank = dpoy_eval.actual_dpoy_rank
    dpoy_hit = dpoy_eval.is_hit

    # External correlations (optional, slower)
    d_raptor_corr = None
    dbpm_corr = None
    if include_external:
        merged = merge_external_metrics(recalc_df, season)
        ext_corr = calculate_external_correlation(merged)
        if "D_RAPTOR" in ext_corr and "spearman" in ext_corr["D_RAPTOR"]:
            d_raptor_corr = ext_corr["D_RAPTOR"]["spearman"]
        if "DBPM" in ext_corr and "spearman" in ext_corr["DBPM"]:
            dbpm_corr = ext_corr["DBPM"]["spearman"]

    return EvaluationResult(
        season=season,
        avg_rank=tier.avg_rank,
        recall_at_10=pool.recall_at_10,
        recall_at_20=pool.recall_at_20,
        blind_spots=len(miss.blind_spots),
        d_raptor_corr=d_raptor_corr,
        dbpm_corr=dbpm_corr,
        dpoy_rank=dpoy_rank,
        dpoy_hit=dpoy_hit,
    )


def calculate_objective_score(
    results: list[EvaluationResult],
    include_external: bool = False,  # Deprecated: kept for API compatibility
) -> float:
    """Calculate optimization objective from evaluation results.

    Objective: MINIMIZE total ranking error

    Loss = AvgRank(All-Defense) + Σ(DPOY_Rank - 1)

    We return negative loss so higher = better (for sorting).

    Where:
    - AvgRank: Average rank of All-Defense players in EDI (lower is better)
    - DPOY_Rank - 1: Gap between DPOY's rank and #1 (0 if DPOY is #1)

    Args:
        results: List of EvaluationResult from different seasons
        include_external: DEPRECATED - ignored

    Returns:
        Negative loss (higher is better, i.e., lower total ranking error)
    """
    if not results:
        return -999.0

    # Aggregate All-Defense average ranks
    avg_ranks = [r.avg_rank for r in results if r.avg_rank < 900]
    if not avg_ranks:
        return -999.0

    # Component 1: Mean of All-Defense average ranks across seasons
    all_defense_loss = float(np.mean(avg_ranks))

    # Component 2: DPOY ranking gap (rank - 1, so #1 = 0 gap)
    dpoy_gaps = []
    for r in results:
        if r.dpoy_rank is not None:
            dpoy_gaps.append(r.dpoy_rank - 1)

    dpoy_loss = float(np.mean(dpoy_gaps)) if dpoy_gaps else 0.0

    # Total loss (weighted sum)
    # Weight DPOY less since it's single player vs 10 All-Defense
    alpha = 1.0  # All-Defense weight
    beta = 0.5  # DPOY weight (single player, so lower weight)

    total_loss = (alpha * all_defense_loss) + (beta * dpoy_loss)

    # Return negative loss (so higher = better for sorting)
    return -total_loss


def load_season_data(season: str) -> pd.DataFrame:
    """Load season data from CSV."""
    csv_path = DATA_DIR / f"nba_defensive_all_players_{season}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    return pd.read_csv(csv_path)


def run_optimization(
    quick: bool = False,
    verbose: bool = True,
) -> list[OptimizationResult]:
    """Run parameter optimization.

    Args:
        quick: Use reduced search space for faster iteration
        verbose: Print progress

    Returns:
        List of OptimizationResult sorted by combined score (best first)
    """
    search_space = QUICK_SEARCH_SPACE if quick else SEARCH_SPACE

    # Generate all parameter combinations
    param_keys = list(search_space.keys())
    param_values = [search_space[k] for k in param_keys]
    combinations = list(itertools.product(*param_values))

    total = len(combinations)
    if verbose:
        print(f"Running optimization over {total} parameter combinations...")
        print(f"Training seasons: {TRAINING_SEASONS}")
        print(f"Validation seasons: {VALIDATION_SEASONS}")
        print()

    # Load data once
    training_data = {s: load_season_data(s) for s in TRAINING_SEASONS}
    validation_data = {s: load_season_data(s) for s in VALIDATION_SEASONS}

    results = []

    for i, values in enumerate(combinations):
        params_dict = dict(zip(param_keys, values))
        params = ParameterSet.from_search(**params_dict)

        # Evaluate on training set (with external correlations)
        train_results = []
        for season in TRAINING_SEASONS:
            er = evaluate_params_on_season(
                training_data[season], params, season, include_external=True
            )
            train_results.append(er)

        train_score = calculate_objective_score(train_results, include_external=True)

        # Evaluate on validation set (no external correlations)
        val_results = []
        for season in VALIDATION_SEASONS:
            er = evaluate_params_on_season(
                validation_data[season], params, season, include_external=False
            )
            val_results.append(er)

        val_score = calculate_objective_score(val_results, include_external=False)

        opt_result = OptimizationResult(
            params=params,
            training_score=train_score,
            validation_score=val_score,
            training_results=train_results,
            validation_results=val_results,
        )
        results.append(opt_result)

        if verbose and (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{total} ({100 * (i + 1) / total:.0f}%)")

    # Sort by combined score
    results.sort(key=lambda r: r.combined_score, reverse=True)

    return results


def evaluate_on_test_set(
    params: ParameterSet, verbose: bool = True
) -> EvaluationResult:
    """Evaluate best parameters on holdout test set.

    Only call this ONCE after optimization is complete.

    Args:
        params: Best parameter set from optimization
        verbose: Print results

    Returns:
        EvaluationResult for test season
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"FINAL EVALUATION ON HOLDOUT TEST SET: {TEST_SEASON}")
        print(f"{'=' * 60}\n")

    df = load_season_data(TEST_SEASON)
    result = evaluate_params_on_season(df, params, TEST_SEASON, include_external=True)

    if verbose:
        print(f"Test Season: {result.season}")
        print(f"Average Rank: {result.avg_rank:.1f}")
        print(f"Recall@10: {result.recall_at_10:.0f}%")
        print(f"Recall@20: {result.recall_at_20:.0f}%")
        print(f"Blind Spots: {result.blind_spots}")
        if result.dpoy_rank is not None:
            hit_str = " [HIT]" if result.dpoy_hit else ""
            print(f"DPOY Rank: #{result.dpoy_rank}{hit_str}")
        if result.d_raptor_corr:
            print(f"D-RAPTOR Correlation: {result.d_raptor_corr:.3f}")
        if result.dbpm_corr:
            print(f"DBPM Correlation: {result.dbpm_corr:.3f}")

    return result


def print_optimization_report(results: list[OptimizationResult], top_n: int = 5):
    """Print detailed optimization report."""
    print("\n" + "=" * 70)
    print("PARAMETER OPTIMIZATION RESULTS")
    print("=" * 70)

    print(f"\nTop {top_n} Parameter Sets:")
    print("-" * 70)

    for i, opt in enumerate(results[:top_n], 1):
        p = opt.params
        print(f"\n#{i} Combined Score: {opt.combined_score:.4f}")
        print(
            f"    Training: {opt.training_score:.4f}, Validation: {opt.validation_score:.4f}"
        )
        print(f"    Parameters:")
        print(f"      Guard D2:  ext={p.guard_d2_ext:.2f}, int={p.guard_d2_int:.2f}")
        print(f"      Front D2:  ext={p.front_d2_ext:.2f}, int={p.front_d2_int:.2f}")
        print(f"      Guard D5 Impact: {p.guard_d5_impact:.2f}")
        print(f"      BAYES_C: {p.bayes_c}")
        print(f"      MD_K: {p.md_k:.3f}")

        print(f"    Training Results:")
        for r in opt.training_results:
            dpoy_str = ""
            if r.dpoy_rank is not None:
                hit_mark = " [HIT]" if r.dpoy_hit else ""
                dpoy_str = f", DPOY=#{r.dpoy_rank}{hit_mark}"
            print(
                f"      {r.season}: AvgRank={r.avg_rank:.1f}, "
                f"R@10={r.recall_at_10:.0f}%, R@20={r.recall_at_20:.0f}%{dpoy_str}"
            )

        print(f"    Validation Results:")
        for r in opt.validation_results:
            dpoy_str = ""
            if r.dpoy_rank is not None:
                hit_mark = " [HIT]" if r.dpoy_hit else ""
                dpoy_str = f", DPOY=#{r.dpoy_rank}{hit_mark}"
            print(
                f"      {r.season}: AvgRank={r.avg_rank:.1f}, "
                f"R@10={r.recall_at_10:.0f}%, R@20={r.recall_at_20:.0f}%{dpoy_str}"
            )

    # Baseline comparison
    print("\n" + "-" * 70)
    print("BASELINE COMPARISON (current parameters):")
    print("-" * 70)
    baseline = ParameterSet.from_search(
        guard_d2_ext=BASELINE_PARAMS["guard_d2_ext"],
        front_d2_int=BASELINE_PARAMS["front_d2_int"],
        guard_d5_impact=BASELINE_PARAMS["guard_d5_impact"],
        bayes_c=BASELINE_PARAMS["bayes_c"],
        md_k=BASELINE_PARAMS["md_k"],
    )

    # Find baseline in results or compute
    baseline_result = None
    for opt in results:
        p = opt.params
        if (
            p.guard_d2_ext == baseline.guard_d2_ext
            and p.front_d2_int == baseline.front_d2_int
            and p.guard_d5_impact == baseline.guard_d5_impact
            and p.bayes_c == baseline.bayes_c
            and abs(p.md_k - baseline.md_k) < 0.001
        ):
            baseline_result = opt
            break

    if baseline_result:
        print(f"Baseline Score: {baseline_result.combined_score:.4f}")
        best = results[0]
        improvement = (
            (best.combined_score - baseline_result.combined_score)
            / baseline_result.combined_score
            * 100
        )
        print(f"Best Score: {best.combined_score:.4f}")
        print(f"Improvement: {improvement:+.1f}%")
    else:
        print("(Baseline parameters not in search space)")

    # Best parameters summary
    print("\n" + "=" * 70)
    print("RECOMMENDED PARAMETERS")
    print("=" * 70)
    best = results[0].params
    print(f"""
ROLE_CONFIG = {{
    "Guards": {{
        "D2_EXT_WEIGHT": {best.guard_d2_ext},
        "D2_INT_WEIGHT": {best.guard_d2_int},
        "D5_IMPACT": {best.guard_d5_impact},
    }},
    "Frontcourt": {{
        "D2_EXT_WEIGHT": {best.front_d2_ext},
        "D2_INT_WEIGHT": {best.front_d2_int},
        "D5_IMPACT": {best.front_d5_impact},
    }},
}}

BAYES_C = {best.bayes_c}
MD_K = {best.md_k}
""")


def main():
    """Main entry point."""
    quick_mode = "--quick" in sys.argv
    test_mode = "--test" in sys.argv
    test_optimized = "--test-optimized" in sys.argv

    if test_mode:
        # Evaluate current baseline on test set
        print("Evaluating baseline parameters on holdout test set...")
        baseline = ParameterSet.from_search(
            guard_d2_ext=BASELINE_PARAMS["guard_d2_ext"],
            front_d2_int=BASELINE_PARAMS["front_d2_int"],
            guard_d5_impact=BASELINE_PARAMS["guard_d5_impact"],
            bayes_c=BASELINE_PARAMS["bayes_c"],
            md_k=BASELINE_PARAMS["md_k"],
        )
        evaluate_on_test_set(baseline)
        return

    if test_optimized:
        # Evaluate best optimized parameters on test set
        # Best params from quick optimization (new AvgRank objective):
        # guard_d2_ext=0.65, front_d2_int=0.55, guard_d5_impact=0.40, bayes_c=60, md_k=0.018
        print("Evaluating OPTIMIZED parameters on holdout test set...")
        optimized = ParameterSet.from_search(
            guard_d2_ext=0.65,
            front_d2_int=0.55,
            guard_d5_impact=0.40,
            bayes_c=60,
            md_k=0.018,
        )
        evaluate_on_test_set(optimized)
        return

    # Run optimization
    results = run_optimization(quick=quick_mode)
    print_optimization_report(results)

    # Skip interactive prompt in non-interactive environments
    print("\n" + "-" * 70)
    try:
        response = input("Evaluate best parameters on holdout test set? (y/n): ")
        if response.lower() == "y":
            evaluate_on_test_set(results[0].params)
    except EOFError:
        print("Non-interactive mode. Use --test-optimized to evaluate on holdout.")


if __name__ == "__main__":
    main()
