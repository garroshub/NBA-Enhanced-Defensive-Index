"""Evaluation metrics for defensive analysis model.

This module provides functions to evaluate the EDI model against:
1. All-Defensive Team selections (Layer 1 - Coverage)
2. Advanced defensive statistics (Layer 2 - Correlation)
"""

from typing import Any

import pandas as pd
from scipy import stats


def calculate_coverage(
    df: pd.DataFrame,
    top_n: int = 10,
    score_col: str = "EDI_Total",
    label_col: str = "Is_All_Defense",
) -> dict[str, float]:
    """Calculate coverage metrics for model predictions.

    Args:
        df: DataFrame with model scores and ground truth labels
        top_n: Number of top predictions to evaluate
        score_col: Column name for model scores
        label_col: Column name for ground truth (1=All-Defense, 0=Not)

    Returns:
        Dictionary with precision@k and recall@k metrics.
    """
    # Sort by score descending and take top_n
    sorted_df = df.sort_values(score_col, ascending=False).head(top_n)

    # Count hits in top_n predictions
    hits = sorted_df[label_col].sum()

    # Total actual positives
    total_positives = df[label_col].sum()

    # Precision@k: hits / k
    precision = hits / top_n if top_n > 0 else 0.0

    # Recall@k: hits / total positives
    recall = hits / total_positives if total_positives > 0 else 0.0

    return {
        f"precision@{top_n}": precision,
        f"recall@{top_n}": recall,
        "hits": int(hits),
        "top_n": top_n,
        "total_positives": int(total_positives),
    }


def calculate_correlations(
    df: pd.DataFrame,
    target_col: str,
    ref_cols: list[str],
) -> dict[str, dict[str, Any]]:
    """Calculate correlation between target score and reference metrics.

    Args:
        df: DataFrame with target and reference columns
        target_col: Column name for model score (e.g., "EDI_Total")
        ref_cols: List of reference column names (e.g., ["DEF_RATING", "STL"])

    Returns:
        Dictionary mapping ref_col -> {pearson, spearman, p_value_pearson, p_value_spearman}
    """
    results = {}

    target = df[target_col].dropna()

    for col in ref_cols:
        if col not in df.columns:
            continue

        ref = df[col].dropna()

        # Align indices
        common_idx = target.index.intersection(ref.index)
        if len(common_idx) < 3:
            continue

        t = target.loc[common_idx]
        r = ref.loc[common_idx]

        # Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(t, r)

        # Spearman correlation (rank-based)
        spearman_r, spearman_p = stats.spearmanr(t, r)

        results[col] = {
            "pearson": pearson_r,
            "spearman": spearman_r,
            "p_value_pearson": pearson_p,
            "p_value_spearman": spearman_p,
            "n": len(common_idx),
        }

    return results


def generate_evaluation_report(
    coverage_results: dict[str, Any],
    correlation_results: dict[str, dict[str, Any]],
    season: str,
) -> str:
    """Generate a text report summarizing evaluation results.

    Args:
        coverage_results: Output from calculate_coverage()
        correlation_results: Output from calculate_correlations()
        season: Season string for report header

    Returns:
        Formatted report string.
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"EDI Model Evaluation Report - {season}")
    lines.append("=" * 60)
    lines.append("")

    # Layer 1: Coverage
    lines.append("## Layer 1: All-Defensive Team Coverage")
    lines.append("-" * 40)
    top_n = coverage_results.get("top_n", 10)
    precision = coverage_results.get(f"precision@{top_n}", 0)
    recall = coverage_results.get(f"recall@{top_n}", 0)
    hits = coverage_results.get("hits", 0)
    total = coverage_results.get("total_positives", 0)

    lines.append(f"Top {top_n} Predictions:")
    lines.append(f"  - Precision@{top_n}: {precision:.1%}")
    lines.append(f"  - Recall@{top_n}: {recall:.1%}")
    lines.append(f"  - Hits: {hits}/{total} All-Defensive players found")
    lines.append("")

    # Layer 2: Correlations
    lines.append("## Layer 2: Correlation with Advanced Stats")
    lines.append("-" * 40)

    if correlation_results:
        lines.append(f"{'Metric':<20} {'Pearson':>10} {'Spearman':>10} {'N':>6}")
        lines.append("-" * 48)
        for col, vals in correlation_results.items():
            p = vals.get("pearson", 0)
            s = vals.get("spearman", 0)
            n = vals.get("n", 0)
            lines.append(f"{col:<20} {p:>10.3f} {s:>10.3f} {n:>6}")
    else:
        lines.append("No correlation data available.")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)
