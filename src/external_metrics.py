"""External advanced metrics fetcher for model validation.

This module fetches external high-level defensive metrics from public sources
to validate the EDI model against independent measurements.

Supported Sources:
1. FiveThirtyEight RAPTOR (D-RAPTOR) - GitHub CSV
2. Basketball Reference D-BPM - Website scraping

Note: These metrics provide independent validation that our EDI model
captures similar signal to professionally-developed metrics.
"""

import io
import re
from functools import lru_cache
from typing import Any

import pandas as pd
import requests


# FiveThirtyEight RAPTOR data URLs
RAPTOR_BASE_URL = (
    "https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-raptor"
)
RAPTOR_MODERN_URL = f"{RAPTOR_BASE_URL}/modern_RAPTOR_by_player.csv"
RAPTOR_HISTORICAL_URL = f"{RAPTOR_BASE_URL}/historical_RAPTOR_by_player.csv"

# Basketball Reference URL template for advanced stats
BREF_ADVANCED_URL = (
    "https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html"
)


@lru_cache(maxsize=1)
def fetch_raptor_data() -> pd.DataFrame:
    """Fetch RAPTOR data from FiveThirtyEight GitHub.

    Returns:
        DataFrame with columns:
        - player_name: Player name
        - season: Season year (e.g., 2023 for 2022-23 season)
        - raptor_defense: Defensive RAPTOR score
        - raptor_offense: Offensive RAPTOR score
        - raptor_total: Total RAPTOR score
        - war_total: Wins Above Replacement
    """
    try:
        response = requests.get(RAPTOR_MODERN_URL, timeout=30)
        response.raise_for_status()

        df = pd.read_csv(io.StringIO(response.text))

        # Select relevant columns
        cols = [
            "player_name",
            "player_id",
            "season",
            "poss",
            "mp",
            "raptor_offense",
            "raptor_defense",
            "raptor_total",
            "war_total",
            "war_reg_season",
        ]
        available_cols = [c for c in cols if c in df.columns]
        df = df[available_cols].copy()

        return df

    except requests.RequestException as e:
        print(f"Error fetching RAPTOR data: {e}")
        return pd.DataFrame()


def get_raptor_for_season(season: str) -> pd.DataFrame:
    """Get RAPTOR data for a specific NBA season.

    Args:
        season: Season string (e.g., "2023-24" or "2022-23")

    Returns:
        DataFrame with RAPTOR metrics for that season
    """
    df = fetch_raptor_data()
    if df.empty:
        return df

    # Convert season string to year
    # "2023-24" -> 2024 (RAPTOR uses end year)
    # "2022-23" -> 2023
    start_year = int(season.split("-")[0])
    raptor_year = start_year + 1

    season_df = df[df["season"] == raptor_year].copy()

    # Normalize player names for matching
    season_df["player_name_normalized"] = season_df["player_name"].apply(normalize_name)

    return season_df


def normalize_name(name: str) -> str:
    """Normalize player name for matching across datasets.

    Handles common variations:
    - Suffixes (Jr., III, etc.)
    - Accented characters
    - Different spacing
    """
    if pd.isna(name):
        return ""

    name = str(name).strip()

    # Remove suffixes
    suffixes = [" Jr.", " Jr", " III", " II", " IV", " Sr.", " Sr"]
    for suffix in suffixes:
        name = name.replace(suffix, "")

    # Normalize accented characters (simplified)
    replacements = {
        "ć": "c",
        "č": "c",
        "ž": "z",
        "š": "s",
        "đ": "d",
        "ö": "o",
        "ü": "u",
        "ä": "a",
        "é": "e",
        "ñ": "n",
        "ģ": "g",
        "ņ": "n",
        "ī": "i",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)

    # Remove extra spaces
    name = " ".join(name.split())

    return name.lower()


def fetch_bref_advanced_stats(season: str) -> pd.DataFrame:
    """Fetch advanced stats from Basketball Reference.

    Includes DBPM (Defensive Box Plus/Minus) which is a good validation metric.

    Args:
        season: Season string (e.g., "2023-24")

    Returns:
        DataFrame with BPM metrics
    """
    # Convert to end year for BBRef URL
    start_year = int(season.split("-")[0])
    end_year = start_year + 1

    url = BREF_ADVANCED_URL.format(year=end_year)

    try:
        # BBRef requires headers to avoid blocking
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        # Parse HTML tables
        tables = pd.read_html(io.StringIO(response.text))

        # Find the advanced stats table (usually first one with DBPM)
        for table in tables:
            if "DBPM" in table.columns or "dbpm" in [c.lower() for c in table.columns]:
                df = table.copy()
                # Clean up multi-level columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [
                        col[-1] if isinstance(col, tuple) else col for col in df.columns
                    ]

                # Normalize column names
                df.columns = [str(c).strip() for c in df.columns]

                # Remove header rows that got included as data
                df = df[df["Player"] != "Player"]

                # Select key columns
                key_cols = [
                    "Player",
                    "Pos",
                    "Age",
                    "Tm",
                    "G",
                    "MP",
                    "BPM",
                    "DBPM",
                    "OBPM",
                    "VORP",
                ]
                available = [c for c in key_cols if c in df.columns]
                df = df[available].copy()

                # Convert numeric columns
                for col in ["G", "MP", "BPM", "DBPM", "OBPM", "VORP"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                df["player_name_normalized"] = df["Player"].apply(normalize_name)

                return df

        print(f"Could not find advanced stats table for {season}")
        return pd.DataFrame()

    except requests.RequestException as e:
        print(f"Error fetching BBRef data: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error parsing BBRef data: {e}")
        return pd.DataFrame()


def merge_external_metrics(
    model_df: pd.DataFrame,
    season: str,
    name_col: str = "PLAYER_NAME",
) -> pd.DataFrame:
    """Merge external metrics into model DataFrame.

    Adds columns:
    - D_RAPTOR: Defensive RAPTOR score
    - RAPTOR_Total: Total RAPTOR score
    - DBPM: Defensive Box Plus/Minus from BBRef

    Args:
        model_df: DataFrame with EDI model results
        season: Season string (e.g., "2023-24")
        name_col: Column name for player names

    Returns:
        DataFrame with external metrics added
    """
    model_df = model_df.copy()
    model_df["_name_normalized"] = model_df[name_col].apply(normalize_name)

    # Fetch RAPTOR data
    raptor_df = get_raptor_for_season(season)
    if not raptor_df.empty:
        raptor_cols = {
            "raptor_defense": "D_RAPTOR",
            "raptor_offense": "O_RAPTOR",
            "raptor_total": "RAPTOR_Total",
            "war_total": "RAPTOR_WAR",
        }

        raptor_merge = raptor_df[
            ["player_name_normalized"] + list(raptor_cols.keys())
        ].copy()
        raptor_merge = raptor_merge.rename(columns=raptor_cols)

        model_df = model_df.merge(
            raptor_merge,
            left_on="_name_normalized",
            right_on="player_name_normalized",
            how="left",
        )
        if "player_name_normalized" in model_df.columns:
            model_df = model_df.drop(columns=["player_name_normalized"])

    # Fetch BBRef BPM data
    bref_df = fetch_bref_advanced_stats(season)
    if not bref_df.empty:
        bref_cols = ["DBPM", "OBPM", "BPM", "VORP"]
        available_cols = [c for c in bref_cols if c in bref_df.columns]

        if available_cols:
            bref_merge = bref_df[["player_name_normalized"] + available_cols].copy()

            # Handle players on multiple teams - take weighted average or TOT row
            bref_merge = (
                bref_merge.groupby("player_name_normalized").first().reset_index()
            )

            model_df = model_df.merge(
                bref_merge,
                left_on="_name_normalized",
                right_on="player_name_normalized",
                how="left",
            )
            if "player_name_normalized" in model_df.columns:
                model_df = model_df.drop(columns=["player_name_normalized"])

    # Clean up
    if "_name_normalized" in model_df.columns:
        model_df = model_df.drop(columns=["_name_normalized"])

    return model_df


def calculate_external_correlation(
    model_df: pd.DataFrame,
    score_col: str = "EDI_Total",
) -> dict[str, Any]:
    """Calculate correlation between EDI and external metrics.

    Args:
        model_df: DataFrame with EDI and external metrics merged
        score_col: Column name for EDI scores

    Returns:
        Dictionary with correlation metrics
    """
    results = {}

    external_cols = ["D_RAPTOR", "DBPM", "RAPTOR_Total", "BPM"]

    for ext_col in external_cols:
        if ext_col not in model_df.columns:
            continue

        # Filter to players with both metrics
        valid_df = model_df[[score_col, ext_col]].dropna()

        if len(valid_df) < 10:
            results[ext_col] = {"error": "Insufficient data for correlation"}
            continue

        # Pearson correlation
        corr = valid_df[score_col].corr(valid_df[ext_col])

        # Spearman rank correlation (more robust)
        spearman = valid_df[score_col].corr(valid_df[ext_col], method="spearman")

        results[ext_col] = {
            "pearson": round(float(corr), 3),
            "spearman": round(float(spearman), 3),
            "n_players": len(valid_df),
            "interpretation": _interpret_correlation(float(corr)),
        }

    return results


def _interpret_correlation(r: float) -> str:
    """Interpret correlation coefficient strength."""
    r_abs = abs(r)
    if r_abs >= 0.7:
        return "Strong"
    elif r_abs >= 0.5:
        return "Moderate"
    elif r_abs >= 0.3:
        return "Weak"
    return "Very Weak/None"


def generate_external_validation_report(
    model_df: pd.DataFrame,
    season: str,
    score_col: str = "EDI_Total",
    name_col: str = "PLAYER_NAME",
) -> str:
    """Generate external validation report for a season.

    Args:
        model_df: DataFrame with EDI model results
        season: Season string (e.g., "2023-24")
        score_col: Column name for EDI scores
        name_col: Column name for player names

    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"EXTERNAL VALIDATION REPORT - {season}")
    lines.append("=" * 60)
    lines.append("")

    # Merge external data
    merged_df = merge_external_metrics(model_df, season, name_col)

    # Calculate correlations
    correlations = calculate_external_correlation(merged_df, score_col)

    lines.append("📈 CORRELATION WITH EXTERNAL METRICS")
    lines.append("-" * 40)
    lines.append("")
    lines.append("These metrics validate that our EDI model captures similar")
    lines.append("signal to professionally-developed defensive metrics.")
    lines.append("")

    if not correlations:
        lines.append("⚠️ No external data available for this season")
    else:
        lines.append(
            f"{'Metric':<15} {'Pearson':<10} {'Spearman':<10} {'N':<6} {'Strength':<12}"
        )
        lines.append("-" * 55)

        for metric, data in correlations.items():
            if "error" in data:
                lines.append(f"{metric:<15} {data['error']}")
            else:
                lines.append(
                    f"{metric:<15} {data['pearson']:<10.3f} {data['spearman']:<10.3f} "
                    f"{data['n_players']:<6} {data['interpretation']:<12}"
                )

    lines.append("")
    lines.append("📖 INTERPRETATION GUIDE")
    lines.append("-" * 40)
    lines.append(
        "• D-RAPTOR: FiveThirtyEight's defensive rating (higher = better defender)"
    )
    lines.append("• DBPM: Basketball-Reference's Defensive Box Plus/Minus")
    lines.append("• Pearson: Linear correlation (-1 to 1)")
    lines.append("• Spearman: Rank correlation (-1 to 1)")
    lines.append("")
    lines.append("Expected correlations for a valid defensive metric:")
    lines.append("• With D-RAPTOR: 0.4-0.7 (moderate to strong)")
    lines.append("• With DBPM: 0.3-0.6 (weak to moderate)")
    lines.append("")

    # Top 10 comparison
    lines.append("🔝 TOP 10 COMPARISON")
    lines.append("-" * 40)

    merged_df_sorted = merged_df.sort_values(score_col, ascending=False).head(10)

    # Check which external columns are available
    ext_cols = ["D_RAPTOR", "DBPM"]
    available_ext = [
        c for c in ext_cols if c in merged_df.columns and merged_df[c].notna().any()
    ]

    if available_ext:
        header = f"{'Rank':<5} {'Player':<25} {'EDI':<8}"
        for col in available_ext:
            header += f" {col:<10}"
        lines.append(header)
        lines.append("-" * (40 + 10 * len(available_ext)))

        for i, (_, row) in enumerate(merged_df_sorted.iterrows(), 1):
            line = f"{i:<5} {row[name_col][:24]:<25} {row[score_col]:<8.2f}"
            for col in available_ext:
                val = row.get(col, None)
                if pd.notna(val):
                    line += f" {val:<10.2f}"
                else:
                    line += f" {'N/A':<10}"
            lines.append(line)
    else:
        lines.append("No external metrics available for comparison")

    return "\n".join(lines)


if __name__ == "__main__":
    # Test the module
    print("Testing external metrics fetcher...")

    raptor = fetch_raptor_data()
    print(f"RAPTOR data: {len(raptor)} records")

    if not raptor.empty:
        # Show available seasons
        seasons = sorted(raptor["season"].unique())
        print(f"Available seasons: {seasons}")

        # Show sample for recent season
        recent = raptor[raptor["season"] == max(seasons)].head(5)
        print("\nSample data (top 5 by D-RAPTOR):")
        print(
            recent[
                ["player_name", "season", "raptor_defense", "raptor_total"]
            ].to_string()
        )
