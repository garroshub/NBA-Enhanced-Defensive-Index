"""
NBA EDI Web Export Module
=========================
Generates JSON data for the React frontend application.

This module imports the core analysis logic from nba_defense_mvp.py
and transforms the output into a web-friendly JSON format.

Features:
- Multi-season support (current + 4 historical seasons)
- Dynamic Bayesian shrinkage based on season progress
- Dynamic Sigmoid availability adjustment for partial seasons
- Trend tracking (daily changes in EDI and ranks)
- Confidence indicators for sample size transparency

Usage:
    python src/web_export.py                    # Generate all seasons
    python src/web_export.py --current-only     # Generate current season only
"""

import json
import sys
import io
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Path configuration
BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DATA_DIR = BASE_DIR / "web" / "lib"
EXISTING_DATA_PATH = WEB_DATA_DIR / "data.json"

# Season configuration
HISTORICAL_SEASONS = ["2024-25", "2023-24", "2022-23", "2021-22"]
GAMES_PER_SEASON = 82

# Dynamic shrinkage constants
C_MIN = 20  # Minimum shrinkage (early season, allow more variance)
C_MAX = 60  # Maximum shrinkage (full season, standard robustness)
C_RAMP_FACTOR = 2.0  # How quickly C ramps up (reaches C_MAX at 50% season)

# Synergy Bonus thresholds (must match nba_defense_mvp.py)
SYNERGY_D1_THRESHOLD = 0.80
SYNERGY_D2_THRESHOLD = 0.75
SYNERGY_FACTOR = 0.5

# =============================================================================
# Partial Season Adjustment Model
# =============================================================================
# For partial seasons, we use a simplified "pure performance" approach:
# 1. Dynamic MIN_GP threshold filters out small samples
# 2. Players meeting threshold are evaluated equally (no ironman bonus)
# 3. Bayesian Efficiency Stabilization shrinks extreme efficiency values
#    toward 1.0 for players with fewer games (small sample correction)
#
# This ensures fair evaluation without arbitrary bonuses during the season.
# =============================================================================

# Threshold parameters
MIN_GP_RATIO = 0.40  # Minimum games = max_gp * 0.40 (e.g., 44 * 0.40 = 17.6 -> 18)
MIN_GP_FLOOR = 10  # Absolute minimum to avoid edge cases in very early season

# Bayesian Efficiency Stabilization
# Shrinks extreme efficiency values toward league average (1.0) for small samples
EFFICIENCY_SHRINKAGE_K = 10  # Equivalent "prior" games at league average efficiency


def calculate_dynamic_min_gp(max_gp: int) -> int:
    """
    Calculate dynamic minimum games played threshold based on season progress.

    The threshold scales with the current maximum games played, ensuring
    we always filter out small samples while adapting to the season progress.

    Args:
        max_gp: Maximum games played by any player in the dataset

    Returns:
        Minimum games required to qualify for rankings
    """
    dynamic_min = int(max_gp * MIN_GP_RATIO)
    return max(MIN_GP_FLOOR, dynamic_min)


def stabilize_efficiency(raw_efficiency: float, games_played: int) -> float:
    """
    Apply Bayesian shrinkage to stabilize efficiency values for small samples.

    For players with few games, extreme efficiency values (e.g., 1.5) are
    unreliable. This function shrinks them toward the league average (1.0)
    based on sample size.

    Formula:
        Eff_Stable = (Eff_Raw * GP + 1.0 * K) / (GP + K)

    Where:
        - 1.0 is the league average efficiency (Expected = Actual)
        - K is the shrinkage constant (default 10 games)

    Examples (K=10):
        - GP=20, Eff=1.5 -> (1.5*20 + 10) / 30 = 1.33
        - GP=40, Eff=1.5 -> (1.5*40 + 10) / 50 = 1.40
        - GP=20, Eff=0.7 -> (0.7*20 + 10) / 30 = 0.80

    Args:
        raw_efficiency: Original efficiency value from the model
        games_played: Number of games the player has played

    Returns:
        Stabilized efficiency value (shrunk toward 1.0)
    """
    league_avg = 1.0
    k = EFFICIENCY_SHRINKAGE_K
    return (raw_efficiency * games_played + league_avg * k) / (games_played + k)


def recalculate_edi_for_partial_season(df: pd.DataFrame, max_gp: int) -> pd.DataFrame:
    """
    Recalculate EDI_Total for partial seasons with Bayesian efficiency stabilization.

    This function reconstructs EDI from component scores (D1-D5, Weights,
    Efficiency) using a pure performance-based approach:
    - Players below dynamic MIN_GP are filtered out (small samples)
    - No ironman bonus - all qualified players evaluated equally
    - Efficiency values are stabilized via Bayesian shrinkage toward 1.0

    The Bayesian stabilization ensures that players with fewer games don't
    get artificially inflated/deflated EDI due to unstable efficiency values.

    Formula:
        Eff_Stable = (Eff_Raw * GP + 1.0 * K) / (GP + K)
        EDI_Raw = weighted_average(D1-D5 scores with Eff_Stable adjustment)
        EDI_Total = EDI_Raw + Synergy_Bonus

    Args:
        df: DataFrame with all component columns from analyze_season()
        max_gp: Maximum games played by any player

    Returns:
        DataFrame with recalculated EDI_Total
    """
    df = df.copy()

    # Calculate dynamic minimum threshold
    min_gp = calculate_dynamic_min_gp(max_gp)
    print(f"   Dynamic MIN_GP threshold: {min_gp} games (max_gp={max_gp})")

    # Verify required columns exist
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
        "Efficiency",
        "GP",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"Warning: Missing columns for EDI recalculation: {missing}")
        return df

    # Step 1: Apply Bayesian Efficiency Stabilization
    # Shrink extreme efficiency values toward 1.0 based on sample size
    df["Efficiency_Stable"] = df.apply(
        lambda row: stabilize_efficiency(row["Efficiency"], row["GP"]), axis=1
    )

    # Step 2: Calculate Actual_Output (D1 + D2 weighted average)
    w1_w2_sum = df["W1"] + df["W2"] + 1e-6
    df["Actual_Output_Recalc"] = (
        df["D1_Score"] * df["W1"] + df["D2_Score"] * df["W2"]
    ) / w1_w2_sum

    # Step 3: Calculate Input_Score (D3 + D4 weighted average)
    w3_w4_sum = df["W3"] + df["W4"] + 1e-6
    df["Input_Score_Recalc"] = (
        df["D3_Score"] * df["W3"] + df["D4_Score"] * df["W4"]
    ) / w3_w4_sum

    # Step 4: Calculate EDI_Raw using stabilized efficiency
    output_weighted = (
        df["Actual_Output_Recalc"] * df["Efficiency_Stable"] * (df["W1"] + df["W2"])
    )
    input_weighted = df["Input_Score_Recalc"] * (df["W3"] + df["W4"])
    d5_weighted = df["D5_Score"] * df["W5"]

    total_weight = df["W1"] + df["W2"] + df["W3"] + df["W4"] + df["W5"]

    df["EDI_Raw"] = np.where(
        total_weight > 0,
        (output_weighted + input_weighted + d5_weighted) / total_weight * 100,
        50.0,
    )

    # Step 5: Calculate Synergy Bonus
    def calc_synergy_bonus(row):
        d1, d2 = row["D1_Score"], row["D2_Score"]
        if d1 >= SYNERGY_D1_THRESHOLD and d2 >= SYNERGY_D2_THRESHOLD:
            raw_synergy = (d1 - SYNERGY_D1_THRESHOLD) * (d2 - SYNERGY_D2_THRESHOLD)
            return np.sqrt(raw_synergy) * SYNERGY_FACTOR * 100
        return 0.0

    df["Synergy_Bonus_Recalc"] = df.apply(calc_synergy_bonus, axis=1)

    # Step 6: Final EDI_Total (no availability bonus, pure performance)
    df["EDI_Total"] = df["EDI_Raw"] + df["Synergy_Bonus_Recalc"]

    # Update Efficiency column to use stabilized value for display
    df["Efficiency"] = df["Efficiency_Stable"]

    # Log recalculation summary
    raw_max = df["EDI_Raw"].max()
    final_max = df["EDI_Total"].max()
    qualified_count = len(df[df["GP"] >= min_gp])
    eff_shrinkage_example = stabilize_efficiency(1.5, 20)  # Example: GP=20, Eff=1.5

    print(f"   Bayesian Efficiency Stabilization: K={EFFICIENCY_SHRINKAGE_K}")
    print(f"   Example: Eff=1.5, GP=20 -> Eff_Stable={eff_shrinkage_example:.3f}")
    print(f"   Qualified players (GP>={min_gp}): {qualified_count}")
    print(f"   EDI recalculated: Raw max={raw_max:.1f}, Final max={final_max:.1f}")

    return df


def fetch_team_data(season: str) -> pd.DataFrame:
    """
    Fetch player-to-team mapping from nba_api.

    Args:
        season: Season string (e.g., "2025-26")

    Returns:
        DataFrame with PLAYER_ID and TEAM_ABBREVIATION columns
    """
    import time
    from nba_api.stats.endpoints import leaguedashplayerstats

    try:
        time.sleep(0.6)  # Rate limit
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season, per_mode_detailed="PerGame"
        )
        df = stats.get_data_frames()[0]
        return df[["PLAYER_ID", "TEAM_ABBREVIATION"]].drop_duplicates()
    except Exception as e:
        print(f"Warning: Could not fetch team data: {e}")
        return pd.DataFrame()


def detect_current_season() -> str:
    """
    Detect the current NBA season based on system date.
    NBA season spans October to June.

    Returns:
        Season string in format "YYYY-YY" (e.g., "2025-26")
    """
    now = datetime.now()
    year = now.year
    month = now.month

    if month >= 10:
        # October-December: new season started
        return f"{year}-{str(year + 1)[2:]}"
    elif month <= 6:
        # January-June: season in progress from previous year
        return f"{year - 1}-{str(year)[2:]}"
    else:
        # July-September: offseason, use last completed season
        return f"{year - 1}-{str(year)[2:]}"


def calculate_dynamic_c(games_played: int, max_games: int = GAMES_PER_SEASON) -> float:
    """
    Calculate dynamic Bayesian shrinkage constant based on season progress.

    Early season: Lower C allows more variance in scores.
    Late season: Higher C provides robust estimates.

    Args:
        games_played: Maximum games played by any player in the dataset
        max_games: Total games in a full season (default 82)

    Returns:
        Dynamic shrinkage constant C
    """
    season_progress = min(1.0, games_played / max_games)
    # Ramp up C faster than linear (reaches C_MAX at ~50% of season)
    ramp = min(1.0, season_progress * C_RAMP_FACTOR)
    dynamic_c = C_MIN + (C_MAX - C_MIN) * ramp
    return dynamic_c


def calculate_confidence_level(games_played: int) -> str:
    """
    Determine confidence level based on games played.

    Args:
        games_played: Number of games the player has played

    Returns:
        Confidence level: "high", "medium", or "low"
    """
    if games_played >= 40:
        return "high"
    elif games_played >= 20:
        return "medium"
    else:
        return "low"


def load_existing_data() -> Optional[dict]:
    """
    Load existing data.json for trend comparison.

    Returns:
        Parsed JSON data or None if file does not exist
    """
    if EXISTING_DATA_PATH.exists():
        try:
            with open(EXISTING_DATA_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load existing data: {e}")
            return None
    return None


def calculate_trends(
    current_players: list[dict], old_data: Optional[dict], season: str
) -> list[dict]:
    """
    Calculate EDI and rank changes compared to previous data.

    Args:
        current_players: List of current player dictionaries
        old_data: Previously saved data.json content
        season: Season string to look up in old data

    Returns:
        Updated player list with trend information
    """
    if old_data is None or season not in old_data.get("seasons", {}):
        # No previous data, mark all as new
        for player in current_players:
            player["trend"] = {"edi_change": None, "rank_change": None, "status": "new"}
        return current_players

    # Build lookup from old data
    old_players = {p["id"]: p for p in old_data["seasons"][season]}

    for player in current_players:
        player_id = player["id"]
        if player_id in old_players:
            old_player = old_players[player_id]
            old_edi = old_player.get("scores", {}).get("edi", 0)
            old_rank = old_player.get("ranks", {}).get("overall", 0)

            edi_change = round(player["scores"]["edi"] - old_edi, 1)
            rank_change = old_rank - player["ranks"]["overall"]  # Positive = improved

            if edi_change > 0.5:
                status = "up"
            elif edi_change < -0.5:
                status = "down"
            else:
                status = "stable"

            player["trend"] = {
                "edi_change": edi_change,
                "rank_change": rank_change,
                "status": status,
            }
        else:
            player["trend"] = {"edi_change": None, "rank_change": None, "status": "new"}

    return current_players


def format_player_for_web(row: pd.Series, rank: int) -> dict:
    """
    Transform a DataFrame row into web-friendly JSON structure.

    Args:
        row: Pandas Series containing player data
        rank: Overall EDI rank

    Returns:
        Formatted player dictionary
    """

    # Handle NaN values
    def safe_float(val, decimals=1):
        if pd.isna(val):
            return None
        return round(float(val), decimals)

    def safe_int(val):
        if pd.isna(val):
            return None
        return int(val)

    # Convert D1-D5 scores to 0-100 scale (they are stored as 0-1)
    def score_to_100(val):
        if pd.isna(val):
            return None
        return round(float(val) * 100, 1)

    return {
        "id": safe_int(row.get("PLAYER_ID")),
        "name": str(row.get("PLAYER_NAME", "Unknown")),
        "team": str(row.get("TEAM_ABBREVIATION", "N/A"))
        if pd.notna(row.get("TEAM_ABBREVIATION"))
        else None,
        "position": str(row.get("PLAYER_POSITION", "N/A"))
        if pd.notna(row.get("PLAYER_POSITION"))
        else None,
        "role": str(row.get("ROLE", "Unknown")),
        "scores": {
            "edi": safe_float(row.get("EDI_Total")),
            "d1": score_to_100(row.get("D1_Score")),
            "d2": score_to_100(row.get("D2_Score")),
            "d3": score_to_100(row.get("D3_Score")),
            "d4": score_to_100(row.get("D4_Score")),
            "d5": score_to_100(row.get("D5_Score")),
            "efficiency": safe_float(row.get("Efficiency"), 2),
        },
        "ranks": {"overall": rank},
        "stats": {
            "gp": safe_int(row.get("GP")),
            "min": safe_float(row.get("MIN")),
            "stl": safe_float(row.get("STL")),
            "blk": safe_float(row.get("BLK")),
        },
        "confidence": calculate_confidence_level(safe_int(row.get("GP")) or 0),
    }


def generate_season_data(
    season: str, is_current: bool = False
) -> tuple[list[dict], dict]:
    """
    Generate web data for a single season.

    Args:
        season: Season string (e.g., "2025-26")
        is_current: Whether this is the current in-progress season

    Returns:
        Tuple of (player_list, season_metadata)
    """
    # Import the core analysis function
    # We do this inside the function to avoid circular imports
    # and to allow the module to be imported without running analysis
    from nba_defense_mvp import analyze_season

    print(f"\n{'=' * 60}")
    print(f"Generating data for season: {season}")
    print(f"{'=' * 60}")

    try:
        df = analyze_season(season)
    except Exception as e:
        print(f"Error analyzing season {season}: {e}")
        return [], {"error": str(e)}

    if df.empty:
        print(f"Warning: No data returned for season {season}")
        return [], {"error": "No data available"}

    # Fetch and merge team data
    team_df = fetch_team_data(season)
    if not team_df.empty:
        df = df.merge(team_df, on="PLAYER_ID", how="left")
        print(
            f"   Team data merged for {df['TEAM_ABBREVIATION'].notna().sum()} players"
        )
    else:
        df["TEAM_ABBREVIATION"] = None
        print("   Warning: Team data unavailable")

    # Calculate season progress for current season
    max_gp = int(df["GP"].max())
    season_progress = min(1.0, max_gp / GAMES_PER_SEASON)

    # Calculate dynamic C for display purposes
    dynamic_c = calculate_dynamic_c(max_gp)

    print(f"Season progress: {season_progress * 100:.1f}% ({max_gp} games)")
    print(f"Dynamic shrinkage C: {dynamic_c:.1f}")

    # Recalculate EDI using dynamic Sigmoid for partial seasons
    # This fixes the issue where fixed g0=45 crushes scores when max_gp < 45
    if season_progress < 0.90:  # Less than ~74 games played
        print("   Applying dynamic Sigmoid recalculation for partial season...")
        df = recalculate_edi_for_partial_season(df, max_gp)

    # Sort by EDI and assign ranks
    df_sorted = df.sort_values("EDI_Total", ascending=False).reset_index(drop=True)

    # Format each player
    players = []
    for idx, row in df_sorted.iterrows():
        rank = idx + 1
        player_data = format_player_for_web(row, rank)
        players.append(player_data)

    # Season metadata
    metadata = {
        "total_players": len(players),
        "max_games_played": max_gp,
        "season_progress": round(season_progress, 2),
        "dynamic_c": round(dynamic_c, 1),
        "is_current": is_current,
    }

    print(f"Processed {len(players)} players")

    return players, metadata


def generate_all_data(current_only: bool = False) -> dict:
    """
    Generate complete web data for all configured seasons.

    Args:
        current_only: If True, only generate current season data

    Returns:
        Complete data structure for web application
    """
    current_season = detect_current_season()
    print(f"Detected current season: {current_season}")

    # Load existing data for trend calculation
    old_data = load_existing_data()
    if old_data:
        print("Loaded existing data for trend comparison")
    else:
        print("No existing data found, trends will be marked as new")

    # Determine which seasons to process
    if current_only:
        seasons_to_process = [current_season]
    else:
        seasons_to_process = [current_season] + HISTORICAL_SEASONS

    # Remove duplicates while preserving order
    seasons_to_process = list(dict.fromkeys(seasons_to_process))

    print(f"Seasons to process: {seasons_to_process}")

    # Build output structure
    output = {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "current_season": current_season,
            "available_seasons": seasons_to_process,
            "model_version": "EDI v2.4",
        },
        "seasons": {},
    }

    # Process each season
    for season in seasons_to_process:
        is_current = season == current_season
        players, metadata = generate_season_data(season, is_current)

        if players:
            # Calculate trends for current season
            if is_current:
                players = calculate_trends(players, old_data, season)

            output["seasons"][season] = players
            output["meta"][f"{season}_info"] = metadata

    return output


def save_data(data: dict) -> Path:
    """
    Save generated data to JSON file.

    Args:
        data: Complete data structure

    Returns:
        Path to saved file
    """
    # Ensure output directory exists
    WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_path = WEB_DATA_DIR / "data.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nData saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")

    return output_path


def main():
    """Main entry point for web data generation."""
    print("=" * 60)
    print("NBA EDI Web Export")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Parse command line arguments
    current_only = "--current-only" in sys.argv

    if current_only:
        print("Mode: Current season only")
    else:
        print("Mode: All seasons (current + historical)")

    # Generate data
    data = generate_all_data(current_only=current_only)

    # Save to file
    if len(data["seasons"]) > 0:
        # Check if any season has 0 players
        valid_data = True
        for season, players in data["seasons"].items():
            if len(players) == 0:
                print(f"Error: Season {season} has 0 players! Aborting save.")
                valid_data = False
                break

        if valid_data:
            output_path = save_data(data)
        else:
            print("Export aborted due to empty data.")
    else:
        print("Error: No seasons generated! Aborting save.")

    # Summary
    print("\n" + "=" * 60)
    print("Export Complete")
    print("=" * 60)
    print(f"Seasons exported: {len(data['seasons'])}")
    for season, players in data["seasons"].items():
        print(f"  {season}: {len(players)} players")

    return output_path


if __name__ == "__main__":
    main()
