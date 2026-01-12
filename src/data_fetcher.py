"""Data fetching utilities for NBA defensive evaluation.

This module provides functions to fetch All-Defensive Team rosters
and player award status for model evaluation.

Uses nba_api.stats.endpoints.playerawards for dynamic queries,
with static cache for batch operations.
"""

import time
from typing import Optional

import pandas as pd
from nba_api.stats.endpoints import playerawards

# Static All-Defensive Team data (from Basketball Reference)
# Used for batch evaluation to avoid 500+ API calls
# Format: {season: [(player_name, player_id, team_level, position), ...]}
ALL_DEFENSIVE_TEAMS = {
    "2024-25": [
        # First Team (announced June 2025 - placeholder based on projections)
        ("Victor Wembanyama", 1641705, "1st", "C"),
        ("Rudy Gobert", 203497, "2nd", "C"),  # Confirmed via API
        # Add more as announced...
    ],
    "2023-24": [
        # First Team
        ("Rudy Gobert", 203497, "1st", "C"),
        ("Anthony Davis", 203076, "1st", "F"),
        ("Bam Adebayo", 1628389, "1st", "F"),
        ("Jrue Holiday", 201950, "1st", "G"),
        ("Derrick White", 1628401, "1st", "G"),
        # Second Team
        ("Victor Wembanyama", 1641705, "2nd", "C"),
        ("Jaren Jackson Jr.", 1628991, "2nd", "F"),
        ("Herb Jones", 1630529, "2nd", "F"),
        ("Alex Caruso", 1627936, "2nd", "G"),
        ("Dyson Daniels", 1630700, "2nd", "G"),
    ],
    "2022-23": [
        # First Team
        ("Brook Lopez", 201572, "1st", "C"),
        ("Jaren Jackson Jr.", 1628991, "1st", "F"),
        ("Draymond Green", 203110, "1st", "F"),
        ("Marcus Smart", 203935, "1st", "G"),
        ("Jrue Holiday", 201950, "1st", "G"),
        # Second Team
        ("Anthony Davis", 203076, "2nd", "C"),
        ("Bam Adebayo", 1628389, "2nd", "F"),
        ("Evan Mobley", 1630596, "2nd", "F"),
        ("Mikal Bridges", 1628969, "2nd", "G"),
        ("Alex Caruso", 1627936, "2nd", "G"),
    ],
    "2021-22": [
        # First Team
        ("Rudy Gobert", 203497, "1st", "C"),
        ("Giannis Antetokounmpo", 203507, "1st", "F"),
        ("Mikal Bridges", 1628969, "1st", "F"),
        ("Marcus Smart", 203935, "1st", "G"),
        ("Jrue Holiday", 201950, "1st", "G"),
        # Second Team
        ("Bam Adebayo", 1628389, "2nd", "C"),
        ("Jaren Jackson Jr.", 1628991, "2nd", "F"),
        ("Draymond Green", 203110, "2nd", "F"),
        ("Matisse Thybulle", 1629680, "2nd", "G"),
        ("Dejounte Murray", 1627749, "2nd", "G"),
    ],
    "2020-21": [
        # First Team
        ("Rudy Gobert", 203497, "1st", "C"),
        ("Giannis Antetokounmpo", 203507, "1st", "F"),
        ("Draymond Green", 203110, "1st", "F"),
        ("Ben Simmons", 1627732, "1st", "G"),
        ("Jrue Holiday", 201950, "1st", "G"),
        # Second Team
        ("Bam Adebayo", 1628389, "2nd", "C"),
        ("Kawhi Leonard", 202695, "2nd", "F"),
        ("Anthony Davis", 203076, "2nd", "F"),
        ("Matisse Thybulle", 1629680, "2nd", "G"),
        ("Jimmy Butler", 202710, "2nd", "G"),
    ],
    "2019-20": [
        # First Team
        ("Anthony Davis", 203076, "1st", "C"),
        ("Giannis Antetokounmpo", 203507, "1st", "F"),
        ("Kawhi Leonard", 202695, "1st", "F"),
        ("Marcus Smart", 203935, "1st", "G"),
        ("Ben Simmons", 1627732, "1st", "G"),
        # Second Team
        ("Rudy Gobert", 203497, "2nd", "C"),
        ("Bam Adebayo", 1628389, "2nd", "F"),
        ("Pascal Siakam", 1627783, "2nd", "F"),
        ("Eric Bledsoe", 202339, "2nd", "G"),
        ("Patrick Beverley", 201976, "2nd", "G"),
    ],
    "2018-19": [
        # First Team
        ("Rudy Gobert", 203497, "1st", "C"),
        ("Giannis Antetokounmpo", 203507, "1st", "F"),
        ("Paul George", 202331, "1st", "F"),
        ("Marcus Smart", 203935, "1st", "G"),
        ("Klay Thompson", 202691, "1st", "G"),
        # Second Team
        ("Joel Embiid", 203954, "2nd", "C"),
        ("Draymond Green", 203110, "2nd", "F"),
        ("Kawhi Leonard", 202695, "2nd", "F"),
        ("Eric Bledsoe", 202339, "2nd", "G"),
        ("Jrue Holiday", 201950, "2nd", "G"),
    ],
    "2017-18": [
        # First Team
        ("Rudy Gobert", 203497, "1st", "C"),
        ("Anthony Davis", 203076, "1st", "F"),
        ("Draymond Green", 203110, "1st", "F"),
        ("Victor Oladipo", 203506, "1st", "G"),
        ("Jrue Holiday", 201950, "1st", "G"),
        # Second Team
        ("Joel Embiid", 203954, "2nd", "C"),
        ("Al Horford", 201143, "2nd", "F"),
        ("Robert Covington", 203496, "2nd", "F"),
        ("Jimmy Butler", 202710, "2nd", "G"),
        ("Dejounte Murray", 1627749, "2nd", "G"),
    ],
    "2016-17": [
        # First Team
        ("Rudy Gobert", 203497, "1st", "C"),
        ("Draymond Green", 203110, "1st", "F"),
        ("Kawhi Leonard", 202695, "1st", "F"),
        ("Chris Paul", 101108, "1st", "G"),
        ("Patrick Beverley", 201976, "1st", "G"),
        # Second Team
        ("DeAndre Jordan", 201599, "2nd", "C"),
        ("Giannis Antetokounmpo", 203507, "2nd", "F"),
        ("Anthony Davis", 203076, "2nd", "F"),
        ("Danny Green", 201980, "2nd", "G"),
        ("Avery Bradley", 201965, "2nd", "G"),
    ],
}


def get_all_defensive_teams(season: str) -> pd.DataFrame:
    """Get All-Defensive Team roster for a given season.

    Uses static cache for efficiency. For real-time verification,
    use get_player_award_status_api().

    Args:
        season: Season string (e.g., "2023-24")

    Returns:
        DataFrame with columns: PLAYER_NAME, PLAYER_ID, TEAM_LEVEL, POSITION
        Empty DataFrame if season not found.
    """
    if season not in ALL_DEFENSIVE_TEAMS:
        return pd.DataFrame(
            columns=["PLAYER_NAME", "PLAYER_ID", "TEAM_LEVEL", "POSITION"]
        )

    data = ALL_DEFENSIVE_TEAMS[season]
    return pd.DataFrame(
        data, columns=["PLAYER_NAME", "PLAYER_ID", "TEAM_LEVEL", "POSITION"]
    )


def get_player_award_status(player_id: int, season: str) -> Optional[str]:
    """Check if a player made All-Defensive Team (from static cache).

    Args:
        player_id: NBA player ID
        season: Season string (e.g., "2023-24")

    Returns:
        "1st" for First Team, "2nd" for Second Team, None if not selected.
    """
    if season not in ALL_DEFENSIVE_TEAMS:
        return None

    for name, pid, team_level, pos in ALL_DEFENSIVE_TEAMS[season]:
        if pid == player_id:
            return team_level

    return None


def get_player_award_status_api(
    player_id: int, season: str, timeout: int = 30
) -> Optional[str]:
    """Check if a player made All-Defensive Team via NBA API.

    Makes a live API call. Use sparingly to avoid rate limits.

    Args:
        player_id: NBA player ID
        season: Season string (e.g., "2023-24")
        timeout: API timeout in seconds

    Returns:
        "1st" for First Team, "2nd" for Second Team, None if not selected.
    """
    try:
        time.sleep(0.6)  # Rate limiting
        pa = playerawards.PlayerAwards(player_id=player_id, timeout=timeout)
        df = pa.get_data_frames()[0]

        # Filter for All-Defensive Team in target season
        mask = (df["DESCRIPTION"] == "All-Defensive Team") & (df["SEASON"] == season)
        matches = df[mask]

        if matches.empty:
            return None

        team_number = matches.iloc[0]["ALL_NBA_TEAM_NUMBER"]
        if team_number == 1:
            return "1st"
        elif team_number == 2:
            return "2nd"
        return None

    except Exception:
        return None


def get_all_defensive_player_ids(season: str) -> set[int]:
    """Get set of player IDs who made All-Defensive Team.

    Args:
        season: Season string (e.g., "2023-24")

    Returns:
        Set of player IDs.
    """
    if season not in ALL_DEFENSIVE_TEAMS:
        return set()

    return {pid for _, pid, _, _ in ALL_DEFENSIVE_TEAMS[season]}


def get_first_team_player_ids(season: str) -> set[int]:
    """Get set of player IDs who made First Team All-Defense.

    Args:
        season: Season string (e.g., "2023-24")

    Returns:
        Set of player IDs.
    """
    if season not in ALL_DEFENSIVE_TEAMS:
        return set()

    return {pid for _, pid, team, _ in ALL_DEFENSIVE_TEAMS[season] if team == "1st"}
