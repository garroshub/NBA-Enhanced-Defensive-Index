"""
NBA EDI Defense Model - Shared Constants and Utility Functions
===============================================================

This module contains all shared constants and utility functions used across
the EDI defense evaluation system. All scripts should import from here to
ensure consistent behavior.

Version: 0.65
Last Updated: 2026-01-17
"""

import pandas as pd

# =============================================================================
# Model Version
# =============================================================================
EDI_VERSION = "0.7"

# =============================================================================
# Evaluation Seasons
# =============================================================================
EVALUATION_SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]

# =============================================================================
# Bayesian Shrinkage Constants
# =============================================================================
BAYES_C = 60  # Shrinkage constant (at n=C, data weight = 50%)

# =============================================================================
# Sigmoid Availability Constants
# =============================================================================
SIGMOID_G0 = 45  # Midpoint (games where availability = 0.5)
SIGMOID_K = 0.15  # Slope factor

# =============================================================================
# Roamer Classification Constants (Optimized 2025-01-17)
# =============================================================================
# Roamer_Pct threshold for classifying swing positions as Roamer
# Only swing positions (F, F-C, C-F, F-G, G-F) can be Roamers
# Pure C (Gobert) and pure G (Smart) stay in their base categories
ROAMER_THRESHOLD = 0.15

# D5 weight reduction sensitivity for Roamers
ROAMER_K = 0.3

# Weight redistribution for Roamers (when D5 weight is reduced)
# 30% to output layer (D1/D2), 70% to hustle layer (D3)
ROAMER_WEIGHT_REDIST_OUTPUT = 0.3
ROAMER_WEIGHT_REDIST_HUSTLE = 0.7

# D2 exterior weight adjustment for Roamers
D2_EXT_ROAMER_K = 0.5

# =============================================================================
# Synergy Bonus Constants (Optimized 2025-01-17)
# =============================================================================
# Rewards "switchable" defenders who can protect rim AND defend perimeter
SYNERGY_D1_THRESHOLD = 0.80  # D1 (matchup suppression) threshold
SYNERGY_D2_THRESHOLD = 0.75  # D2 (rim + perimeter) threshold
SYNERGY_FACTOR = 0.5  # Bonus multiplier

# =============================================================================
# Benchmark Evaluation Constants
# =============================================================================
POOL_SIZE = 30  # Top N players for recall calculations

# =============================================================================
# DPOY Ground Truth (with correct role classification)
# =============================================================================
DPOY_INFO = {
    "2019-20": ("Giannis Antetokounmpo", "Roamer"),  # F, Roamer_Pct=0.40
    "2020-21": ("Rudy Gobert", "Frontcourt"),  # C (pure center)
    "2021-22": ("Marcus Smart", "Backcourt"),  # G
    "2022-23": ("Jaren Jackson Jr.", "Roamer"),  # F-C, Roamer_Pct=1.00
    "2023-24": ("Rudy Gobert", "Frontcourt"),  # C (pure center)
}


# =============================================================================
# Position Classification Functions
# =============================================================================
def classify_role(position: str) -> str:
    """Classify player into Backcourt or Frontcourt (legacy 2-category).

    This is the simple binary classification used in the main EDI calculation.
    For DPOY evaluation, use classify_role_3cat() instead.

    Args:
        position: Player position string (e.g., 'G', 'F', 'C', 'F-C')

    Returns:
        'Guards' or 'Frontcourt'
    """
    if pd.isna(position) or position == "":
        return "Frontcourt"

    pos = str(position).upper().strip()

    # Pure guards -> Guards (Backcourt equivalent for weight calculation)
    if pos in ("G", "PG", "SG"):
        return "Guards"

    # Everything else -> Frontcourt
    return "Frontcourt"


def classify_role_3cat(
    position: str, roamer_pct: float = 0.0, threshold: float = ROAMER_THRESHOLD
) -> str:
    """Classify player into 3 categories: Backcourt, Roamer, or Frontcourt.

    Three-category classification based on positional versatility:
    - Backcourt: Pure guards (G, PG, SG) - never Roamer
    - Roamer: Swing positions (F, F-C, C-F, F-G, G-F) with high Roamer_Pct
    - Frontcourt: Pure centers (C) or low-Roamer forwards

    Key insight: Roamer is defined by POSITIONAL VERSATILITY, not just Roamer_Pct.
    - Pure C (like Gobert) stays Frontcourt even with high Roamer_Pct
    - Pure G stays Backcourt
    - Only swing positions can be Roamers

    Args:
        position: Player position string (e.g., 'G', 'F', 'C', 'G-F', 'F-C')
        roamer_pct: Player's Roamer percentile (0-1), from EDI data
        threshold: Percentile threshold for Roamer classification (default 0.15)

    Returns:
        'Backcourt', 'Roamer', or 'Frontcourt'
    """
    if pd.isna(position) or position == "":
        return "Frontcourt"  # Default to Frontcourt if unknown

    pos = str(position).upper().strip()

    # Pure Guards -> Backcourt (never Roamer)
    if pos in ("G", "PG", "SG"):
        return "Backcourt"

    # Pure Centers -> Frontcourt (never Roamer, even with high Roamer_Pct)
    # Gobert (C) stays Frontcourt - he's a traditional rim protector
    if pos == "C":
        return "Frontcourt"

    # Swing positions can be Roamer if Roamer_Pct >= threshold
    # F-C, C-F, F-G, G-F, F (pure forwards are swing by nature)
    is_swing = ("F" in pos) or ("-" in pos)
    if is_swing and roamer_pct >= threshold:
        return "Roamer"

    # Default: Frontcourt (low Roamer_Pct forwards, or other positions)
    return "Frontcourt"


# =============================================================================
# Bayesian Score Function
# =============================================================================
def bayesian_score(raw_pct: float, n: int, c: int = BAYES_C) -> tuple[float, float]:
    """Apply Bayesian shrinkage to raw percentile.

    Args:
        raw_pct: Raw percentile (0-1)
        n: Sample size
        c: Shrinkage constant (default BAYES_C)

    Returns:
        Tuple of (shrunk_score, confidence_weight)
    """
    shrunk = (n * raw_pct + c * 0.5) / (n + c)
    confidence = n / (n + c)
    return shrunk, confidence


# =============================================================================
# Sigmoid Availability Function
# =============================================================================
def sigmoid_availability(
    games: int, g0: int = SIGMOID_G0, k: float = SIGMOID_K
) -> float:
    """Calculate availability factor using Sigmoid function.

    - Below g0: Rapidly approaches 0 (filters low-GP players)
    - At g0: Returns 0.5 (passing threshold)
    - Above g0: Gradually approaches 1.0 (diminishing returns)

    Args:
        games: Games played
        g0: Sigmoid midpoint (default 45)
        k: Slope factor (default 0.15)

    Returns:
        Availability factor (0-1)
    """
    import numpy as np

    return 1 / (1 + np.exp(-k * (games - g0)))
