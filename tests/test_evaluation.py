"""Tests for evaluation module."""

import pandas as pd
import pytest

from src.evaluation import calculate_coverage, calculate_correlations


class TestCalculateCoverage:
    """Tests for coverage metrics."""

    def test_precision_at_k_basic(self):
        """Test precision@k with known data."""
        data = {
            "PLAYER_ID": range(10),
            "EDI_Total": [90, 80, 70, 60, 50, 40, 30, 20, 10, 0],
            "Is_All_Defense": [1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        }
        df = pd.DataFrame(data)

        coverage = calculate_coverage(df, top_n=5)

        # Top 5 have 3 winners: precision@5 = 3/5 = 0.6
        assert coverage["precision@5"] == 0.6

    def test_recall_at_k_basic(self):
        """Test recall@k with known data."""
        data = {
            "PLAYER_ID": range(10),
            "EDI_Total": [90, 80, 70, 60, 50, 40, 30, 20, 10, 0],
            "Is_All_Defense": [1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        }
        df = pd.DataFrame(data)

        coverage = calculate_coverage(df, top_n=5)

        # 3 total winners, 3 found in top 5: recall@5 = 3/3 = 1.0
        assert coverage["recall@5"] == 1.0

    def test_perfect_coverage(self):
        """Test when model perfectly identifies all winners."""
        data = {
            "PLAYER_ID": range(10),
            "EDI_Total": [100, 90, 80, 70, 60, 50, 40, 30, 20, 10],
            "Is_All_Defense": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        }
        df = pd.DataFrame(data)

        coverage = calculate_coverage(df, top_n=5)

        assert coverage["precision@5"] == 1.0
        assert coverage["recall@5"] == 1.0

    def test_zero_coverage(self):
        """Test when model misses all winners."""
        data = {
            "PLAYER_ID": range(10),
            "EDI_Total": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "Is_All_Defense": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        }
        df = pd.DataFrame(data)

        coverage = calculate_coverage(df, top_n=5)

        assert coverage["precision@5"] == 0.0
        assert coverage["recall@5"] == 0.0


class TestCalculateCorrelations:
    """Tests for correlation analysis."""

    def test_perfect_negative_correlation(self):
        """Test with perfectly negatively correlated data."""
        data = {
            "EDI_Total": [10, 20, 30, 40, 50],
            "DEF_RATING": [110, 108, 105, 100, 95],
        }
        df = pd.DataFrame(data)

        corr = calculate_correlations(
            df, target_col="EDI_Total", ref_cols=["DEF_RATING"]
        )

        assert "DEF_RATING" in corr
        # Strong negative correlation expected (lower DEF_RATING = better)
        assert corr["DEF_RATING"]["pearson"] < -0.9

    def test_multiple_reference_columns(self):
        """Test with multiple reference columns."""
        data = {
            "EDI_Total": [10, 20, 30, 40, 50],
            "DEF_RATING": [110, 108, 105, 100, 95],
            "STL": [0.5, 1.0, 1.5, 2.0, 2.5],
            "BLK": [0.2, 0.4, 0.6, 0.8, 1.0],
        }
        df = pd.DataFrame(data)

        corr = calculate_correlations(
            df, target_col="EDI_Total", ref_cols=["DEF_RATING", "STL", "BLK"]
        )

        assert "DEF_RATING" in corr
        assert "STL" in corr
        assert "BLK" in corr
        # STL and BLK should have positive correlation
        assert corr["STL"]["pearson"] > 0.9
        assert corr["BLK"]["pearson"] > 0.9
