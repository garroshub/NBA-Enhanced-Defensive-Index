"""Tests for data_fetcher module."""

import pandas as pd
import pytest

from src.data_fetcher import get_all_defensive_teams, get_player_award_status


class TestGetAllDefensiveTeams:
    """Tests for get_all_defensive_teams function."""

    def test_returns_dataframe_for_valid_season(self):
        df = get_all_defensive_teams("2023-24")
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_contains_expected_columns(self):
        df = get_all_defensive_teams("2023-24")
        required_cols = ["PLAYER_NAME", "PLAYER_ID", "TEAM_LEVEL", "POSITION"]
        for col in required_cols:
            assert col in df.columns

    def test_contains_known_winner_gobert(self):
        df = get_all_defensive_teams("2023-24")
        assert "Rudy Gobert" in df["PLAYER_NAME"].values

    def test_has_first_and_second_team(self):
        df = get_all_defensive_teams("2023-24")
        assert "1st" in df["TEAM_LEVEL"].values
        assert "2nd" in df["TEAM_LEVEL"].values

    def test_returns_empty_for_future_season(self):
        df = get_all_defensive_teams("2030-31")
        assert df.empty


class TestGetPlayerAwardStatus:
    """Tests for get_player_award_status function."""

    def test_gobert_2023_24_first_team(self):
        gobert_id = 203497
        status = get_player_award_status(gobert_id, "2023-24")
        assert status == "1st"

    def test_non_winner_returns_none(self):
        # Use a player ID unlikely to have won
        status = get_player_award_status(999999, "2023-24")
        assert status is None
