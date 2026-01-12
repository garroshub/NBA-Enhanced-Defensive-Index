# Evaluation System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a 2-layer evaluation system for defensive metrics: (1) Coverage of All-Defensive Team selections, (2) Correlation with advanced defensive stats.

**Architecture:** 
- A new Python script `src/evaluate_defense.py` 
- Uses `PlayerAwards` for ground truth labels (Layer 1)
- Uses `BoxScoreAdvancedV2` + existing metrics for stat correlation (Layer 2)
- Outputs a text report comparing our `EDI` score vs. reality

**Tech Stack:** `nba_api`, `pandas`, `scipy.stats` (Spearman/Pearson), `numpy`

## Task 1: Fetch Ground Truth Data (All-Defensive Teams)

**Files:**
- Create: `src/data_fetcher.py` (extract data logic from mvp script)
- Test: `tests/test_data_fetcher.py`

**Step 1: Write the failing test**

```python
import pandas as pd
from src.data_fetcher import get_all_defensive_roster

def test_get_all_defensive_roster():
    # Test with a known historical season
    df = get_all_defensive_roster("2023-24")
    
    assert not df.empty
    assert "Rudy Gobert" in df["PLAYER_NAME"].values
    assert "First Team" in df["TEAM_LEVEL"].values  # or ALL_NBA_TEAM_NUMBER
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_data_fetcher.py -v`

**Step 3: Write minimal implementation**

```python
from nba_api.stats.endpoints import playerawards
import pandas as pd

# Note: This is tricky because playerawards takes player_id. 
# We need a strategy: either iterate known top defenders OR scrape.
# Given constraints, let's use a cached list or iterate active roster if small.
# BETTER APPROACH for first pass:
# Use 'commonallplayers' to get active players, then filter? Too slow (500+ calls).
# ALTERNATIVE: Use `leaguedashplayerawards` IF it exists? No.
#
# NEW STRATEGY: Hardcode/Cache recent years for V0 or iterate ONLY target list.
# BUT evaluating "Coverage" requires the FULL list.
#
# Let's use `basketball_reference_web_scraper` approach via direct HTML parse or library?
# Let's stick to `nba_api` but maybe we can't get FULL roster easily without many calls.
#
# WAIT: `playerawards` is per player. 
# `commonallplayers` gives us everyone. 
#
# REVISED PLAN for this step: 
# Implement a function that accepts a list of player_IDs to check, 
# OR use a static known list for 2024/25 validation for now?
# 
# Actually, `nba_api` doesn't have a "get all awards for season" endpoint.
# We will implement a `get_all_defensive_roster` that uses a pre-fetched mapping or limited scan.
#
# FOR V0: Let's assume we pass in the dataframe of players we analyzed, 
# and we check WHICH of them won awards. (Coverage of OUR predictions vs ACTUAL awards).
#
# So the function signature changes: `get_award_status(player_ids, season)`
```

Refining test for `get_award_status`:

```python
def test_get_award_status():
    # Mocking or using real ID for known winner
    gobert_id = 203497
    status = get_player_award_status(gobert_id, "2023-24")
    assert status == "1st Team" 
```

**Step 4: Run test**

**Step 5: Commit**

## Task 2: Calculate Coverage Metric (Layer 1)

**Files:**
- Create: `src/evaluation.py`
- Test: `tests/test_evaluation.py`

**Step 1: Write failing test**

```python
import pandas as pd
from src.evaluation import calculate_coverage

def test_calculate_coverage():
    # Setup: 10 players, 5 are "real" All-Defense
    # Our model ranks them.
    # We want to know: Of the top 5 our model picked, how many are real?
    
    data = {
        "PLAYER_ID": range(10),
        "EDI_Score": [90, 80, 70, 60, 50, 40, 30, 20, 10, 0], # Our rank
        "Is_All_Defense": [1, 1, 0, 1, 0, 0, 0, 0, 0, 0] # Real labels (3 winners in top 5)
    }
    df = pd.DataFrame(data)
    
    # Top 5 coverage
    coverage = calculate_coverage(df, top_n=5)
    
    # 3 out of 5 winners found in our top 5? 
    # Or 3 out of total winners (3) found in our top 5? (Recall)
    # Let's define metric: "Precision@K"
    
    # In this case: Top 5 predictions have 3 winners. Precision@5 = 0.6
    assert coverage["precision@5"] == 0.6
```

**Step 2: Run test**

**Step 3: Implement**

```python
def calculate_coverage(df, top_n=10):
    sorted_df = df.sort_values("EDI_Score", ascending=False).head(top_n)
    hits = sorted_df["Is_All_Defense"].sum()
    return {"precision@k": hits / top_n}
```

**Step 4: Verify**

**Step 5: Commit**

## Task 3: Correlation Analysis (Layer 2)

**Files:**
- Modify: `src/evaluation.py`
- Test: `tests/test_evaluation.py`

**Step 1: Write test**

```python
def test_correlation_analysis():
    data = {
        "EDI_Score": [10, 20, 30, 40, 50],
        "DEF_RATING": [110, 108, 105, 100, 95] # Lower is better, so negative correlation expected
    }
    df = pd.DataFrame(data)
    
    corr = calculate_correlations(df, target_col="EDI_Score", ref_cols=["DEF_RATING"])
    
    assert "DEF_RATING" in corr
    assert corr["DEF_RATING"] < 0 # Strong negative correlation
```

**Step 2: Run test**

**Step 3: Implement**

**Step 4: Verify**

**Step 5: Commit**

## Task 4: Main Evaluation Script

**Files:**
- Create: `src/run_evaluation.py`
- Test: Integration test

**Step 1: Script flow**
1. Load `nba_defensive_all_players_202X.csv` (our model output)
2. Fetch Ground Truth (Awards) for that season
3. Fetch Advanced Stats (if not in CSV)
4. Merge
5. Run Layer 1 (Coverage)
6. Run Layer 2 (Correlation)
7. Print Report

**Step 2: Commit**
