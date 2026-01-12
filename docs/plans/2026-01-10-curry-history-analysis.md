# Stephen Curry Career Analysis Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a historical analysis mode in `nba_defense_mvp.py` to compare Stephen Curry's defensive metrics across three eras: Rookie (2009-10), Dynasty (2016-17), and Current (2024-25).

**Architecture:** Refactor the existing linear script into a reusable `analyze_season(season)` function. Add a `--history` CLI flag that triggers a multi-season analysis, extracts Curry's data, and generates a comparison radar chart.

**Tech Stack:** Python, `nba_api`, `pandas`, `matplotlib`.

### Task 1: Refactor Main Logic into Function

**Files:**
- Modify: `D:\OpenCode\Projects\NBA\nba_defense_mvp.py`

**Step 1: Define `analyze_season` function**
Wrap the current "STEP 1" (Fetch) and "STEP 2" (Process) logic into a function.
- Input: `season` (str)
- Output: `base_df` (DataFrame containing calculated scores for all players)
- Keep the global `ROLE_CONFIG` and helper functions (`get_league_data`, `standardize_pt_defend_columns`, `bayesian_score`, `classify_role`) outside.

**Step 2: Update Main Execution Block**
- Modify the `if __name__ == "__main__":` block (or script level code) to call `analyze_season(SEASON)` for the default case.
- Ensure the output logic (STEP 3 & 4) remains functional using the returned `base_df`.

**Step 3: Verify Default Behavior**
- Run `python nba_defense_mvp.py 2024-25` to ensure it still produces the same reports and charts as before.

### Task 2: Implement Historical Analysis Logic

**Files:**
- Modify: `D:\OpenCode\Projects\NBA\nba_defense_mvp.py`

**Step 1: Add `--history` argument support**
- Update `sys.argv` handling to detect `--history`.

**Step 2: Implement Multi-Season Loop**
- If `--history` is detected:
    - Define seasons: `["2009-10", "2016-17", "2024-25"]`.
    - Loop through each season and call `analyze_season(season)`.
    - Extract Stephen Curry's data (`PLAYER_ID = 201939` or Name match).
    - Store the results (Scores D1-D5, EDI) in a list.

**Step 3: Handle Missing Data Gracefully**
- 2009-10 will lack D1, D2, D3.
- Ensure the scores default to 0.5 (as per current logic) but maybe mark them for the chart?
- For now, just relying on the existing "fill 0.5" logic is acceptable, but we should print a summary of what was found.

### Task 3: Generate Historical Comparison Chart

**Files:**
- Modify: `D:\OpenCode\Projects\NBA\nba_defense_mvp.py`

**Step 1: Create `plot_history_radar` function**
- Implement a function to plot a comparison radar chart.
- Input: `history_df` (DataFrame with 3 rows: Rookie, Dynasty, Current).
- Metrics: D1, D2, D3, D4, D5.
- Style: Overlay 3 polygons with different colors/styles (e.g., Dotted for Rookie, Dashed for Dynasty, Solid for Current).

**Step 2: Call Plot Function in History Mode**
- Generate and save `nba_defense_curry_history.png`.

**Step 3: Output CSV Summary**
- Save `nba_defense_curry_history.csv` with the raw metrics and scores.

### Task 4: Run and Verify

**Step 1: Run History Mode**
- Command: `python nba_defense_mvp.py --history`
- Expected: Script runs for 3 seasons (takes time), prints progress, and generates the PNG/CSV.

**Step 2: Verify Output**
- Check `nba_defense_curry_history.png`.
- Check `nba_defense_curry_history.csv`.

