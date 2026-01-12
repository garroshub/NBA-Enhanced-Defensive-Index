# Historical Data Verification Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Verify availability of NBA API data for Stephen Curry's career analysis (2009-10, 2014-15, 2016-17).

**Architecture:** A single Python script `test_history.py` that queries specific endpoints for historical seasons and prints availability status.

**Tech Stack:** Python, `nba_api`, `pandas`.

### Task 1: Create and Run Verification Script

**Files:**
- Create: `test_history.py`

**Step 1: Create `test_history.py`**

Create the file with the following content:

```python
import sys
import io
import time
import pandas as pd
from nba_api.stats.endpoints import (
    leaguedashptdefend,
    leaguehustlestatsplayer,
    leaguedashplayerstats,
    leagueseasonmatchups,
)

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SEASONS_TO_CHECK = ["2009-10", "2014-15", "2016-17"]

def check_endpoint(name, func, season, **kwargs):
    print(f"Checking {name} for {season}...", end=" ", flush=True)
    try:
        # Add timeout to avoid hanging
        kwargs['timeout'] = 30
        resp = func(season=season, **kwargs)
        df = resp.get_data_frames()[0]
        if not df.empty:
            print(f"✅ OK ({len(df)} rows)")
            return True
        else:
            print("❌ Empty")
            return False
    except Exception as e:
        print(f"❌ Error: {str(e)[:50]}...")
        return False
    finally:
        time.sleep(1) # Be nice to API

print("=== NBA API Historical Data Availability Check ===")

results = []

for season in SEASONS_TO_CHECK:
    print(f"\n--- Season: {season} ---")
    
    # 1. D4/D5: Player Stats (Base) - Should be available
    r_d4 = check_endpoint(
        "D4 (Base Stats)", 
        leaguedashplayerstats.LeagueDashPlayerStats, 
        season, 
        per_mode_detailed="PerGame"
    )
    
    # 2. D1/D2: Tracking (PtDefend) - Likely starts 2013-14
    r_d1 = check_endpoint(
        "D1/D2 (Tracking)", 
        leaguedashptdefend.LeagueDashPtDefend, 
        season, 
        defense_category="Overall"
    )
    
    # 3. D3: Hustle - Likely starts 2015-16
    r_d3 = check_endpoint(
        "D3 (Hustle)", 
        leaguehustlestatsplayer.LeagueHustleStatsPlayer, 
        season, 
        per_mode_time="PerGame"
    )
    
    # 4. Matchups - Tracking dependent
    r_matchup = check_endpoint(
        "Matchups", 
        leagueseasonmatchups.LeagueSeasonMatchups, 
        season, 
        season_type_playoffs="Regular Season",
        per_mode_simple="Totals"
    )
    
    results.append({
        "Season": season,
        "D4_Base": r_d4,
        "D1_Tracking": r_d1,
        "D3_Hustle": r_d3,
        "Matchups": r_matchup
    })

print("\n=== Summary ===")
df_res = pd.DataFrame(results)
print(df_res)
```

**Step 2: Run the verification script**

Run: `python test_history.py`

**Step 3: Analyze Output**

Check the summary table printed at the end. 
- Confirm if 2009-10 has any tracking data (Expect: No).
- Confirm when Hustle stats start (Expect: 2016-17 Yes, 2014-15 Maybe No).

**Step 4: Cleanup**

Remove the test file (optional, but good practice if strictly temporary).
```bash
rm test_history.py
```
