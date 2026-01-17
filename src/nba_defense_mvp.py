"""
NBA Defensive Impact (EDI) - MVP Analysis Script
=================================================
Analyzes 5 defensive dimensions and outputs radar charts.

Usage:
    python src/nba_defense_mvp.py           # Default: 2024-25 season
    python src/nba_defense_mvp.py 2025-26   # Specify season
"""

import sys
import io
import pandas as pd
import numpy as np
import time
from pathlib import Path

import matplotlib.pyplot as plt
from scipy import stats
from nba_api.stats.endpoints import (
    leaguedashptdefend,
    leaguehustlestatsplayer,
    leaguedashplayerstats,
    leagueseasonmatchups,
    leaguedashplayerbiostats,
    commonteamroster,
    commonallplayers,
)
from nba_api.stats.static import teams
from sklearn.linear_model import LinearRegression

# Fix Windows console encoding for Chinese characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# --- Configuration ---
HISTORY_MODE = False
if len(sys.argv) > 1:
    if sys.argv[1] == "--history":
        HISTORY_MODE = True
        SEASON = "2024-25"
    else:
        SEASON = sys.argv[1]
else:
    SEASON = "2024-25"

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TARGET_PLAYERS = [
    "Bam Adebayo",
    "Rudy Gobert",
    "Jrue Holiday",
    "Victor Wembanyama",
    "Trae Young",
]

# MIN_GP 将在数据获取后动态计算 (当前赛季最大比赛场次的一半)
MIN_MIN = 15

# 贝叶斯收缩常数
BAYES_C = 60  # 收缩强度：样本量达到 C 时，数据权重 = 50%

# Sigmoid 可用性参数 (Availability Sigmoid)
# 用于惩罚低出场球员，同时防止铁人刷分
SIGMOID_G0 = 45  # Sigmoid 中点 (场次阈值)
SIGMOID_K = 0.15  # Sigmoid 斜率

# Roamer 动态权重调节系数 (Roamer Dynamic Weight Adjustment)
# 用于识别"扫荡型内线"(如 JJJ)，降低其篮板权重和外线防守权重
# Roamer_Index = BLK_per_36 / (DREB_PCT + 0.01)
# W5_Adjusted = W5_Base * D5_IMPACT * (1 - ROAMER_K * Roamer_Percentile)
ROAMER_K = 0.3  # D5 灵敏度：0.3 表示最极端的扫荡者 D5 权重降至 0.7

# Roamer 分类阈值 (Roamer Classification Threshold)
# Frontcourt 球员 Roamer_Pct >= ROAMER_THRESHOLD 时被分类为 Roamer
# [2025-01-17 优化结果: 0.15 阈值覆盖主要扫荡型内线]
ROAMER_THRESHOLD = 0.15  # 15th 百分位以上视为 Roamer

# Roamer 权重重分配系数 (Roamer Weight Redistribution)
# 将 Roamer 球员因 D5 降权而失去的权重，重新分配到其表现突出的维度
# 这体现了"职责分工"的公平性：扫荡者牺牲篮板换取协防，应获得相应补偿
# 分配策略: 30% 补给产出层 (W1+W2)，70% 补给活力层 (W3)
# [2025-01-17 优化结果: DPOY Avg Rank=2.80, Recall@30=35/50, JJJ Roamer #6]
ROAMER_WEIGHT_REDIST_OUTPUT = 0.3  # 分配给产出层 (D1/D2) 的比例
ROAMER_WEIGHT_REDIST_HUSTLE = 0.7  # 分配给活力层 (D3) 的比例

# D2 外线权重调整系数 (D2 Exterior Weight Adjustment for Roamers)
# Roamer 球员主要护筐，外线防守样本少且不代表其价值
# adjusted_ext = base_ext * (1 - D2_EXT_ROAMER_K * Roamer_Pct)
# Roamer_Pct=1.0: 外线权重从 45% 降至 22.5%
# Roamer_Pct=0.0: 外线权重保持 45%
D2_EXT_ROAMER_K = 0.5  # 最大降低 50% 的外线权重

# =============================================================================
# 协同效应加分 (Synergy Bonus for Switchable Defenders)
# =============================================================================
# 奖励"换防全能型"防守者：内线能护筐 + 外线也能防
# 典型代表: JJJ, Anthony Davis, Bam Adebayo, Wembanyama
# 设计理念: 基于 D1 (对位压制) 和 D2 (内外封锁) 的真实防守效果
#           而非 D3/D4 的"投入指标"，避免奖励像库里这种高活力但效果一般的球员
# 公式 (平方根版，削峰填谷):
#   if D1 >= T1 and D2 >= T2:
#       Synergy_Bonus = sqrt((D1 - T1) * (D2 - T2)) * SYNERGY_FACTOR * 100
# 平方根逻辑: 缓解马太效应，让"准顶级全能者"也能获得合理加分
# [2025-01-17 优化结果: D1=0.80, D2=0.75, SF=0.5 实现最佳平衡]
SYNERGY_D1_THRESHOLD = 0.80  # D1 (对位压制) 阈值，约 80th 百分位
SYNERGY_D2_THRESHOLD = 0.75  # D2 (内外封锁) 阈值，约 75th 百分位
SYNERGY_FACTOR = 0.5  # 协同因子 (削弱版: 顶级+8~12分, JJJ类+4~6分)


def sigmoid_availability(games, g0=SIGMOID_G0, k=SIGMOID_K):
    """计算可用性因子 (Sigmoid 函数)。

    - 低于 g0: 快速接近 0 (淘汰低出场球员)
    - 等于 g0: 返回 0.5 (及格线)
    - 高于 g0: 逐渐接近 1.0 (边际效用递减，防止铁人刷分)

    Args:
        games: 出场场次
        g0: Sigmoid 中点 (默认 45 场)
        k: 斜率因子 (默认 0.15)

    Returns:
        可用性因子 (0-1)
    """
    return 1.0 / (1.0 + np.exp(-k * (games - g0)))


# =============================================================================
# 角色相关性系数 (Role Relevance Coefficient)
# Guards: 纯 "G" (不含 F 或 C)
# Frontcourt: 含 "F" 或 "C" (包括 G-F, F-G, F, F-C, C-F, C)
# =============================================================================
ROLE_CONFIG = {
    "Guards": {
        "D2_EXT_WEIGHT": 0.65,  # 外线 (三分防守) 权重
        "D2_INT_WEIGHT": 0.35,  # 内线 (护筐) 权重
        "D5_IMPACT": 0.4,  # 篮板权重衰减系数
    },
    "Frontcourt": {
        "D2_EXT_WEIGHT": 0.45,  # 外线 (三分防守) 权重
        "D2_INT_WEIGHT": 0.55,  # 内线 (护筐) 权重
        "D5_IMPACT": 1.0,  # 篮板权重完整保留
    },
}


def classify_role(position):
    """
    根据位置分类球员角色
    Guards: 纯 G (不含 F 或 C)
    Frontcourt: 其他所有 (含 F 或 C)
    """
    if pd.isna(position) or position == "":
        return "Frontcourt"  # 默认前场
    pos = str(position).upper()
    # 只有纯 G 才算 Guards，其他都算 Frontcourt
    if "F" in pos or "C" in pos:
        return "Frontcourt"
    if "G" in pos:
        return "Guards"
    return "Frontcourt"  # 默认前场


def get_league_data(endpoint_func, **kwargs):
    """Helper to fetch data with retry logic"""
    try:
        # CommonTeamRoster doesn't accept 'season_type_all_star'
        if (
            "CommonTeamRoster" in endpoint_func.__name__
            or "CommonAllPlayers" in endpoint_func.__name__
        ):
            params = {
                "season": SEASON,
                "timeout": 60,
                **kwargs,
            }
        else:
            params = {
                "season": SEASON,
                "season_type_all_star": "Regular Season",
                "timeout": 60,
                **kwargs,
            }

        resp = endpoint_func(**params)
        df = resp.get_data_frames()[0]
        time.sleep(0.6)  # Reduced from 1.2 to speed up loop
        return df
    except Exception as e:
        print(f"  ERROR fetching {endpoint_func.__name__}: {e}")
        return pd.DataFrame()


def standardize_pt_defend_columns(df, fg_pct_col="D_FG_PCT", fga_col="D_FGA"):
    """
    Fix column names for LeagueDashPtDefend.
    Different categories have different column names for FG% and FGA:
    - Overall: D_FG_PCT, D_FGA
    - Less Than 6Ft: LT_06_PCT, LT_06_FGA (Often just FGA or D_FGA)
    - 3 Pointers: FG3_PCT, FG3A
    """
    if df.empty:
        return df

    # Rename PLAYER_ID
    if "CLOSE_DEF_PERSON_ID" in df.columns:
        df = df.rename(columns={"CLOSE_DEF_PERSON_ID": "PLAYER_ID"})

    # Rename FG% column to standard name
    if fg_pct_col in df.columns and fg_pct_col != "D_FG_PCT":
        df = df.rename(columns={fg_pct_col: "D_FG_PCT"})

    potential_fga_cols = ["D_FGA", "FGA", "FG3A", "FGA_LT_06", "LT_06_FGA"]
    found_col = None
    for col in potential_fga_cols:
        if col in df.columns:
            found_col = col
            break

    if found_col:
        if found_col != fga_col:
            df = df.rename(columns={found_col: fga_col})

    if "D_FGA" in df.columns and fga_col != "D_FGA":
        df = df.rename(columns={"D_FGA": fga_col})

    if "FGA" in df.columns and fga_col != "FGA":
        df = df.rename(columns={"FGA": fga_col})

    if "FG3A" in df.columns and fga_col != "FG3A":
        df = df.rename(columns={"FG3A": fga_col})

    return df


def bayesian_score(raw_pct, n, c=BAYES_C):
    """
    贝叶斯收缩公式
    raw_pct: 原始百分位 (0-1)
    n: 样本量
    c: 收缩常数
    返回: 收缩后的分数, 置信度权重
    """
    score = (n * raw_pct + c * 0.5) / (n + c)
    weight = n / (n + c)
    return score, weight


def analyze_season(target_season):
    print(f"\n=== Analyzing Season: {target_season} ===")

    # =============================================================================
    # STEP 1: FETCH DATA
    # =============================================================================

    print("1. Fetching D1: Shot Suppression (Overall)...")
    d1_df = get_league_data(
        leaguedashptdefend.LeagueDashPtDefend,
        season=target_season,
        defense_category="Overall",
    )
    d1_df = standardize_pt_defend_columns(d1_df, fg_pct_col="D_FG_PCT", fga_col="D_FGA")
    print(f"   -> {len(d1_df)} players")

    print("2. Fetching D2: Shot Profile (Rim - Less Than 6Ft)...")
    d2_rim_df = get_league_data(
        leaguedashptdefend.LeagueDashPtDefend,
        season=target_season,
        defense_category="Less Than 6Ft",
    )
    # FIX: Column is 'LT_06_PCT' not 'D_FG_PCT'
    d2_rim_df = standardize_pt_defend_columns(
        d2_rim_df, fg_pct_col="LT_06_PCT", fga_col="LT_06_FGA"
    )
    print(f"   -> {len(d2_rim_df)} players")

    print("3. Fetching D2: Shot Profile (3PT)...")
    d2_3pt_df = get_league_data(
        leaguedashptdefend.LeagueDashPtDefend,
        season=target_season,
        defense_category="3 Pointers",
    )
    # FIX: Column is 'FG3_PCT' not 'D_FG_PCT'
    d2_3pt_df = standardize_pt_defend_columns(
        d2_3pt_df, fg_pct_col="FG3_PCT", fga_col="FG3A"
    )
    print(f"   -> {len(d2_3pt_df)} players")

    print("4. Fetching D3: Hustle Stats...")
    d3_df = get_league_data(
        leaguehustlestatsplayer.LeagueHustleStatsPlayer,
        season=target_season,
        per_mode_time="PerGame",
    )
    print(f"   -> {len(d3_df)} players")

    print("5. Fetching D4: Player Stats (STL, BLK, PF)...")
    d4_df = get_league_data(
        leaguedashplayerstats.LeagueDashPlayerStats,
        season=target_season,
        per_mode_detailed="PerGame",
    )
    print(f"   -> {len(d4_df)} players")

    print("6. Fetching D5: Advanced Stats (DREB_PCT)...")
    d5_adv_df = get_league_data(
        leaguedashplayerstats.LeagueDashPlayerStats,
        season=target_season,
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Advanced",
    )
    print(f"   -> {len(d5_adv_df)} players")

    # Fetch LeagueSeasonMatchups for Matchup Difficulty calculation
    print("7. Fetching LeagueSeasonMatchups for Matchup Difficulty...")
    matchup_df = pd.DataFrame()
    try:
        matchup_resp = leagueseasonmatchups.LeagueSeasonMatchups(
            season=target_season,
            season_type_playoffs="Regular Season",
            per_mode_simple="Totals",
            timeout=120,
        )
        matchup_df = matchup_resp.get_data_frames()[0]
        print(f"   -> {len(matchup_df)} matchup records")
        time.sleep(0.6)
    except Exception as e:
        print(f"   WARNING: Could not fetch matchups: {e}")

    # =============================================================================
    # STEP 2: DATA PROCESSING
    # =============================================================================
    print("\nProcessing Data...")

    if d4_df.empty:
        print("CRITICAL ERROR: Failed to fetch base player stats. Exiting.")
        return pd.DataFrame()

    # Base DataFrame with eligible players

    # 动态计算 MIN_GP：当前赛季最大比赛场次的一半
    MAX_GP = d4_df["GP"].max()
    MIN_GP = MAX_GP // 2
    print(f"\n   当前赛季最大比赛场次: {MAX_GP}, 入围门槛: GP >= {MIN_GP}")

    base_df = d4_df[(d4_df["GP"] >= MIN_GP) & (d4_df["MIN"] >= MIN_MIN)].copy()[
        ["PLAYER_ID", "PLAYER_NAME", "GP", "MIN", "PF", "STL", "BLK"]
    ]

    # Fetch Position Data
    print("7. Fetching Player Positions...")
    try:
        # Use commonteamroster iterated over teams as fallback if biostats fails or returns incomplete data
        # But first try biostats as it was in original code (commented out but then imported)

        # Original code used leaguedashplayerbiostats at line 336
        bio_df = get_league_data(
            leaguedashplayerbiostats.LeagueDashPlayerBioStats, season=target_season
        )

        # Original code also had a fallback using commonteamroster
        nba_teams = teams.get_teams()
        all_rosters = []

        print("   Fetching rosters for position data (30 teams)...")
        for t in nba_teams:
            try:
                roster = get_league_data(
                    commonteamroster.CommonTeamRoster,
                    season=target_season,
                    team_id=t["id"],
                )
                if not roster.empty:
                    all_rosters.append(roster[["PLAYER_ID", "POSITION"]])
            except:
                pass

        if all_rosters:
            roster_df = pd.concat(all_rosters)
            # Merge position
            base_df = base_df.merge(roster_df, on="PLAYER_ID", how="left")
            base_df["PLAYER_POSITION"] = base_df["POSITION"]  # Unify column name
        else:
            base_df["PLAYER_POSITION"] = "Unknown"

    except Exception as e:
        print(f"   WARNING: Could not fetch positions: {e}")
        base_df["PLAYER_POSITION"] = "Unknown"

    # Merge DREB_PCT from advanced stats
    if not d5_adv_df.empty and "DREB_PCT" in d5_adv_df.columns:
        adv_cols = d5_adv_df[["PLAYER_ID", "DREB_PCT"]]
        base_df = base_df.merge(adv_cols, on="PLAYER_ID", how="left")
        print("   D5 Advanced data merged: OK")
    else:
        print("   WARNING: DREB_PCT data missing, using fallback")
        base_df["DREB_PCT"] = np.nan

    print(f"   -> {len(base_df)} eligible players (GP>={MIN_GP}, MIN>={MIN_MIN})")

    # =============================================================================
    # D1-D5 分数计算 (贝叶斯收缩) + 权重 w_k (贝叶斯置信度)
    # =============================================================================

    C = BAYES_C

    # --- D4: Defensive IQ (防守球商) ---
    base_df["Stocks"] = base_df["STL"] + base_df["BLK"]
    base_df["D4_Ratio"] = base_df["Stocks"] / (base_df["PF"] + 1)
    base_df["D4_Raw"] = base_df["D4_Ratio"].rank(pct=True)  # 原始百分位
    base_df["D4_N"] = base_df["MIN"] * base_df["GP"]  # 样本量 = 总上场分钟数
    d4_result = base_df.apply(lambda r: bayesian_score(r["D4_Raw"], r["D4_N"]), axis=1)
    base_df["D4_Score"] = d4_result.apply(lambda x: x[0])
    base_df["W4"] = d4_result.apply(lambda x: x[1])
    print("   D4 (Defensive IQ): OK")

    # =============================================================================
    # Matchup Difficulty (MD) Calculation
    # =============================================================================
    print("   Calculating Matchup Difficulty (MD)...")
    md_df = pd.DataFrame()

    if not matchup_df.empty:
        # Step 1: Calculate each offensive player's scoring ability
        off_agg = (
            matchup_df.groupby("OFF_PLAYER_ID")
            .agg({"PLAYER_PTS": "sum", "PARTIAL_POSS": "sum"})
            .reset_index()
        )
        off_agg = off_agg[off_agg["PARTIAL_POSS"] >= 50]  # Min 50 possessions
        off_agg["OFF_PTS_PER_100"] = (
            off_agg["PLAYER_PTS"] / off_agg["PARTIAL_POSS"]
        ) * 100
        off_ability = off_agg.set_index("OFF_PLAYER_ID")["OFF_PTS_PER_100"].to_dict()

        # Step 2: Calculate each defender's Matchup Difficulty
        def calc_defender_md(group):
            total_poss = 0
            weighted_sum = 0
            for _, row in group.iterrows():
                off_id = row["OFF_PLAYER_ID"]
                poss = row["PARTIAL_POSS"]
                if off_id in off_ability and poss > 0:
                    weighted_sum += off_ability[off_id] * poss
                    total_poss += poss
            if total_poss >= 50:  # Min 50 possessions defended
                return weighted_sum / total_poss
            return np.nan

        def_groups = matchup_df.groupby("DEF_PLAYER_ID")
        md_values = def_groups.apply(calc_defender_md)
        md_df = pd.DataFrame(
            {"PLAYER_ID": md_values.index, "MATCHUP_DIFFICULTY": md_values.values}
        )
        md_df = md_df.dropna()

        # Calculate MD percentile and Z-score for adjustment
        md_mean = md_df["MATCHUP_DIFFICULTY"].mean()
        md_std = md_df["MATCHUP_DIFFICULTY"].std()
        md_df["MD_Zscore"] = (md_df["MATCHUP_DIFFICULTY"] - md_mean) / md_std
        md_df["MD_Percentile"] = md_df["MATCHUP_DIFFICULTY"].rank(pct=True)

        print(f"   -> MD calculated for {len(md_df)} defenders")
        print(f"   -> League avg MD: {md_mean:.2f}, Std: {md_std:.2f}")
    else:
        print("   WARNING: No matchup data available for MD calculation")
        # Define defaults for missing MD
        md_mean = 24.0

    # Merge MD into base_df
    if not md_df.empty:
        base_df = base_df.merge(
            md_df[["PLAYER_ID", "MATCHUP_DIFFICULTY", "MD_Zscore", "MD_Percentile"]],
            on="PLAYER_ID",
            how="left",
        )
        # Fill missing MD with league average (neutral adjustment)
        base_df["MD_Zscore"] = base_df["MD_Zscore"].fillna(0)
        base_df["MD_Percentile"] = base_df["MD_Percentile"].fillna(0.5)
        base_df["MATCHUP_DIFFICULTY"] = base_df["MATCHUP_DIFFICULTY"].fillna(
            md_mean if not md_df.empty else 24.0
        )
        print("   MD merged into base_df: OK")
    else:
        base_df["MD_Zscore"] = 0
        base_df["MD_Percentile"] = 0.5
        base_df["MATCHUP_DIFFICULTY"] = 24.0  # Default league average
        print("   WARNING: MD data unavailable, using default values")

    # --- D1: Shot Suppression (Value Added + Matchup Difficulty Adjustment) ---
    if not d1_df.empty and "D_FG_PCT" in d1_df.columns and "PLAYER_ID" in d1_df.columns:
        d1_min = d1_df[
            ["PLAYER_ID", "D_FG_PCT", "D_FGA", "PCT_PLUSMINUS", "NORMAL_FG_PCT"]
        ]
        base_df = base_df.merge(d1_min, on="PLAYER_ID", how="left")
        base_df.loc[base_df["D_FGA"] < 5, "D_FG_PCT"] = np.nan
        base_df.loc[base_df["D_FGA"] < 5, "PCT_PLUSMINUS"] = np.nan

        # MD adjustment: subtract expected impact from PLUSMINUS
        # Higher MD (defending better scorers) allows higher PLUSMINUS
        # Each +1 std MD allows +1.5% higher opponent FG%
        MD_K = 0.018  # MD adjustment coefficient (1.8% per std)
        base_df["PCT_PLUSMINUS_ADJ"] = base_df["PCT_PLUSMINUS"] - (
            MD_K * base_df["MD_Zscore"]
        )

        # Use adjusted PCT_PLUSMINUS for ranking
        base_df["D1_Raw"] = 1 - base_df["PCT_PLUSMINUS_ADJ"].rank(pct=True)
        base_df["D1_N"] = base_df["D_FGA"].fillna(0) * base_df["GP"]  # 样本量

        d1_result = base_df.apply(
            lambda r: bayesian_score(
                r["D1_Raw"] if pd.notna(r["D1_Raw"]) else 0.5, r["D1_N"]
            ),
            axis=1,
        )
        base_df["D1_Score"] = d1_result.apply(lambda x: x[0])
        base_df["W1"] = d1_result.apply(lambda x: x[1])
        print("   D1 (Suppression - Value Added + MD Adjustment): OK")
    else:
        print("   WARNING: D1 data missing")
        base_df["D1_Score"] = 0.5
        base_df["W1"] = 0.0

    # --- D2: Shot Profile (Rim + 3PT) - Value Added + MD Adjustment ---
    # MD adjustment: subtract expected impact (same logic as D1)
    MD_K = 0.015  # MD adjustment coefficient (1.5% per std)

    # Rim (护筐) - 使用 PLUSMINUS + MD调整
    if (
        not d2_rim_df.empty
        and "D_FG_PCT" in d2_rim_df.columns
        and "PLAYER_ID" in d2_rim_df.columns
    ):
        d2_rim = d2_rim_df[["PLAYER_ID", "D_FG_PCT", "LT_06_FGA", "PLUSMINUS"]].rename(
            columns={
                "D_FG_PCT": "Rim_DFG",
                "LT_06_FGA": "Rim_FGA",
                "PLUSMINUS": "Rim_PLUSMINUS",
            }
        )
        base_df = base_df.merge(d2_rim, on="PLAYER_ID", how="left")
        # MD Adjustment for Rim: subtract expected impact
        base_df["Rim_PLUSMINUS_ADJ"] = base_df["Rim_PLUSMINUS"] - (
            MD_K * base_df["MD_Zscore"]
        )
        # Value Added: PLUSMINUS 越负越好 (使用调整后的值)
        base_df["Rim_Raw"] = 1 - base_df["Rim_PLUSMINUS_ADJ"].rank(pct=True)
        print("   D2 (Rim - Value Added + MD): OK")
    else:
        print(
            "   WARNING: D2 Rim data missing - columns:",
            d2_rim_df.columns.tolist() if not d2_rim_df.empty else "EMPTY",
        )
        base_df["Rim_Raw"] = 0.5
        base_df["Rim_FGA"] = 0
        base_df["Rim_PLUSMINUS"] = np.nan
        base_df["Rim_PLUSMINUS_ADJ"] = np.nan

    # 3PT (三分) - 使用 PLUSMINUS + MD调整
    if (
        not d2_3pt_df.empty
        and "D_FG_PCT" in d2_3pt_df.columns
        and "PLAYER_ID" in d2_3pt_df.columns
    ):
        d2_3pt = d2_3pt_df[["PLAYER_ID", "D_FG_PCT", "FG3A", "PLUSMINUS"]].rename(
            columns={
                "D_FG_PCT": "3PT_DFG",
                "FG3A": "FG3_FGA",
                "PLUSMINUS": "3PT_PLUSMINUS",
            }
        )
        base_df = base_df.merge(d2_3pt, on="PLAYER_ID", how="left")
        # MD Adjustment for 3PT: subtract expected impact
        base_df["3PT_PLUSMINUS_ADJ"] = base_df["3PT_PLUSMINUS"] - (
            MD_K * base_df["MD_Zscore"]
        )
        # Value Added: PLUSMINUS 越负越好 (使用调整后的值)
        base_df["3PT_Raw"] = 1 - base_df["3PT_PLUSMINUS_ADJ"].rank(pct=True)
        print("   D2 (3PT - Value Added + MD): OK")
    else:
        print(
            "   WARNING: D2 3PT data missing - columns:",
            d2_3pt_df.columns.tolist() if not d2_3pt_df.empty else "EMPTY",
        )
        base_df["3PT_Raw"] = 0.5
        base_df["FG3_FGA"] = 0
        base_df["3PT_PLUSMINUS"] = np.nan
        base_df["3PT_PLUSMINUS_ADJ"] = np.nan

    # Combined D2: 基于角色的内外线权重
    rim_raw = base_df["Rim_Raw"].fillna(0.5)
    pt3_raw = base_df["3PT_Raw"].fillna(0.5)

    # 添加角色分类
    base_df["ROLE"] = base_df["PLAYER_POSITION"].apply(classify_role)

    # --- 计算 Roamer_Index 和 Roamer_Pct (用于 D2 和 D5 权重调整) ---
    # Roamer_Index = BLK_per_36 / (DREB_PCT + 0.01)
    # 高值 = 扫荡型内线 (盖帽多，篮板少)
    base_df["BLK_per_36"] = base_df["BLK"] / base_df["MIN"] * 36
    base_df["Roamer_Index"] = base_df["BLK_per_36"] / (base_df["DREB_PCT"] + 0.01)

    # 仅对 Frontcourt 球员计算百分位排名
    frontcourt_mask = base_df["ROLE"] == "Frontcourt"
    base_df.loc[frontcourt_mask, "Roamer_Pct"] = base_df.loc[
        frontcourt_mask, "Roamer_Index"
    ].rank(pct=True)
    base_df["Roamer_Pct"] = base_df["Roamer_Pct"].fillna(0)  # Guards 不受影响

    def calc_d2_raw(row):
        role = row["ROLE"]
        config = ROLE_CONFIG[role]
        base_int = config["D2_INT_WEIGHT"]
        base_ext = config["D2_EXT_WEIGHT"]

        # Roamer 动态外线权重调整 (仅 Frontcourt)
        # adjusted_ext = base_ext * (1 - D2_EXT_ROAMER_K * Roamer_Pct)
        roamer_pct = row.get("Roamer_Pct", 0) if role == "Frontcourt" else 0
        adjusted_ext = base_ext * (1 - D2_EXT_ROAMER_K * roamer_pct)
        adjusted_int = 1 - adjusted_ext

        return (rim_raw[row.name] * adjusted_int) + (pt3_raw[row.name] * adjusted_ext)

    base_df["D2_Raw"] = base_df.apply(calc_d2_raw, axis=1)

    # 样本量 = 护筐防守次数 + 三分防守次数 (基于角色加权，含 Roamer 调整)
    rim_fga = base_df["Rim_FGA"].fillna(0) * base_df["GP"]
    fg3_fga = base_df["FG3_FGA"].fillna(0) * base_df["GP"]

    def calc_d2_n(row):
        role = row["ROLE"]
        config = ROLE_CONFIG[role]
        base_int = config["D2_INT_WEIGHT"]
        base_ext = config["D2_EXT_WEIGHT"]

        # Roamer 动态外线权重调整 (与 D2_Raw 一致)
        roamer_pct = row.get("Roamer_Pct", 0) if role == "Frontcourt" else 0
        adjusted_ext = base_ext * (1 - D2_EXT_ROAMER_K * roamer_pct)
        adjusted_int = 1 - adjusted_ext

        return (rim_fga[row.name] * adjusted_int) + (fg3_fga[row.name] * adjusted_ext)

    base_df["D2_N"] = base_df.apply(calc_d2_n, axis=1)
    print("   D2 (Zone Defense - Roamer-adjusted): OK")

    d2_result = base_df.apply(lambda r: bayesian_score(r["D2_Raw"], r["D2_N"]), axis=1)
    base_df["D2_Score"] = d2_result.apply(lambda x: x[0])
    base_df["W2"] = d2_result.apply(lambda x: x[1])

    # --- D3: Hustle Index ---
    if not d3_df.empty:
        hustle_cols = d3_df[
            ["PLAYER_ID", "DEFLECTIONS", "CHARGES_DRAWN", "CONTESTED_SHOTS"]
        ]
        base_df = base_df.merge(hustle_cols, on="PLAYER_ID", how="left")

        defl = base_df["DEFLECTIONS"].fillna(0)
        chrg = base_df["CHARGES_DRAWN"].fillna(0)
        cont = base_df["CONTESTED_SHOTS"].fillna(0)

        base_df["Z_Defl"] = stats.zscore(defl)
        base_df["Z_Chrg"] = stats.zscore(chrg)
        base_df["Z_Cont"] = stats.zscore(cont)

        base_df["Hustle_Raw"] = (
            base_df["Z_Defl"] + (base_df["Z_Chrg"] * 2) + base_df["Z_Cont"]
        )
        base_df["D3_Raw"] = base_df["Hustle_Raw"].rank(pct=True)
        base_df["D3_N"] = base_df["MIN"] * base_df["GP"]

        d3_result = base_df.apply(
            lambda r: bayesian_score(r["D3_Raw"], r["D3_N"]), axis=1
        )
        base_df["D3_Score"] = d3_result.apply(lambda x: x[0])
        base_df["W3"] = d3_result.apply(lambda x: x[1])
        print("   D3 (Hustle): OK")
    else:
        print("   WARNING: D3 Hustle data missing")
        base_df["D3_Score"] = 0.5
        base_df["W3"] = 0.0

    # --- D5: Anchor / Rebound Protection (篮板保护) ---
    base_df["D5_Raw"] = base_df["DREB_PCT"].rank(pct=True)
    base_df["D5_N"] = base_df["MIN"] * base_df["GP"]

    d5_result = base_df.apply(
        lambda r: bayesian_score(
            r["D5_Raw"] if pd.notna(r["D5_Raw"]) else 0.5, r["D5_N"]
        ),
        axis=1,
    )
    base_df["D5_Score"] = d5_result.apply(lambda x: x[0])
    base_df["W5_Base"] = d5_result.apply(lambda x: x[1])

    # 应用角色相关性系数 (Role Relevance Coefficient)
    base_df["W5"] = base_df.apply(
        lambda r: r["W5_Base"] * ROLE_CONFIG[r["ROLE"]]["D5_IMPACT"], axis=1
    )

    # --- Roamer 动态 D5 权重调节 ---
    # Roamer_Pct 已在 D2 计算前计算完成
    # 应用动态调整: W5 = W5 * (1 - ROAMER_K * Roamer_Pct)
    # 先保存调整前的 W5 用于计算权重损失
    base_df["W5_Before_Roamer"] = base_df["W5"].copy()
    base_df["W5"] = base_df["W5"] * (1 - ROAMER_K * base_df["Roamer_Pct"])

    # --- Roamer 权重重分配 (Weight Redistribution) ---
    # 将 Roamer 球员因 D5 降权而失去的权重，重新分配到产出层和活力层
    # 这体现了"职责分工"的公平性：扫荡者牺牲篮板换取协防，应获得相应补偿
    base_df["W5_Lost"] = base_df["W5_Before_Roamer"] - base_df["W5"]

    # 分配给产出层 (W1/W2): 按 W1:W2 的原有比例分配
    w1_w2_total = base_df["W1"] + base_df["W2"] + 1e-6
    base_df["W1"] = base_df["W1"] + base_df["W5_Lost"] * ROAMER_WEIGHT_REDIST_OUTPUT * (
        base_df["W1"] / w1_w2_total
    )
    base_df["W2"] = base_df["W2"] + base_df["W5_Lost"] * ROAMER_WEIGHT_REDIST_OUTPUT * (
        base_df["W2"] / w1_w2_total
    )

    # 分配给活力层 (W3): 全部分配给 D3
    base_df["W3"] = base_df["W3"] + base_df["W5_Lost"] * ROAMER_WEIGHT_REDIST_HUSTLE

    # 打印权重重分配统计
    roamer_beneficiaries = (base_df["W5_Lost"] > 0.05).sum()
    if roamer_beneficiaries > 0:
        print(f"   Roamer 权重重分配: {roamer_beneficiaries} 名球员受益")
        top_beneficiaries = base_df.nlargest(3, "W5_Lost")[["PLAYER_NAME", "W5_Lost"]]
        for _, row in top_beneficiaries.iterrows():
            print(
                f"      {row['PLAYER_NAME']}: W5 损失 {row['W5_Lost']:.3f} -> 重分配到 W1/W2/W3"
            )

    print(
        "   D5 (Anchor/DREB%): OK (Role-adjusted, Roamer-corrected, Weight-redistributed)"
    )

    # =============================================================================
    # 效率模型框架 (Efficiency Model Framework)
    # =============================================================================

    # Fill NaN scores with 0.5 (neutral)
    score_cols = ["D1_Score", "D2_Score", "D3_Score", "D4_Score", "D5_Score"]
    weight_cols = ["W1", "W2", "W3", "W4", "W5"]

    for col in score_cols:
        base_df[col] = base_df[col].fillna(0.5)

    for col in weight_cols:
        base_df[col] = base_df[col].fillna(0)

    # Step 1: 计算实际产出 (D1 + D2 的加权平均)
    base_df["Actual_Output"] = (
        base_df["D1_Score"] * base_df["W1"] + base_df["D2_Score"] * base_df["W2"]
    ) / (base_df["W1"] + base_df["W2"] + 1e-6)

    # Step 2: 计算投入分 (D3 + D4 的加权平均)
    base_df["Input_Score"] = (
        base_df["D3_Score"] * base_df["W3"] + base_df["D4_Score"] * base_df["W4"]
    ) / (base_df["W3"] + base_df["W4"] + 1e-6)

    # Step 3: 用线性回归建立 投入 → 预期产出 的模型
    valid_mask = (
        (base_df["W1"] > 0.1)
        & (base_df["W2"] > 0.1)
        & (base_df["W3"] > 0.1)
        & (base_df["W4"] > 0.1)
    )
    if valid_mask.sum() > 10:
        X_train = base_df.loc[valid_mask, "Input_Score"].values.reshape(-1, 1)
        y_train = base_df.loc[valid_mask, "Actual_Output"].values

        reg_model = LinearRegression()
        reg_model.fit(X_train, y_train)

        # 预测所有球员的预期产出
        base_df["Expected_Output"] = reg_model.predict(
            base_df["Input_Score"].values.reshape(-1, 1)
        )

        print(
            f"   回归模型: Expected_Output = {reg_model.intercept_:.4f} + {reg_model.coef_[0]:.4f} * Input_Score"
        )
        print(f"   R² = {reg_model.score(X_train, y_train):.4f}")
    else:
        print(
            "   WARNING: Not enough valid data for regression model. Using Actual as Expected."
        )
        base_df["Expected_Output"] = base_df["Actual_Output"]
        # Dummy model for Hansen logic
        reg_model = LinearRegression()
        reg_model.intercept_ = 0
        reg_model.coef_ = np.array([1.0])

    # Step 4: 计算效率系数
    base_df["Efficiency"] = base_df["Actual_Output"] / (
        base_df["Expected_Output"] + 1e-6
    )
    base_df["Efficiency"] = base_df["Efficiency"].clip(0.5, 1.5)

    # Step 5: 计算新的 EDI
    output_weighted = (
        base_df["Actual_Output"]
        * base_df["Efficiency"]
        * (base_df["W1"] + base_df["W2"])
    )
    input_weighted = base_df["Input_Score"] * (base_df["W3"] + base_df["W4"])
    d5_weighted = base_df["D5_Score"] * base_df["W5"]

    total_weight = (
        base_df["W1"] + base_df["W2"] + base_df["W3"] + base_df["W4"] + base_df["W5"]
    )

    base_df["EDI_Total"] = np.where(
        total_weight > 0,
        (output_weighted + input_weighted + d5_weighted) / total_weight * 100,
        50.0,
    )

    # Step 6: 应用 Sigmoid 可用性调整
    # 公式: EDI_Final = EDI_Raw * Sigmoid(GP)
    # 这会惩罚低出场球员，同时防止铁人刷分 (边际效用递减)
    base_df["Sigmoid_Factor"] = base_df["GP"].apply(sigmoid_availability)
    base_df["EDI_Total"] = base_df["EDI_Total"] * base_df["Sigmoid_Factor"]

    # Step 7: 协同效应加分 (Synergy Bonus for Switchable Defenders)
    # 奖励同时具备高 D1 (对位压制) 和高 D2 (内外封锁) 的"换防全能型"防守者
    # 公式 (平方根版): sqrt((D1 - T1) * (D2 - T2)) * Factor * 100
    def calc_synergy_bonus(row):
        d1, d2 = row["D1_Score"], row["D2_Score"]
        if d1 >= SYNERGY_D1_THRESHOLD and d2 >= SYNERGY_D2_THRESHOLD:
            # 平方根公式：削峰填谷，缓解马太效应
            raw_synergy = (d1 - SYNERGY_D1_THRESHOLD) * (d2 - SYNERGY_D2_THRESHOLD)
            return np.sqrt(raw_synergy) * SYNERGY_FACTOR * 100
        return 0.0

    base_df["Synergy_Bonus"] = base_df.apply(calc_synergy_bonus, axis=1)
    base_df["EDI_Total"] = base_df["EDI_Total"] + base_df["Synergy_Bonus"]

    synergy_count = (base_df["Synergy_Bonus"] > 0).sum()
    if synergy_count > 0:
        print(f"   协同效应加分: {synergy_count} 名球员获得加分")
        top_synergy = base_df.nlargest(3, "Synergy_Bonus")[
            ["PLAYER_NAME", "Synergy_Bonus"]
        ]
        for _, row in top_synergy.iterrows():
            print(f"      {row['PLAYER_NAME']}: +{row['Synergy_Bonus']:.2f}")

    # 计算效率残差 (用于分析)
    base_df["Efficiency_Residual"] = (
        base_df["Actual_Output"] - base_df["Expected_Output"]
    )

    print("   效率模型计算完成:")
    print(
        f"   -> 高效球员 (Efficiency > 1.1): {len(base_df[base_df['Efficiency'] > 1.1])}"
    )
    print(
        f"   -> 低效球员 (Efficiency < 0.9): {len(base_df[base_df['Efficiency'] < 0.9])}"
    )

    print(
        f"\n   角色分布: Guards={len(base_df[base_df['ROLE'] == 'Guards'])}, Frontcourt={len(base_df[base_df['ROLE'] == 'Frontcourt'])}"
    )

    # =============================================================================
    # 2025-26 赛季特殊处理: 为 Hansen Yang 单独计算 EDI (忽略 GP 限制)
    # =============================================================================
    if target_season == "2025-26":
        hansen_mask = d4_df["PLAYER_NAME"].str.contains("Hansen", case=False, na=False)
        if (
            hansen_mask.any()
            and not base_df["PLAYER_NAME"]
            .str.contains("Hansen", case=False, na=False)
            .any()
        ):
            print("\n   [特殊处理] 为 Hansen Yang 计算 EDI (忽略 GP 限制)...")

            # 获取 Hansen 的基础数据
            hansen_base = (
                d4_df[hansen_mask]
                .copy()[["PLAYER_ID", "PLAYER_NAME", "GP", "MIN", "PF", "STL", "BLK"]]
                .iloc[0:1]
            )

            # 获取位置信息
            hansen_id = hansen_base["PLAYER_ID"].values[0]
            hansen_base["PLAYER_POSITION"] = "F"  # 前锋
            hansen_base["ROLE"] = "Frontcourt"

            # D4: Defensive IQ
            hansen_base["Stocks"] = hansen_base["STL"] + hansen_base["BLK"]
            hansen_base["D4_Ratio"] = hansen_base["Stocks"] / (hansen_base["PF"] + 1)
            # 使用联盟排名计算百分位
            all_d4_ratios = d4_df["STL"] + d4_df["BLK"]
            all_d4_ratios = all_d4_ratios / (d4_df["PF"] + 1)
            hansen_d4_raw = (all_d4_ratios < hansen_base["D4_Ratio"].values[0]).mean()
            hansen_d4_n = hansen_base["MIN"].values[0] * hansen_base["GP"].values[0]
            hansen_base["D4_Score"] = (hansen_d4_n * hansen_d4_raw + C * 0.5) / (
                hansen_d4_n + C
            )
            hansen_base["W4"] = hansen_d4_n / (hansen_d4_n + C)

            # D1: Shot Suppression (Value Added + MD Adjustment)
            if not d1_df.empty and hansen_id in d1_df["PLAYER_ID"].values:
                hansen_d1 = d1_df[d1_df["PLAYER_ID"] == hansen_id].iloc[0]
                hansen_dfg = hansen_d1["D_FG_PCT"]
                hansen_dfga = hansen_d1["D_FGA"]
                hansen_pct_plusminus = hansen_d1["PCT_PLUSMINUS"]

                # Get Hansen's MD (if available)
                hansen_md_zscore = 0.0  # Default neutral
                if not md_df.empty and hansen_id in md_df["PLAYER_ID"].values:
                    hansen_md_row = md_df[md_df["PLAYER_ID"] == hansen_id].iloc[0]
                    hansen_md_zscore = hansen_md_row["MD_Zscore"]
                    hansen_base["MATCHUP_DIFFICULTY"] = hansen_md_row[
                        "MATCHUP_DIFFICULTY"
                    ]
                    hansen_base["MD_Zscore"] = hansen_md_zscore
                    hansen_base["MD_Percentile"] = hansen_md_row["MD_Percentile"]
                else:
                    hansen_base["MATCHUP_DIFFICULTY"] = 24.0
                    hansen_base["MD_Zscore"] = 0.0
                    hansen_base["MD_Percentile"] = 0.5

                # Apply MD adjustment: Adjusted_VA = PCT_PLUSMINUS * (1 + k * MD_Zscore)
                MD_K = 0.3
                hansen_pct_plusminus_adj = hansen_pct_plusminus * (
                    1 + MD_K * hansen_md_zscore
                )

                # Use adjusted PCT_PLUSMINUS to calculate percentile
                # Calculate adjusted VA for all players in d1_df for comparison
                if not md_df.empty:
                    d1_with_md = d1_df.merge(
                        md_df[["PLAYER_ID", "MD_Zscore"]], on="PLAYER_ID", how="left"
                    )
                    d1_with_md["MD_Zscore"] = d1_with_md["MD_Zscore"].fillna(0)
                    d1_with_md["PCT_PLUSMINUS_ADJ"] = d1_with_md["PCT_PLUSMINUS"] * (
                        1 + MD_K * d1_with_md["MD_Zscore"]
                    )
                    hansen_d1_raw = (
                        1
                        - (
                            d1_with_md["PCT_PLUSMINUS_ADJ"] < hansen_pct_plusminus_adj
                        ).mean()
                    )
                else:
                    hansen_d1_raw = (
                        1 - (d1_df["PCT_PLUSMINUS"] < hansen_pct_plusminus).mean()
                    )

                hansen_d1_n = hansen_dfga * hansen_base["GP"].values[0]
                hansen_base["D1_Score"] = (hansen_d1_n * hansen_d1_raw + C * 0.5) / (
                    hansen_d1_n + C
                )
                hansen_base["W1"] = hansen_d1_n / (hansen_d1_n + C)
                hansen_base["D_FG_PCT"] = hansen_dfg
                hansen_base["PCT_PLUSMINUS"] = hansen_pct_plusminus
                hansen_base["PCT_PLUSMINUS_ADJ"] = hansen_pct_plusminus_adj
            else:
                hansen_base["D1_Score"] = 0.5
                hansen_base["W1"] = 0.0
                hansen_base["D_FG_PCT"] = np.nan
                hansen_base["PCT_PLUSMINUS"] = np.nan
                hansen_base["PCT_PLUSMINUS_ADJ"] = np.nan
                hansen_base["MATCHUP_DIFFICULTY"] = 24.0
                hansen_base["MD_Zscore"] = 0.0
                hansen_base["MD_Percentile"] = 0.5

            # D2: Rim + 3PT (Frontcourt weights: 内线60% / 外线40%) - Value Added
            hansen_rim_raw, hansen_3pt_raw = 0.5, 0.5
            hansen_rim_fga, hansen_3pt_fga = 0, 0

            if not d2_rim_df.empty and hansen_id in d2_rim_df["PLAYER_ID"].values:
                hansen_rim = d2_rim_df[d2_rim_df["PLAYER_ID"] == hansen_id].iloc[0]
                hansen_rim_dfg = hansen_rim["D_FG_PCT"]
                hansen_rim_plusminus = hansen_rim["PLUSMINUS"]
                # Value Added: PLUSMINUS 越负越好
                hansen_rim_raw = (
                    1 - (d2_rim_df["PLUSMINUS"] < hansen_rim_plusminus).mean()
                )
                hansen_rim_fga = hansen_rim.get("LT_06_FGA", 0)
                hansen_base["Rim_DFG"] = hansen_rim_dfg
                hansen_base["Rim_PLUSMINUS"] = hansen_rim_plusminus

            if not d2_3pt_df.empty and hansen_id in d2_3pt_df["PLAYER_ID"].values:
                hansen_3pt = d2_3pt_df[d2_3pt_df["PLAYER_ID"] == hansen_id].iloc[0]
                hansen_3pt_dfg = hansen_3pt["D_FG_PCT"]
                hansen_3pt_plusminus = hansen_3pt["PLUSMINUS"]
                # Value Added: PLUSMINUS 越负越好
                hansen_3pt_raw = (
                    1 - (d2_3pt_df["PLUSMINUS"] < hansen_3pt_plusminus).mean()
                )
                hansen_3pt_fga = hansen_3pt.get("FG3A", 0)
                hansen_base["3PT_DFG"] = hansen_3pt_dfg
                hansen_base["3PT_PLUSMINUS"] = hansen_3pt_plusminus

            # Frontcourt: 内线60% / 外线40%
            hansen_d2_raw = hansen_rim_raw * 0.6 + hansen_3pt_raw * 0.4
            hansen_d2_n = (hansen_rim_fga * 0.6 + hansen_3pt_fga * 0.4) * hansen_base[
                "GP"
            ].values[0]
            hansen_base["D2_Score"] = (hansen_d2_n * hansen_d2_raw + C * 0.5) / (
                hansen_d2_n + C
            )
            hansen_base["W2"] = hansen_d2_n / (hansen_d2_n + C)

            # D3: Hustle Index
            if not d3_df.empty and hansen_id in d3_df["PLAYER_ID"].values:
                hansen_d3 = d3_df[d3_df["PLAYER_ID"] == hansen_id].iloc[0]
                defl = hansen_d3.get("DEFLECTIONS", 0)
                chrg = hansen_d3.get("CHARGES_DRAWN", 0)
                cont = hansen_d3.get("CONTESTED_SHOTS", 0)

                # 计算 Z-score 相对于联盟
                z_defl = (defl - d3_df["DEFLECTIONS"].mean()) / d3_df[
                    "DEFLECTIONS"
                ].std()
                z_chrg = (chrg - d3_df["CHARGES_DRAWN"].mean()) / d3_df[
                    "CHARGES_DRAWN"
                ].std()
                z_cont = (cont - d3_df["CONTESTED_SHOTS"].mean()) / d3_df[
                    "CONTESTED_SHOTS"
                ].std()
                hansen_hustle = z_defl + z_chrg * 2 + z_cont

                # 计算百分位
                all_hustle = (
                    d3_df["DEFLECTIONS"] - d3_df["DEFLECTIONS"].mean()
                ) / d3_df["DEFLECTIONS"].std()
                all_hustle += (
                    (d3_df["CHARGES_DRAWN"] - d3_df["CHARGES_DRAWN"].mean())
                    / d3_df["CHARGES_DRAWN"].std()
                    * 2
                )
                all_hustle += (
                    d3_df["CONTESTED_SHOTS"] - d3_df["CONTESTED_SHOTS"].mean()
                ) / d3_df["CONTESTED_SHOTS"].std()
                hansen_d3_raw = (all_hustle < hansen_hustle).mean()
                hansen_d3_n = hansen_base["MIN"].values[0] * hansen_base["GP"].values[0]
                hansen_base["D3_Score"] = (hansen_d3_n * hansen_d3_raw + C * 0.5) / (
                    hansen_d3_n + C
                )
                hansen_base["W3"] = hansen_d3_n / (hansen_d3_n + C)
                hansen_base["DEFLECTIONS"] = defl
            else:
                hansen_base["D3_Score"] = 0.5
                hansen_base["W3"] = 0.0

            # D5: DREB%
            if not d5_adv_df.empty and hansen_id in d5_adv_df["PLAYER_ID"].values:
                hansen_dreb = d5_adv_df[d5_adv_df["PLAYER_ID"] == hansen_id][
                    "DREB_PCT"
                ].values[0]
                hansen_d5_raw = (d5_adv_df["DREB_PCT"] < hansen_dreb).mean()
                hansen_d5_n = hansen_base["MIN"].values[0] * hansen_base["GP"].values[0]
                hansen_base["D5_Score"] = (hansen_d5_n * hansen_d5_raw + C * 0.5) / (
                    hansen_d5_n + C
                )
                hansen_base["W5"] = hansen_d5_n / (hansen_d5_n + C)  # Frontcourt: 1.0
                hansen_base["DREB_PCT"] = hansen_dreb
            else:
                hansen_base["D5_Score"] = 0.5
                hansen_base["W5"] = 0.0
                hansen_base["DREB_PCT"] = np.nan

            # 计算 EDI_Total (使用效率模型框架)
            # Step 1: 计算实际产出 (Actual Output)
            hansen_w1 = hansen_base["W1"].values[0]
            hansen_w2 = hansen_base["W2"].values[0]
            hansen_w3 = hansen_base["W3"].values[0]
            hansen_w4 = hansen_base["W4"].values[0]
            hansen_w5 = hansen_base["W5"].values[0]

            hansen_d1 = hansen_base["D1_Score"].values[0]
            hansen_d2 = hansen_base["D2_Score"].values[0]
            hansen_d3 = hansen_base["D3_Score"].values[0]
            hansen_d4 = hansen_base["D4_Score"].values[0]
            hansen_d5 = hansen_base["D5_Score"].values[0]

            hansen_actual_output = (hansen_d1 * hansen_w1 + hansen_d2 * hansen_w2) / (
                hansen_w1 + hansen_w2 + 1e-6
            )
            hansen_base["Actual_Output"] = hansen_actual_output

            # Step 2: 计算投入分 (Input Score)
            hansen_input_score = (hansen_d3 * hansen_w3 + hansen_d4 * hansen_w4) / (
                hansen_w3 + hansen_w4 + 1e-6
            )
            hansen_base["Input_Score"] = hansen_input_score

            # Step 3: 用已拟合的回归模型预测预期产出
            if valid_mask.sum() > 10:
                hansen_expected_output = reg_model.predict([[hansen_input_score]])[0]
            else:
                hansen_expected_output = hansen_actual_output

            hansen_base["Expected_Output"] = hansen_expected_output

            # Step 4: 计算效率系数 (限制在 [0.5, 1.5] 范围)
            hansen_efficiency = hansen_actual_output / (hansen_expected_output + 1e-6)
            hansen_efficiency = np.clip(hansen_efficiency, 0.5, 1.5)
            hansen_base["Efficiency"] = hansen_efficiency

            # Step 5: 计算效率残差
            hansen_base["Efficiency_Residual"] = (
                hansen_actual_output - hansen_expected_output
            )

            # Step 6: 计算 EDI (使用效率模型公式)
            output_weighted = (
                hansen_actual_output * hansen_efficiency * (hansen_w1 + hansen_w2)
            )
            input_weighted = hansen_input_score * (hansen_w3 + hansen_w4)
            d5_weighted = hansen_d5 * hansen_w5

            total_weight = hansen_w1 + hansen_w2 + hansen_w3 + hansen_w4 + hansen_w5
            hansen_base["EDI_Total"] = (
                (output_weighted + input_weighted + d5_weighted) / total_weight * 100
                if total_weight > 0
                else 50.0
            )

            print(
                f"   Hansen 效率模型: Input={hansen_input_score:.3f}, Expected={hansen_expected_output:.3f}, Actual={hansen_actual_output:.3f}, Efficiency={hansen_efficiency:.3f}"
            )

            # 添加到 base_df
            base_df = pd.concat([base_df, hansen_base], ignore_index=True)
            print(
                f"   Hansen Yang EDI: {hansen_base['EDI_Total'].values[0]:.2f} (GP={hansen_base['GP'].values[0]})"
            )

    return base_df


# Helper to print top N with optional extra player
def print_top_n(df, title, n=5, extra_player=None, extra_label=None):
    if df.empty:
        return

    # Define column map and display columns first
    cn_col_map = {
        "PLAYER_NAME": "球员",
        "PLAYER_POSITION": "位置",
        "EDI_Total": "防守统治力",
        "D1_Score": "对位压制",
        "D2_Score": "内外封锁",
        "D3_Score": "活力指数",
        "D4_Score": "防守球商",
        "D5_Score": "篮板保护",
        "Stocks": "抢断+盖帽",
        "D4_Ratio": "球商比值",
        "DREB_PCT": "防守篮板%",
        "D_FG_PCT": "对手命中%",
        "PCT_PLUSMINUS": "对位压制差%",
        "PCT_PLUSMINUS_ADJ": "MD调整压制差%",
        "NORMAL_FG_PCT": "对手预期命中%",
        "MATCHUP_DIFFICULTY": "对位难度",
        "MD_Percentile": "对位难度%",
        "Rim_DFG": "护筐命中%",
        "Rim_PLUSMINUS": "护筐压制差%",
        "3PT_DFG": "三分命中%",
        "3PT_PLUSMINUS": "三分压制差%",
        "DEFLECTIONS": "干扰次数",
        "PF": "犯规",
        # 效率模型相关
        "Efficiency": "防守效率系数",
        "Actual_Output": "实际产出",
        "Expected_Output": "预期产出",
        "Input_Score": "投入分",
        "Efficiency_Residual": "效率残差",
    }

    display_cols = [
        "PLAYER_NAME",
        "PLAYER_POSITION",
        "EDI_Total",
        "D1_Score",
        "D2_Score",
        "D3_Score",
        "D4_Score",
        "D5_Score",
    ]

    total_count = len(df)
    print(f"\n🏆 {title} Top {n}:")
    sorted_df = df.sort_values("EDI_Total", ascending=False)
    top_df = sorted_df.head(n)

    # If extra_player specified and not in top N, add them
    if extra_player:
        extra_mask = sorted_df["PLAYER_NAME"].str.contains(
            extra_player, case=False, na=False
        )
        if extra_mask.any():
            extra_row = sorted_df[extra_mask].iloc[0:1]
            # Check if already in top N
            if (
                not top_df["PLAYER_NAME"]
                .str.contains(extra_player, case=False, na=False)
                .any()
            ):
                # Calculate rank
                extra_rank = (
                    sorted_df["EDI_Total"] > extra_row["EDI_Total"].values[0]
                ).sum() + 1
                top_df = pd.concat([top_df, extra_row])
                print(f"   (包含 {extra_player}, 排名 #{extra_rank}/{total_count})")

    # Filter display columns
    current_display_cols = [c for c in display_cols if c in df.columns]
    disp = top_df[current_display_cols].copy()
    disp = disp.rename(columns=cn_col_map)
    print(disp.round(2).to_string(index=False))
    return sorted_df  # Return for visualization


def create_individual_radar_charts(df, save_path, main_title="球员防守能力画像"):
    """
    为每个球员创建单独的雷达图，横向排列
    参考图片样式：每个球员一个子图，显示球员名和 EDI 分数
    """
    n_players = len(df)
    if n_players == 0:
        print("   WARNING: No players to plot")
        return

    # Dimension labels in Chinese
    categories = ["对位压制", "内外封锁", "活力指数", "防守球商", "篮板保护"]
    N = len(categories)

    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the loop

    # Color palette for each player (different colors)
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"]

    # Create figure with subplots (1 row, n_players columns)
    fig_width = 4 * n_players
    fig, axes = plt.subplots(
        1, n_players, figsize=(fig_width, 5), subplot_kw=dict(polar=True)
    )

    # Handle single player case
    if n_players == 1:
        axes = [axes]

    # Plot each player in their own subplot
    for idx, (_, row) in enumerate(df.iterrows()):
        ax = axes[idx]
        color = colors[idx % len(colors)]

        values = [
            row["D1_Score"],
            row["D2_Score"],
            row["D3_Score"],
            row["D4_Score"],
            row["D5_Score"],
        ]
        values += values[:1]  # Complete the loop

        # Plot the radar
        ax.plot(angles, values, "o-", linewidth=2, color=color)
        ax.fill(angles, values, alpha=0.25, color=color)

        # Set category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=9, fontweight="bold")

        # Set y-axis limits (0 to 1 for percentile scores)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.5, 1.0])
        ax.set_yticklabels(["50%", "100%"], size=8, color="gray")

        # Add gridlines
        ax.grid(True, linestyle="--", alpha=0.5)

        # Title with player name and EDI score
        ax.set_title(
            f"{row['PLAYER_NAME']}\n防守统治力: {row['EDI_Total']:.1f}",
            size=11,
            fontweight="bold",
            pad=15,
        )

    # Main title
    fig.suptitle(f"{main_title} ({SEASON})", size=16, fontweight="bold", y=1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"已保存: {save_path}")
    plt.close()


def plot_history_radar(history_df):
    """
    绘制库里职业生涯防守演变雷达图 (中文版)
    """
    import matplotlib

    matplotlib.rcParams["font.sans-serif"] = [
        "SimHei",
        "Microsoft YaHei",
        "Arial Unicode MS",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False

    if history_df.empty:
        return

    # 五维中文标签
    categories = [
        "D1: 对位压制",
        "D2: 内外封锁",
        "D3: 活力指数",
        "D4: 防守球商",
        "D5: 篮板保护",
    ]
    N = len(categories)

    # 计算角度
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # 初始化图表
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # 三个阶段的样式配置
    styles = {
        "2016-17": {
            "color": "#FFC72C",  # 勇士金色
            "linestyle": "--",
            "linewidth": 2.5,
            "label": "2016-17 巅峰王朝",
            "alpha": 0.15,
        },
        "2021-22": {
            "color": "#1D428A",  # 勇士蓝色
            "linestyle": "-.",
            "linewidth": 2.5,
            "label": "2021-22 王者归来",
            "alpha": 0.15,
        },
        "2024-25": {
            "color": "#006BB6",  # NBA蓝
            "linestyle": "-",
            "linewidth": 3,
            "label": "2024-25 老将赛季",
            "alpha": 0.2,
        },
    }

    # 绘制每个赛季
    for _, row in history_df.iterrows():
        season = row["SEASON_ID"]
        style = styles.get(
            season,
            {
                "color": "gray",
                "linestyle": "-",
                "linewidth": 1,
                "label": season,
                "alpha": 0.1,
            },
        )

        values = [
            row.get("D1_Score", 0.5),
            row.get("D2_Score", 0.5),
            row.get("D3_Score", 0.5),
            row.get("D4_Score", 0.5),
            row.get("D5_Score", 0.5),
        ]
        values = [0.5 if pd.isna(v) else v for v in values]
        values += values[:1]

        ax.plot(
            angles,
            values,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            label=style["label"],
            marker="o",
            markersize=6,
        )
        ax.fill(angles, values, color=style["color"], alpha=style["alpha"])

    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12, fontweight="bold")

    # 设置Y轴
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], size=9, color="gray")

    # 标题和图例
    plt.title(
        "斯蒂芬·库里：防守能力演变\n(巅峰王朝 vs 王者归来 vs 老将赛季)",
        size=16,
        fontweight="bold",
        y=1.1,
    )
    plt.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=11)

    # 保存
    save_path = FIGURES_DIR / "nba_defense_curry_history.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"已保存: {save_path}")
    plt.close()


if __name__ == "__main__":
    if HISTORY_MODE:
        print(f"=== NBA Defense MVP Analysis (History Mode: Curry Eras) ===")
        target_seasons = ["2016-17", "2021-22", "2024-25"]
        curry_id = 201939
        history_results = []

        for s in target_seasons:
            try:
                # Use a larger timeout or retry logic handled inside analyze_season/get_league_data
                season_df = analyze_season(s)

                if not season_df.empty:
                    # Find Curry
                    curry_data = season_df[season_df["PLAYER_ID"] == curry_id]
                    if not curry_data.empty:
                        row = curry_data.iloc[0].to_dict()
                        row["SEASON_ID"] = s
                        history_results.append(row)
                        print(
                            f"   -> Found Curry in {s}: EDI={row.get('EDI_Total', 0):.1f}"
                        )
                    else:
                        print(f"   -> WARNING: Stephen Curry not found in {s}")
                        # Append empty/default row to keep structure? Or just skip?
                        # Let's append a row with 0s/NaNs so we know we tried
                        empty_row = {col: np.nan for col in season_df.columns}
                        empty_row["PLAYER_NAME"] = "Stephen Curry"
                        empty_row["PLAYER_ID"] = curry_id
                        empty_row["SEASON_ID"] = s
                        history_results.append(empty_row)
            except Exception as e:
                print(f"   ERROR analyzing {s}: {e}")

        # Summary Table
        if history_results:
            history_df = pd.DataFrame(history_results)

            # Reorder columns for readability
            cols_order = [
                "SEASON_ID",
                "PLAYER_NAME",
                "EDI_Total",
                "D1_Score",
                "D2_Score",
                "D3_Score",
                "D4_Score",
                "D5_Score",
                "Efficiency",
                "Actual_Output",
                "Expected_Output",
            ]
            # Filter to exist cols
            cols_order = [c for c in cols_order if c in history_df.columns]

            print("\n" + "=" * 60)
            print("🏀 Stephen Curry: Career Defense Evolution (3 Eras)")
            print("=" * 60)
            print(history_df[cols_order].round(2).to_string(index=False))

            # Save
            history_df.to_csv(DATA_DIR / "nba_defense_curry_history.csv", index=False)
            print(
                f"\n📁 History saved to: {DATA_DIR / 'nba_defense_curry_history.csv'}"
            )

            # Plot
            plot_history_radar(history_df)

    else:
        print(f"=== NBA Defense MVP Analysis ({SEASON}) ===")
        print(f"Target Players: {TARGET_PLAYERS}\n")

        # Call the encapsulated function
        base_df = analyze_season(SEASON)

        # =============================================================================
        # STEP 3: FILTER TARGET PLAYERS & RANKING
        # =============================================================================
        if not base_df.empty:
            # Filter for specific target players for the radar chart
            final_results = []
            for target in TARGET_PLAYERS:
                match_mask = (
                    base_df["PLAYER_NAME"].astype(str).str.contains(target, case=False)
                )
                match = base_df[match_mask]
                if not match.empty:
                    final_results.append(match.iloc[0])
                else:
                    print(f"   WARNING: Could not find data for {target}")

            results_df = pd.DataFrame(final_results)

            # =============================================================================
            # STEP 4: OUTPUT RESULTS
            # =============================================================================

            print("\n" + "=" * 80)
            print("                    🏀 NBA 防守统治力排行榜 🏀")
            print("=" * 80)
            print(
                "\n📊 【模型方法论】贝叶斯五维防守评估框架 (Bayesian 5-Dimension Defensive Evaluation)"
            )
            print(
                "   核心思想: 将防守拆解为5个独立维度，每个维度使用百分位排名(0-100%)作为先验概率，"
            )
            print(
                '             通过贝叶斯收缩调整后，使用效率模型框架加权平均得出"防守统治力"指数。'
            )
            print("\n📊 贝叶斯逻辑:")
            print("   • 先验分布: 每个维度的联盟分布作为先验 (Prior)")
            print("   • 似然函数: 球员实际表现数据作为似然 (Likelihood)")
            print("   • 后验估计: D_k = (n × raw_pct + C × 0.5) / (n + C)")
            print(f"   • 收缩常数: C = {BAYES_C} (样本量达到 C 时，数据权重 = 50%)")
            print("\n📊 效率模型框架 (Efficiency Model Framework):")
            print(
                "   • 投入层 (Input): D3 (活力指数) + D4 (防守球商) - 影响防守结果的努力/方式"
            )
            print(
                "   • 产出层 (Output): D1 (对位压制) + D2 (内外封锁) - 防守的直接结果"
            )
            print("   • 独立层: D5 (篮板保护) - 不参与效率计算")
            print("   • 回归模型: Expected_Output = α + β × Input_Score")
            print(
                "   • 效率系数: Efficiency = Actual_Output / Expected_Output (限制在 0.5-1.5)"
            )
            print(
                "   • EDI公式: EDI = [Output×Efficiency×(W1+W2) + Input×(W3+W4) + D5×W5] / 总权重"
            )
            print("   • 高效球员: 投入少但产出高 (天赋/防守智慧) → Efficiency > 1.0")
            print("   • 低效球员: 投入多但产出低 (空有努力) → Efficiency < 1.0")
            print("\n📊 角色相关性系数 (Role Relevance Coefficient):")
            print("   ┌─────────────┬───────────────────────────┬─────────────────┐")
            print("   │ 角色        │ D2 内外封锁权重           │ D5 篮板权重系数 │")
            print("   ├─────────────┼───────────────────────────┼─────────────────┤")
            print("   │ Guards (G)  │ 外线60% / 内线40%         │ 0.5 (衰减50%)   │")
            print("   │ Frontcourt  │ 外线40% / 内线60%         │ 1.0 (完整保留)  │")
            print("   └─────────────┴───────────────────────────┴─────────────────┘")
            print("\n📊 五维说明 (Value Added + Matchup Difficulty 改进版):")
            print(
                "   | D1 对位压制: PCT_PLUSMINUS × (1 + 0.3 × MD_Zscore) | MD调整：防守强人加分，躲避防守减分 |"
            )
            print(
                "   | D2 内外封锁: PLUSMINUS (护筐+三分，按角色加权) | 同样使用 Value Added 方法 |"
            )
            print(
                "   | D3 活力指数: 干扰球 + 造进攻犯规×2 + 干扰投篮 (Z-Score) | 数据源: LeagueHustleStatsPlayer |"
            )
            print(
                '   | D4 防守球商: (抢断 + 盖帽) / (犯规 + 1) | 效率型指标，惩罚"站桩型"低犯规球员 |'
            )
            print(
                "   | D5 篮板保护: 防守篮板率 (DREB%) | 数据源: LeagueDashPlayerStats (Advanced) |"
            )
            print("\n📊 Matchup Difficulty (MD) 对位难度调整:")
            print("   • 数据源: LeagueSeasonMatchups (每对攻防球员的回合数据)")
            print(
                "   • 计算方法: 每位进攻球员的 PTS/100回合 → 每位防守者的对位难度加权平均"
            )
            print("   • MD_Zscore > 0: 防守强得分手 (如Dillon Brooks防守Curry)")
            print("   • MD_Zscore < 0: 躲避强得分手 (如Curry防守弱侧翼)")
            print("   • D1调整: Adjusted_VA = PCT_PLUSMINUS × (1 + 0.3 × MD_Zscore)")
            print(
                "   • 示例: -5%压制差 + MD_Zscore=1 → -6.5% (奖励); MD_Zscore=-2 → -2% (惩罚)"
            )
            print("\n📊 Value Added 改进说明:")
            print("   • 解决问题: 防守低效投手(如底角射手)获得高分的问题")
            print("   • 改进方法: 使用 PCT_PLUSMINUS = D_FG_PCT - NORMAL_FG_PCT")
            print("   • 解读: 让对手比他预期命中率低多少，而非原始命中率")
            print(
                "   • 示例: 让50%命中率球员降到45% (VA=-5%) > 让40%球员维持40% (VA=0%)"
            )
            print("-" * 80)

            # League Top 5
            league_sorted = print_top_n(base_df, "联盟 (League)", n=5)

            # Positional Rankings
            guards = base_df[
                base_df["PLAYER_POSITION"].str.contains("G", na=False)
                & ~base_df["PLAYER_POSITION"].str.contains("F", na=False)
            ]  # 纯后卫
            frontcourt = base_df[base_df["ROLE"] == "Frontcourt"]  # 前场 (含F或C)

            # 后卫 Top 5 + Curry
            guards_sorted = print_top_n(
                guards,
                "后卫 (Guards)",
                n=5,
                extra_player="Stephen Curry",
                extra_label="Curry",
            )
            # 前场 Top 5 + Hansen Yang (仅 2025-26 赛季)
            if SEASON == "2025-26":
                print_top_n(
                    frontcourt,
                    "前场 (Frontcourt)",
                    n=5,
                    extra_player="Hansen",
                    extra_label="Hansen Yang",
                )
            else:
                print_top_n(frontcourt, "前场 (Frontcourt)", n=5)

            # Save all data to CSV
            base_df.to_csv(
                DATA_DIR / f"nba_defensive_all_players_{SEASON}.csv", index=False
            )
            print(
                f"\n📁 已保存: {DATA_DIR / f'nba_defensive_all_players_{SEASON}.csv'} (全部球员)"
            )

            if not results_df.empty:
                results_df = results_df.sort_values("EDI_Total", ascending=False)
                results_df.to_csv(
                    DATA_DIR / f"nba_defensive_mvp_results_{SEASON}.csv", index=False
                )
                print(
                    f"📁 已保存: {DATA_DIR / f'nba_defensive_mvp_results_{SEASON}.csv'} (目标球员)"
                )

            # =============================================================================
            # STEP 5: RADAR CHART VISUALIZATION (每个球员单独一个雷达图)
            # =============================================================================
            print("\n生成雷达图...")

            # Configure matplotlib for Chinese font support
            plt.rcParams["font.sans-serif"] = [
                "Microsoft YaHei",
                "SimHei",
                "DejaVu Sans",
            ]
            plt.rcParams["axes.unicode_minus"] = False

            # 联盟 Top 5 单独雷达图
            league_top5 = base_df.sort_values("EDI_Total", ascending=False).head(5)
            create_individual_radar_charts(
                league_top5,
                FIGURES_DIR / f"nba_defense_league_top5_{SEASON}.png",
                "联盟 Top 5 防守能力画像",
            )

            # 后卫 Top 5 + Curry 单独雷达图
            guards_sorted = guards.sort_values("EDI_Total", ascending=False)
            guard_top5 = guards_sorted.head(5)
            guards_total = len(guards_sorted)

            # 添加 Curry (如果不在 Top 5 中)
            curry_mask = guards_sorted["PLAYER_NAME"].str.contains(
                "Stephen Curry", case=False, na=False
            )
            if curry_mask.any():
                curry_row = guards_sorted[curry_mask].iloc[0:1]
                if (
                    not guard_top5["PLAYER_NAME"]
                    .str.contains("Stephen Curry", case=False, na=False)
                    .any()
                ):
                    curry_rank = (
                        guards_sorted["EDI_Total"] > curry_row["EDI_Total"].values[0]
                    ).sum() + 1
                    guard_top5 = pd.concat([guard_top5, curry_row])
                    curry_note = f" (含Curry #{curry_rank}/{guards_total})"
                else:
                    curry_note = ""
            else:
                curry_note = " (Curry未找到)"

            create_individual_radar_charts(
                guard_top5,
                FIGURES_DIR / f"nba_defense_guard_top5_{SEASON}.png",
                f"后卫 Top 5 防守能力画像{curry_note}",
            )

            # 前场 Top 5 单独雷达图 (2025-26 赛季包含 Hansen Yang)
            # 使用 Frontcourt (ROLE == "Frontcourt"，包含F和C)
            frontcourt_sorted = base_df[base_df["ROLE"] == "Frontcourt"].sort_values(
                "EDI_Total", ascending=False
            )
            frontcourt_top5 = frontcourt_sorted.head(5)
            frontcourt_total = len(frontcourt_sorted)

            # 2025-26 赛季添加 Hansen Yang (如果不在 Top 5 中)
            hansen_note = ""
            if SEASON == "2025-26":
                hansen_mask = frontcourt_sorted["PLAYER_NAME"].str.contains(
                    "Hansen", case=False, na=False
                )
                if hansen_mask.any():
                    hansen_row = frontcourt_sorted[hansen_mask].iloc[0:1]
                    if (
                        not frontcourt_top5["PLAYER_NAME"]
                        .str.contains("Hansen", case=False, na=False)
                        .any()
                    ):
                        hansen_rank = (
                            frontcourt_sorted["EDI_Total"]
                            > hansen_row["EDI_Total"].values[0]
                        ).sum() + 1
                        frontcourt_top5 = pd.concat([frontcourt_top5, hansen_row])
                        hansen_note = f" (含Hansen #{hansen_rank}/{frontcourt_total})"
                else:
                    hansen_note = " (Hansen未找到)"

            create_individual_radar_charts(
                frontcourt_top5,
                FIGURES_DIR / f"nba_defense_frontcourt_top5_{SEASON}.png",
                f"前场 Top 5 防守能力画像{hansen_note}",
            )

            print("\n[完成] 分析结束!")
