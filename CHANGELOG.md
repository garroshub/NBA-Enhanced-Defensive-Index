# NBA EDI 防守评估模型 - 开发日志

## [V0.7] - 2026-01-17

### 代码架构优化 + 3 分类位置系统 + Benchmark 框架

本版本完成了重大的代码重构和架构优化，引入 3 分类位置系统解决 DPOY 评估中的角色归类问题，并新增官方指标基准评估框架。

#### 1. 3 分类位置系统 (3-Category Role Classification)

**问题背景:**
- 原 2 分类系统 (Guards/Frontcourt) 无法正确处理"扫荡型内线" (Roamer)
- JJJ 和 Giannis 等摇摆位置球员被错误归入传统 Frontcourt 与 Gobert 竞争

**解决方案:**
```python
def classify_role_3cat(position, roamer_pct, threshold=0.15):
    """3 分类：Backcourt, Roamer, Frontcourt"""
    # 纯 G (PG, SG, G) -> Backcourt (永不为 Roamer)
    # 纯 C -> Frontcourt (即使 Roamer_Pct 高也不变)
    # 摇摆位置 (F, F-C, C-F, G-F, F-G) + Roamer_Pct >= 0.15 -> Roamer
```

**分类规则:**
| 位置 | Roamer_Pct | 分类 | 示例 |
|------|------------|------|------|
| G, PG, SG | 任意 | Backcourt | Marcus Smart |
| C | 任意 | Frontcourt | Rudy Gobert (即使 Roamer_Pct 高) |
| F, F-C, C-F | >= 0.15 | Roamer | JJJ, Giannis |
| F, PF, SF | < 0.15 | Frontcourt | 传统锋线 |

#### 2. 代码架构优化

**新增 `src/constants.py` 作为共享常量和函数的单一来源:**
```python
# 版本与评估赛季
EDI_VERSION = "0.7"
EVALUATION_SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]

# 优化后参数 (2025-01-17)
ROAMER_THRESHOLD = 0.15
SYNERGY_FACTOR = 0.5
SYNERGY_D1_THRESHOLD = 0.80
SYNERGY_D2_THRESHOLD = 0.75
ROAMER_WEIGHT_REDIST_OUTPUT = 0.3

# 共享函数
classify_role_3cat()      # 3 分类位置函数
bayesian_score()          # 贝叶斯收缩
sigmoid_availability()    # Sigmoid 可用性
```

**删除冗余代码:**
- 删除 `src/optimize_parameters.py` (与 tune_parameters.py 功能重复)
- 从 `tune_parameters.py` 删除 `classify_role_with_roamer` (45 行)
- 从 `benchmark_evaluation.py` 删除 `classify_position` (46 行)
- 统一使用 `constants.py` 中的 `classify_role_3cat`

#### 3. 新增 Benchmark 评估框架

**`src/benchmark_evaluation.py`** - EDI vs NBA 官方指标对比:
- 统一样本池 (GP >= 40, MPG >= 20)
- 3 分类位置内排名
- Spearman 秩相关计算

**`src/fetch_external.py`** - 外部数据获取:
- NBA 官方 API 数据 (DEF_RATING, DEF_WS)
- 缓存机制避免重复请求

**`src/tune_parameters.py`** - 参数优化器:
- 1575 种参数组合搜索
- 目标: 最小化 DPOY 平均排名 + 最大化 Recall@30

#### 4. 评估结果 (3 分类)

| 赛季 | DPOY | 分类 | EDI 排名 | DEF_RATING | DEF_WS |
|------|------|------|----------|------------|--------|
| 2019-20 | Giannis | Roamer | #2 | #1 | #1 |
| 2020-21 | Gobert | Frontcourt | #1 ✅ | #1 | #1 |
| 2021-22 | Smart | Backcourt | #4 | #7 | #2 |
| 2022-23 | JJJ | Roamer | #13 | #2 | #7 |
| 2023-24 | Gobert | Frontcourt | #1 ✅ | #1 | #1 |

**汇总指标:**
| 指标 | EDI | DEF_RATING | DEF_WS |
|------|-----|------------|--------|
| DPOY 平均排名 | 4.2 | 2.4 | 2.4 |
| Recall@30 (5 赛季) | **32/50** ✅ | 23/50 | 29/50 |

> EDI 在 Recall@30 上表现最佳，成功捕获最多 All-Defense 球员

#### 5. 文件变更

**新增文件:**
- `src/constants.py` - 共享常量和函数
- `src/benchmark_evaluation.py` - 基准评估框架
- `src/fetch_external.py` - 外部数据获取
- `src/tune_parameters.py` - 参数优化器
- `reports/benchmark_edi_vs_official.md` - 评估报告
- `docs/` - 文档目录

**删除文件:**
- `src/optimize_parameters.py` (功能合并到 tune_parameters.py)

**修改文件:**
- `src/nba_defense_mvp.py` - 优化参数
- `data/*.csv` - 重新生成 5 个赛季数据
- `figures/*.png` - 更新雷达图

---

## [V0.65] - 2026-01-16

### 新增功能: Roamer 动态权重调整系统

针对"扫荡型内线"(Roamer) 球员（如 Jaren Jackson Jr.）的防守风格进行系统性权重调整，解决此类球员因低篮板数据和三分防守样本不足而被低估的问题。

#### 核心问题
- JJJ 2022-23 获得 DPOY，但 EDI 模型排名仅 Frontcourt #25
- 原因：JJJ 是"扫荡型内线"，以盖帽和护筐为主，篮板由队友 Adams 负责
- 其三分防守样本极少（护筐者不防外线），3PT_Raw = 0.056 严重拖累 D2 分数

#### 解决方案

**1. Roamer 识别指标**
```python
Roamer_Index = BLK_per_36 / (DREB_PCT + 0.01)
Roamer_Pct = Roamer_Index 在 Frontcourt 中的百分位排名
```

**2. D2 外线权重动态调整** (`D2_EXT_ROAMER_K = 0.5`)
```python
# Roamer 球员降低外线防守权重（样本少且不代表其价值）
adjusted_ext = base_ext * (1 - D2_EXT_ROAMER_K * Roamer_Pct)
# JJJ (Roamer_Pct=1.0): 外线权重从 45% 降至 22.5%
# Gobert (Roamer_Pct≈0.05): 外线权重保持 ~43%
```

**3. D5 篮板权重动态调整** (`ROAMER_K = 0.3`)
```python
# Roamer 球员降低篮板权重（战术角色非篮板支柱）
W5 = W5 * (1 - ROAMER_K * Roamer_Pct)
# JJJ (Roamer_Pct=1.0): D5 权重降至 0.7
```

#### 效果验证

**JJJ 排名变化:**
| 赛季 | 之前 | 之后 | 改善 |
|------|------|------|------|
| 2021-22 | F #4 | **F #1** | +3 |
| 2022-23 (DPOY) | F #25 | **F #11** | +14 |
| 2023-24 | F #19 | **F #15** | +4 |

**整体模型效果:**
| 指标 | 之前 | 之后 |
|------|------|------|
| DPOY 平均位置排名 | 7.8 | **5.0** |
| 2023-24 Grade | B | **A** |

#### 技术细节

- 新增常量: `ROAMER_K = 0.3`, `D2_EXT_ROAMER_K = 0.5`
- Roamer_Pct 计算移至 D2 计算前，同时用于 D2 和 D5 调整
- `optimize_parameters.py` 添加 `d2_ext_roamer_k` 到搜索空间

#### 修改文件
- `src/nba_defense_mvp.py`: 添加 Roamer 动态权重调整逻辑
- `src/optimize_parameters.py`: 添加 d2_ext_roamer_k 参数
- `data/*.csv`: 重新生成 5 个赛季数据
- `figures/*.png`: 更新雷达图

---

## [V0.6] - 2026-01-15

### AvgRank 优化目标 + DPOY 评估 + 参数优化器

- 新的优化目标函数: `Loss = Avg_Pos_Rank_AllDefense + (DPOY_Pos_Rank - 1)`
- DPOY 对齐评估: 位置相对排名
- 参数优化器: Grid Search + Train/Val/Test 分割
- Sigmoid 可用性函数替代线性累积

---

## [V0.5] - 2026-01-14

### 三维评估系统 + D-RAPTOR 外部验证

- 三维评估框架: Tier Alignment, Candidate Pool Quality, Miss Analysis
- D-RAPTOR/DBPM 外部验证
- All-Defensive Team 数据缓存

---

## [V0.4] - 2026-01-13

### EDI 模型评估系统

- 添加 evaluation.py 评估模块
- run_evaluation.py 评估运行器
- 位置分组评估 (Pre-2023-24 vs Positionless Era)

---

## [V0.0] - 2026-01-12

### 初始版本

- 贝叶斯五维防守评估框架
- Value Added + Matchup Difficulty 改进
- 效率模型框架
