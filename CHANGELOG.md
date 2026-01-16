# NBA EDI 防守评估模型 - 开发日志

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
