# NBA Enhanced Defensive Index (EDI)

<div align="center">
  <img src="figures/EDI_Summary.png" alt="EDI Framework Summary" width="800" />
</div>

<br />

<div align="center">

[![Live Dashboard](https://img.shields.io/badge/🌐_View-Live_Dashboard-emerald?style=for-the-badge)](https://garroshub.github.io/NBA-Enhanced-Defensive-Index/)
[![Technical Report](https://img.shields.io/badge/📄_Read-Technical_Report-blue?style=for-the-badge)](reports/NBA_EDI_Defense_Report.md)
[![Discussions](https://img.shields.io/badge/💬_Join-Discussions-0b5fff?style=for-the-badge)](https://github.com/garroshub/NBA-Enhanced-Defensive-Index/discussions)

</div>

---

## Abstract

We present the **Enhanced Defensive Index (EDI)**, a novel framework for evaluating NBA defensive performance. Unlike traditional opaque metrics that output a single number, EDI decomposes defense into five distinct dimensions:

1.  **🛡️ Shot Suppression (D1)**: Ability to lower opponent FG% vs expected.
2.  **🎯 Shot Profile (D2)**: Forcing inefficient shots (rim protection & 3PT prevention).
3.  **⚡ Hustle Index (D3)**: Activity metrics (deflections, contests, charges).
4.  **🧠 Defensive IQ (D4)**: Playmaking relative to fouling (Stocks/Fouls ratio).
5.  **⚓ Anchor/Rebounding (D5)**: Possession-ending ability.

We employ **Bayesian shrinkage** to mitigate small sample bias and an **Efficiency Model** to distinguish effort from impact.

---

## What Makes EDI Different

What makes this framework different from traditional defensive metrics:

### 🔬 Mechanism-first, not residual-first
Defense is modeled as a multi-dimensional, interpretable structure rather than compressed into a single residual-based impact number.

### 📊 Bayesian and uncertainty-aware
The framework emphasizes posterior inference and shrinkage instead of relying on fragile point estimates, improving stability in small-sample and early-season contexts.

### 🔍 Diagnosis over ranking
The goal is to explain *why* defensive impact emerges, not just to order players by a scalar score.

### 🎯 Contextual and role-aware
Defensive value is mapped through roles and efficiency (effort versus outcome), distinguishing disciplined deterrence from high-variance gambling, and avoiding position-invariant assumptions.

---

## Validation

EDI was validated against NBA official metrics across 5 seasons (2019-2024):

**All-Defensive Team Coverage**

| Metric | Top 10 | Top 20 | Top 30 |
|--------|--------|--------|--------|
| **EDI** | **19/50** | **25/50** | **32/50** |
| DEF_RATING | 9/50 | 16/50 | 23/50 |
| DEF_WS | 11/50 | 24/50 | 29/50 |

EDI achieves the **highest coverage** of All-Defensive Team selections across all thresholds.

---

## Limitations & Future Work

As a long-time NBA fan who developed this framework independently, I want to be transparent about its current limitations:

- **No professional coaching input**: This model has not been reviewed or validated by professional basketball coaches or scouts. Their insights on defensive schemes, rotations, and matchup strategies could significantly improve the framework.

- **Missing premium data sources**: Due to cost constraints, I was unable to incorporate paid defensive metrics like **D-LEBRON** and **DEPM** (Defensive Estimated Plus-Minus) into the validation comparisons. These metrics may capture aspects of defense that EDI currently misses.

- **Tracking data limitations**: The model relies on publicly available NBA tracking data, which may have measurement noise and does not capture off-ball positioning as precisely as proprietary systems.

**I welcome contributions and feedback from the basketball analytics community.** If you have professional insights, access to additional data sources, or suggestions for model improvements, please open an issue or reach out. The goal is to make EDI a genuinely useful tool for understanding defensive impact.

---

## Repository Structure

*   `src/`: Python implementation of the EDI model and data pipeline.
*   `web/`: Next.js frontend for the interactive dashboard.
*   `reports/`: Technical documentation and validation studies.

## Updates

This repository is updated weekly with the latest NBA tracking data. Each update includes:
1.  Recalculated EDI scores using the Bayesian model.
2.  Rebuilt and deployed dashboard to GitHub Pages.

---

## Star History

<a href="https://star-history.com/#garroshub/NBA-Enhanced-Defensive-Index&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=garroshub/NBA-Enhanced-Defensive-Index&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=garroshub/NBA-Enhanced-Defensive-Index&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=garroshub/NBA-Enhanced-Defensive-Index&type=Date" />
 </picture>
</a>

---

*Academic Project | Not affiliated with the NBA*
