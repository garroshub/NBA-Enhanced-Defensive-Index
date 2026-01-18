# NBA Enhanced Defensive Index (EDI)

<div align="center">
  <img src="figures/EDI_Summary.png" alt="EDI Framework Summary" width="800" />
</div>

<br />

<div align="center">

[![Live Dashboard](https://img.shields.io/badge/🌐_View-Live_Dashboard-emerald?style=for-the-badge)](https://garroshub.github.io/NBA-Defensive-Index/)
[![Technical Report](https://img.shields.io/badge/📄_Read-Technical_Report-blue?style=for-the-badge)](reports/NBA_EDI_Defense_Report.md)

</div>

---

## Abstract

We present the **Enhanced Defensive Index (EDI)**, a novel framework for evaluating NBA defensive performance. Unlike traditional opaque metrics that output a single number, EDI decomposes defense into five distinct dimensions:

1.  **🛡️ Shot Suppression (D1)**: Ability to lower opponent FG% vs expected.
2.  **🎯 Shot Profile (D2)**: Forcing inefficient shots (rim protection & 3PT prevention).
3.  **⚡ Hustle Index (D3)**: Activity metrics (deflections, contests, charges).
4.  **🧠 Defensive IQ (D4)**: Playmaking relative to fouling (Stocks/Fouls ratio).
5.  **⚓ Anchor/Rebounding (D5)**: Possession-ending ability.

We employ **Bayesian shrinkage** to mitigate small sample bias and an **Efficiency Model** to distinguish effort from impact. Results demonstrate that EDI achieves parity with leading advanced metrics while offering superior interpretability.

## Repository Structure

*   `src/`: Python implementation of the EDI model and data pipeline.
*   `web/`: Next.js frontend for the interactive dashboard.
*   `reports/`: Technical documentation and validation studies.

## Automation

This repository is self-updating. A GitHub Action runs daily to:
1.  Fetch the latest NBA tracking data.
2.  Recalculate EDI scores using the Bayesian model.
3.  Rebuild and deploy the dashboard to GitHub Pages.

---

*Academic Project | Not affiliated with the NBA*
