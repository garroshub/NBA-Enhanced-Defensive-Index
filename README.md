# NBA Enhanced Defensive Index (EDI)

<div align="center">
  <img src="figures/EDI_Summary.png" alt="EDI Framework Summary" width="800" />
</div>

<br />

<div align="center">

[![Live Dashboard](https://img.shields.io/badge/🌐_View-Live_Dashboard-emerald?style=for-the-badge)](https://garroshub.github.io/NBA-Enhanced-Defensive-Index/)
[![Technical Report](https://img.shields.io/badge/📄_Read-Technical_Report-blue?style=for-the-badge)](reports/NBA_EDI_Defense_Report.md)

</div>

## Overview

EDI is a public-data NBA defense model. It breaks player defense into five separate scores instead of forcing everything into one black-box number.

1.  **🛡️ Shot Suppression (D1)**: Ability to lower opponent FG% vs expected.
2.  **🎯 Shot Profile (D2)**: Forcing inefficient shots (rim protection & 3PT prevention).
3.  **⚡ Hustle Index (D3)**: Activity metrics (deflections, contests, charges).
4.  **🧠 Defensive IQ (D4)**: Playmaking relative to fouling (Stocks/Fouls ratio).
5.  **⚓ Anchor/Rebounding (D5)**: Possession-ending ability.

Bayesian shrinkage is used to reduce small-sample noise, and the efficiency layer separates defensive activity from defensive payoff.

---

## Validation

Five-season comparison against official NBA defensive metrics:

| Metric | Top 10 | Top 20 | Top 30 |
|--------|--------|--------|--------|
| **EDI** | **19/50** | **25/50** | **32/50** |
| DEF_RATING | 9/50 | 16/50 | 23/50 |
| DEF_WS | 11/50 | 24/50 | 29/50 |

EDI leads the comparison at all three cutoffs.

---

## 2025-26 All-Defensive Check

The 2025-26 dashboard is updated through the completed 82-game regular season. For the All-Defensive comparison, EDI uses an awards-eligibility proxy: **65+ games played and 20+ minutes per game**, while retaining actual official selections to reflect league exceptions and credited-game rules that are not stored in the public season-level export.

| Award-eligible EDI cutoff | Official All-Defensive players captured |
|---------------------------|------------------------------------------|
| Top 10 | **6 / 10** |
| Top 15 | **7 / 10** |
| Top 30 | **10 / 10** |

Award-eligible EDI Top 10:

| Rank | Player | Team | EDI | Official result |
|------|--------|------|-----|-----------------|
| 1 | Chet Holmgren | OKC | 117.0 | First Team |
| 2 | Ausar Thompson | DET | 114.8 | First Team |
| 3 | Victor Wembanyama | SAS | 113.0 | First Team |
| 4 | Evan Mobley | CLE | 108.7 | Not selected |
| 5 | Brook Lopez | LAC | 108.2 | Not selected |
| 6 | Rudy Gobert | MIN | 104.8 | First Team |
| 7 | Dyson Daniels | ATL | 104.1 | Second Team |
| 8 | Cason Wallace | OKC | 102.8 | Second Team |
| 9 | Jaden McDaniels | MIN | 102.7 | Not selected |
| 10 | Draymond Green | GSW | 100.4 | Not selected |

All ten official All-Defensive selections rank inside EDI's award-eligible Top 30.

---

## Project Layout

- `src/`: Python model, evaluation, and export scripts
- `web/`: Next.js dashboard
- `reports/`: long-form write-up and validation notes
- `figures/`: summary images used in the docs

## Local Run

```bash
pip install -r requirements.txt
python src/nba_defense_mvp.py 2025-26
python src/web_export.py
cd web
npm install
npm run dev
```

---

## Update Flow

The repository is refreshed with current NBA tracking data, then the web bundle is rebuilt for GitHub Pages.

---

Academic project. Not affiliated with the NBA.

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=garroshub/NBA-Enhanced-Defensive-Index&type=Date)](https://www.star-history.com/#garroshub/NBA-Enhanced-Defensive-Index&Date)
