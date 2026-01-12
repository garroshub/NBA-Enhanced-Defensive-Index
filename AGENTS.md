# AGENTS.md — NBA EDI Defense Project

> Guidelines for AI coding agents working on this repository.

## Project Overview

Python-based NBA defensive evaluation system using a Bayesian 5-Dimension framework (EDI).
Outputs CSV rankings, radar charts, and validation reports against All-Defensive Team selections.

## Repository Structure

```
src/
├── nba_defense_mvp.py   # Main EDI analysis script (generates data + figures)
├── run_evaluation.py    # CLI for model evaluation against All-Defense ground truth
├── evaluation.py        # Three-dimensional evaluation metrics (dataclasses)
├── data_fetcher.py      # All-Defensive Team cache + nba_api utilities
├── external_metrics.py  # External validation (D-RAPTOR, DBPM)
└── __init__.py
tests/
├── test_evaluation.py   # Coverage & correlation tests
└── test_data_fetcher.py # Award data fetch tests
data/                    # Generated CSV outputs (do not edit manually)
figures/                 # Generated PNG charts (do not edit manually)
reports/                 # Technical reports (Markdown/docx)
```

## Build / Lint / Test Commands

```bash
# Generate EDI data for a season
python src/nba_defense_mvp.py 2023-24

# Evaluate model against All-Defensive Team
python src/run_evaluation.py 2023-24
python src/run_evaluation.py --all              # All 5 seasons
python src/run_evaluation.py --external 2021-22 # With D-RAPTOR validation

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_evaluation.py -v

# Run a single test class or method
pytest tests/test_evaluation.py::TestCalculateCoverage -v
pytest tests/test_evaluation.py::TestCalculateCoverage::test_precision_at_k_basic -v

# Lint with ruff (if installed)
ruff check src/ tests/
ruff format src/ tests/
```

## Code Style Guidelines

### Imports

Order: stdlib → third-party → local. Separate groups with blank line.

```python
import io
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import numpy as np
from scipy import stats

from data_fetcher import get_all_defensive_teams
```

### Formatting

- **Line length**: 88 characters (Black/Ruff default)
- **Indentation**: 4 spaces
- **Quotes**: Double quotes for strings
- **Trailing commas**: Yes, in multi-line structures

### Type Hints

Use type hints for function signatures. Prefer modern syntax (Python 3.10+):

```python
def calculate_coverage(df: pd.DataFrame, top_n: int = 10) -> dict[str, float]:
    ...

def get_player_rank(name: str) -> int | None:
    ...
```

### Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Functions | snake_case | `calculate_tier_alignment` |
| Variables | snake_case | `player_ranks`, `avg_rank` |
| Constants | UPPER_SNAKE | `MIN_GAMES_2024`, `BAYES_C` |
| Classes | PascalCase | `TierAlignment`, `SeasonEvaluation` |
| Dataclasses | PascalCase | `@dataclass class CandidatePoolQuality` |

### Docstrings

Use triple-quoted docstrings for modules, classes, and public functions:

```python
def bayesian_score(raw_pct: float, n: int, c: int = 50) -> tuple[float, float]:
    """Apply Bayesian shrinkage to raw percentile.

    Args:
        raw_pct: Raw percentile (0-1)
        n: Sample size
        c: Shrinkage constant (default 50)

    Returns:
        Tuple of (shrunk_score, confidence_weight)
    """
```

### Error Handling

- Graceful degradation for network failures (return empty DataFrame, log error)
- Never crash on missing optional data (e.g., external metrics)
- Use explicit error messages with context

```python
try:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
except requests.RequestException as e:
    print(f"Error fetching RAPTOR data: {e}")
    return pd.DataFrame()
```

### Windows Compatibility

Scripts set UTF-8 encoding for Chinese character support:

```python
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
```

Do not remove this wrapper.

## Data Contracts

### CSV Output Columns (required)

Model output `data/nba_defensive_all_players_{season}.csv` must include:

- Identity: `PLAYER_NAME`, `PLAYER_ID`, `PLAYER_POSITION`
- Scores: `EDI_Total`, `D1_Score` ... `D5_Score`
- Weights: `W1` ... `W5`
- Efficiency: `Efficiency`, `Expected_Output`, `Actual_Output`

Changing column names requires updating `evaluation.py` and `run_evaluation.py`.

### Season String Format

Always use `"YYYY-YY"` format: `"2023-24"`, `"2022-23"`, etc.

## Testing Guidelines

- Tests live in `tests/` with `test_*.py` naming
- Use pytest with class-based organization
- Mock network calls; do not make real API requests in tests
- Test edge cases: empty data, perfect scores, zero coverage

```python
class TestCalculateCoverage:
    def test_precision_at_k_basic(self):
        ...
    def test_zero_coverage(self):
        ...
```

## Network & Rate Limits

- `nba_api` calls include `time.sleep(0.6)` between requests
- External fetches use `timeout=30` and handle failures gracefully
- `fetch_raptor_data()` uses `@lru_cache` to avoid repeated downloads
- Basketball-Reference may return 403; allow DBPM to be unavailable

## Evaluation Eras

| Era | Logic |
|-----|-------|
| Pre-2023-24 | Position-based pools (4G + 4F + 2C) |
| 2023-24+ | Positionless top 10, 65-game minimum |

## Common Pitfalls

1. **Do not overfit to All-Defense**: Evaluation metrics diagnose; they're not training targets
2. **Do not bulk-regenerate data/figures** without explicit need
3. **Preserve rate-limit sleeps** in `nba_api` calls
4. **Handle `nba_api` column name changes** with fallback logic, not hard crashes
5. **Run `pytest`** after modifying `evaluation.py` or `run_evaluation.py`
