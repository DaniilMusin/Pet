# Pet Repository

This repository contains a collection of trading strategy experiments
primarily focused on Bitcoin futures. The code includes several variants of
adaptive strategies and helper scripts for backtesting and live trading.

## Structure

- **Top-level scripts** – Various strategy implementations such as
  `adaptive_strategy.py`, `enhanced_adaptive_strategy.py` and utilities to
  run backtests.
- **`algorithm_new/`** – A more organized trading module with a CLI entry
  point (`main.py`), optimization tools and data download helpers.
- **Data files** – Historical BTC data in CSV format used by the
  strategies.
- **Log files** – Output from previous runs.

## Usage

Most scripts are intended to be executed directly with Python 3. For
example:

```bash
python run_strategy.py
```

Some of the scripts rely on the CSV data files provided in this
repository. Ensure the required dependencies are installed (for example
`pandas`, `numpy`, `matplotlib`, and `TA-Lib`).

## Notes

Binary wheels, logs and CSV data are tracked in this repository for
convenience, but new such files will be ignored thanks to the included
`.gitignore`.
