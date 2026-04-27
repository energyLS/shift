from pathlib import Path
import sys

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_SCRIPTS = REPO_ROOT / "workflow" / "scripts"

for path in (REPO_ROOT, WORKFLOW_SCRIPTS):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


@pytest.fixture
def steel_tech_costs() -> pd.Series:
    """Synthetic tech-cost data that mirrors the steel electricity chain."""
    rows = [
        ("Alkaline electrolyzer large size", "electricity-input", 1.38),
        ("hydrogen direct iron reduction furnace", "hydrogen-input", 2.1),
        ("hydrogen direct iron reduction furnace", "electricity-input", 1.03),
        ("electric arc furnace", "electricity-input", 0.6395),
    ]

    index = pd.MultiIndex.from_tuples(
        [(technology, parameter) for technology, parameter, _ in rows],
        names=["technology", "parameter"],
    )
    values = [value for _, _, value in rows]
    return pd.Series(values, index=index, dtype=float)


@pytest.fixture
def steel_minimal_case() -> dict:
    """Hand-checkable supply-stage numbers for a 1 t/h steel test case."""
    demand_mt_per_year = 0.00876
    hourly_demand_tph = demand_mt_per_year * 1e6 / 8760

    return {
        "product": "steel",
        "demand_mt_per_year": demand_mt_per_year,
        "hourly_demand_tph": hourly_demand_tph,
        "annual_demand_t": hourly_demand_tph * 8760,
        "expected_steel_electricity_mwh_per_t": 4.6275,
        "expected_annual_electricity_mwh": hourly_demand_tph * 8760 * 4.6275,
    }


@pytest.fixture
def toy_trade_case() -> dict:
    """Minimal trade-stage numbers that can be checked by hand."""
    return {
        "offers": {
            "A": 100.0,
            "B": 150.0,
        },
        "transport_costs": {
            ("A", "B"): 10.0,
            ("B", "B"): 0.0,
        },
        "expected_choice": "A",
        "expected_delivered_cost": 110.0,
    }
